"""Hybrid routing for RAG and database tools.

Routes questions to appropriate handlers based on intent detection.
"""

import json
import re
import time
from enum import Enum
from typing import Any, Optional

# Filler words stripped when reducing a free-text question to a search query.
_SEARCH_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "do", "does", "did", "you", "your",
    "i", "my", "me", "we", "have", "has", "had", "can", "could", "would",
    "in", "on", "of", "for", "to", "with", "and", "or", "any", "some", "this",
    "that", "these", "those", "it", "its", "what", "which", "how", "much",
    "many", "tell", "about", "status", "order", "track", "check", "search",
    "find", "product", "products", "available", "availability", "stock",
    "size", "sizes", "price", "cost", "buy", "get",
}

from app import tracing
from app.contract import classify_legacy_context, resolve_grounding
from app.database import DatabaseConnection
from app.llm import LLMService
from app.rag import RAGPipeline
from app.tools import InventoryTool, OrderStatusTool, ProductSearchTool, ToolError
from app.vector_store import VectorStore


class RouteType(Enum):
    """Type of routing for a question."""
    RAG_ONLY = "rag_only"
    TOOL_ONLY = "tool_only"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class HybridRouter:
    """Routes questions to RAG, tools, or both based on intent."""
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        db: DatabaseConnection,
        llm: Optional[LLMService] = None
    ):
        """Initialize hybrid router.
        
        Args:
            rag_pipeline: RAG pipeline for document queries
            db: Database connection for tools
            llm: Optional LLM service for synthesis
        """
        self.rag_pipeline = rag_pipeline
        self.db = db
        self.llm = llm or LLMService()
        
        # Initialize tools
        self.inventory_tool = InventoryTool(db)
        self.search_tool = ProductSearchTool(db)
        self.order_tool = OrderStatusTool(db)
    
    def detect_intent(self, question: str) -> RouteType:
        """Detect the intent of a question.
        
        Uses keyword matching to determine if question needs:
        - RAG (policy, shipping, return, guide, etc.)
        - Tools (inventory, stock, order, available, etc.)
        - Both (hybrid)
        
        Args:
            question: User's question
            
        Returns:
            RouteType indicating routing strategy
        """
        question_lower = question.lower()
        
        # Keywords for different intents
        rag_keywords = [
            "policy", "policies", "return", "refund", "shipping", "delivery",
            "guide", "how to", "what is", "explain", "describe", "tell me about"
        ]
        
        tool_keywords = [
            "available", "in stock", "stock", "inventory", "check",
            "order status", "order", "track", "search", "find product",
            "price", "cost"
        ]
        
        # Check for keywords
        has_rag_intent = any(keyword in question_lower for keyword in rag_keywords)
        has_tool_intent = any(keyword in question_lower for keyword in tool_keywords)
        
        if has_rag_intent and has_tool_intent:
            return RouteType.HYBRID
        elif has_rag_intent:
            return RouteType.RAG_ONLY
        elif has_tool_intent:
            return RouteType.TOOL_ONLY
        else:
            # Default to RAG for general questions
            return RouteType.RAG_ONLY
    
    def query(self, question: str, **kwargs) -> dict[str, Any]:
        """Route question, generate answer, and emit a per-request trace."""
        api_version = kwargs.pop("api_version", "1")
        trace = tracing.start_trace(api_version=api_version)
        started = time.perf_counter()
        try:
            result = self._dispatch(question, **kwargs)
        except Exception as e:
            trace.error = f"{type(e).__name__}: {e}"
            trace.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            tracing.emit_trace(trace)
            raise
        trace.route = result.get("route_type")
        trace.tools = list(result.get("tools_used", []))
        trace.retrieval = result.get("retrieval", [])
        trace.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        tracing.emit_trace(trace)
        result["trace"] = trace.to_dict()
        return result

    def _dispatch(self, question: str, **kwargs) -> dict[str, Any]:
        """Route question and generate answer.

        Args:
            question: User's question
            **kwargs: Additional routing hints (route_type, product_id, etc.)

        Returns:
            Dictionary with answer, sources, and routing info:
            {
                "answer": str,
                "sources": list,
                "route_type": str,
                "rag_used": bool,
                "tools_used": list
            }
        """
        # Detect route type (or use provided)
        route_type = kwargs.get("route_type")
        if route_type is None:
            route_type = self.detect_intent(question)
        elif isinstance(route_type, str):
            route_type = RouteType(route_type)
        
        tenant_id = kwargs.get("tenant_id", "general")

        # Route to appropriate handler
        if route_type == RouteType.RAG_ONLY:
            return self._handle_rag_only(question, tenant_id=tenant_id)
        elif route_type == RouteType.TOOL_ONLY:
            result = self._handle_tool_only(question, **kwargs)
            if not result["tools_used"]:
                # No structured data reached a tool (e.g. message-only payload).
                # Answer from the tenant's documents instead of giving up.
                return self._handle_rag_only(question, tenant_id=tenant_id)
            return result
        elif route_type == RouteType.HYBRID:
            kwargs["tenant_id"] = tenant_id
            return self._handle_hybrid(question, **kwargs)
        else:
            # Fallback to RAG
            return self._handle_rag_only(question, tenant_id=tenant_id)
    
    def _handle_rag_only(self, question: str, tenant_id: str = "general") -> dict[str, Any]:
        """Handle RAG-only questions.
        
        Args:
            question: User's question
            tenant_id: Tenant collection identifier
            
        Returns:
            Result dictionary with RAG answer
        """
        result = self.rag_pipeline.query(question, collection_name=tenant_id)
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "retrieval": result.get("retrieval", []),
            "context": result.get("context"),
            "route_type": RouteType.RAG_ONLY.value,
            "rag_used": True,
            "tools_used": []
        }
    
    def _handle_tool_only(self, question: str, **kwargs) -> dict[str, Any]:
        """Handle tool-only questions.
        
        Args:
            question: User's question
            **kwargs: Tool parameters (product_id, order_id, etc.)
            
        Returns:
            Result dictionary with tool response
        """
        tools_used = []
        tool_results = []

        # Host apps (e.g. Kickskraze) pass live catalog data so this bot
        # does not need their product IDs in the local Postgres schema.
        # The declared shape (grounding.kind) decides how data is rendered;
        # unknown kinds are rejected upstream, so keys are never guessed.
        if kwargs.get("grounding"):
            g = kwargs["grounding"]
            resolved = resolve_grounding(g["kind"], g["data"])
            return self._handle_grounding(question, resolved, kwargs.get("size"))

        if kwargs.get("product_context"):
            g = classify_legacy_context(kwargs["product_context"])
            resolved = resolve_grounding(g.kind, g.data)
            return self._handle_grounding(question, resolved, kwargs.get("size"))

        # Callers may send a free-text message with no structured fields.
        # Reduce it to keywords so the search tool still runs.
        if not any(k in kwargs for k in ("product_id", "search_query", "order_id")):
            derived = self._derive_search_query(question)
            if derived:
                kwargs["search_query"] = derived

        # Check inventory if product_id provided
        if "product_id" in kwargs:
            try:
                result = self.inventory_tool.check_inventory(
                    product_id=kwargs["product_id"],
                    size=kwargs.get("size"),
                    color=kwargs.get("color")
                )
                tools_used.append("inventory")
                tool_results.append(result)
            except ToolError as e:
                tool_results.append({"error": str(e)})
        
        # Search products if query provided
        if "search_query" in kwargs:
            try:
                results = self.search_tool.search_products(
                    query=kwargs["search_query"],
                    category=kwargs.get("category"),
                    max_results=kwargs.get("max_results", 10)
                )
                if results:
                    tools_used.append("search")
                    tool_results.append({"products": results})
            except ToolError as e:
                tool_results.append({"error": str(e)})
        
        # Check order status if order_id provided
        if "order_id" in kwargs:
            try:
                result = self.order_tool.get_order_status(
                    order_id=kwargs["order_id"],
                    customer_email=kwargs.get("customer_email")
                )
                tools_used.append("order_status")
                tool_results.append(result)
            except ToolError as e:
                tool_results.append({"error": str(e)})
        
        # Generate natural language answer from tool results
        answer = self._synthesize_tool_response(question, tool_results, tools_used)
        
        return {
            "answer": answer,
            "sources": [],
            "route_type": RouteType.TOOL_ONLY.value,
            "rag_used": False,
            "tools_used": tools_used,
            "tool_results": tool_results
        }
    
    def _handle_hybrid(self, question: str, **kwargs) -> dict[str, Any]:
        """Handle hybrid questions requiring both RAG and tools.
        
        Args:
            question: User's question
            **kwargs: Tool parameters and routing hints
            
        Returns:
            Result dictionary combining RAG and tool responses
        """
        tenant_id = kwargs.get("tenant_id", "general")
        rag_result = self.rag_pipeline.query(question, collection_name=tenant_id)
        
        # Get tool responses (caller catalog context or local DB tools)
        tool_result = self._handle_tool_only(question, **kwargs)
        
        # Combine context
        combined_context = self._combine_rag_and_tools(
            question,
            rag_result,
            tool_result
        )
        
        # Generate final answer
        answer = self._synthesize_hybrid_response(
            question,
            combined_context,
            rag_result,
            tool_result
        )
        
        return {
            "answer": answer,
            "sources": rag_result.get("sources", []),
            "retrieval": rag_result.get("retrieval", []),
            "context": rag_result.get("context"),
            "route_type": RouteType.HYBRID.value,
            "rag_used": True,
            "tools_used": tool_result["tools_used"],
            "tool_results": tool_result.get("tool_results", [])
        }
    
    def _synthesize_tool_response(
        self,
        question: str,
        tool_results: list,
        tools_used: list
    ) -> str:
        """Generate natural language answer from tool results.
        
        Args:
            question: Original question
            tool_results: Results from tools
            tools_used: Names of tools used
            
        Returns:
            Natural language answer
            
        Raises:
            ConnectionError: If LLM is unavailable
            TimeoutError: If LLM request times out
        """
        if not tool_results:
            return "I couldn't find any information to answer your question."
        
        # Build context from tool results
        context_parts = []
        
        for i, result in enumerate(tool_results):
            if "error" in result:
                context_parts.append(f"Error: {result['error']}")
            elif "available" in result:
                # Inventory result: keep per-size detail, otherwise
                # size-specific questions can't be answered.
                variants = result.get("variants", [])
                lines = []
                for v in variants:
                    line = f"- size {v.get('size')}"
                    if v.get("color"):
                        line += f" ({v['color']})"
                    line += f": {'in stock' if v.get('in_stock') else 'out of stock'}"
                    if v.get("quantity") is not None:
                        line += f", qty {v['quantity']}"
                    lines.append(line)
                header = f"{result.get('product_name', 'Product')}"
                if result.get("brand"):
                    header += f" by {result['brand']}"
                header += " is in stock" if result.get("available") else " is out of stock"
                if result.get("price") is not None:
                    header += f" (${float(result['price']):.2f})"
                header += "."
                context_parts.append(header + ("\n" + "\n".join(lines) if lines else ""))
            elif "products" in result:
                # Search result: include names so the model can summarize.
                products = result["products"]
                if not products:
                    context_parts.append("No matching products found.")
                else:
                    listing = "; ".join(
                        f"{p.get('name', 'Product')}"
                        + (f" by {p['brand']}" if p.get("brand") else "")
                        + (f" (${float(p['price']):.2f})" if p.get("price") is not None else "")
                        for p in products
                    )
                    context_parts.append(f"Found {len(products)} product(s): {listing}.")
            elif "order_id" in result:
                # Order status result
                context_parts.append(
                    f"Order #{result['order_id']} status: {result['status']}"
                )
        
        context = "\n".join(context_parts)
        
        # Generate natural answer using LLM
        prompt = f"""Based on the following information, answer the user's question naturally.

Question: {question}

Information:
{context}

Provide a clear, helpful answer:"""
        
        # Let errors propagate - caller decides how to handle
        return self.llm.generate(prompt, temperature=0.7, timeout=30)

    def _handle_grounding(
        self,
        question: str,
        resolved,
        size: Optional[str] = None,
    ) -> dict[str, Any]:
        """Answer from validated, caller-declared grounding data.

        Kinds with a registered renderer (product, order) answer
        deterministically; kinds without one are answered by the LLM over the
        raw JSON.
        """
        if resolved.render is not None:
            answer = resolved.render(question, resolved.data, size)
        else:
            answer = self._synthesize_context_response(question, resolved.data)

        return {
            "answer": answer,
            "sources": [],
            "route_type": RouteType.TOOL_ONLY.value,
            "rag_used": False,
            "tools_used": [resolved.tool],
            "tool_results": [resolved.data],
        }

    def _synthesize_context_response(self, question: str, context: dict) -> str:
        """Answer a question from an arbitrary caller-supplied data object."""
        prompt = f"""Answer the user's question using only the data below.
If the data does not contain the answer, say you don't have that information.

Question: {question}

Data:
{json.dumps(context, indent=2, default=str)}

Answer:"""
        return self.llm.generate(prompt, temperature=0.3, timeout=30).strip()

    def _derive_search_query(self, question: str) -> Optional[str]:
        """Reduce a free-text question to keywords for the product search tool."""
        words = re.sub(r"[^a-z0-9\s]", " ", question.lower()).split()
        terms = [w for w in words if w not in _SEARCH_STOPWORDS and not w.isdigit()]
        return " ".join(terms).strip() or None

    def _combine_rag_and_tools(
        self,
        question: str,
        rag_result: dict,
        tool_result: dict
    ) -> str:
        """Combine RAG and tool results into unified context.
        
        Args:
            question: Original question
            rag_result: Result from RAG pipeline
            tool_result: Result from tools
            
        Returns:
            Combined context string
        """
        parts = []
        
        # Add document context
        if rag_result.get("context_used"):
            parts.append(f"Document Information:\n{rag_result['context_used']}")
        
        # Add tool results
        if tool_result.get("tool_results"):
            parts.append(f"Database Information:\n{tool_result['tool_results']}")
        
        return "\n\n".join(parts)
    
    def _synthesize_hybrid_response(
        self,
        question: str,
        combined_context: str,
        rag_result: dict,
        tool_result: dict
    ) -> str:
        """Generate final answer combining RAG and tool information.
        
        Args:
            question: Original question
            combined_context: Combined context from RAG and tools
            rag_result: RAG pipeline result
            tool_result: Tool result
            
        Returns:
            Final synthesized answer
            
        Raises:
            ConnectionError: If LLM is unavailable
            TimeoutError: If LLM request times out
        """
        prompt = f"""Answer the user's question using both document and database information.

Question: {question}

{combined_context}

Provide a comprehensive answer that addresses all aspects of the question:"""
        
        # Let errors propagate - caller decides how to handle
        return self.llm.generate(prompt, temperature=0.7, timeout=30)
