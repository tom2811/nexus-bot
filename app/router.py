"""Hybrid routing for RAG and database tools.

Routes questions to appropriate handlers based on intent detection.
"""

import re
from enum import Enum
from typing import Any, Optional

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
            return self._handle_tool_only(question, **kwargs)
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
        if kwargs.get("product_context"):
            return self._handle_product_context(
                question,
                kwargs["product_context"],
                kwargs.get("size"),
            )
        
        # Try to extract parameters from kwargs or question
        # This is simplified - in production, use NER or structured input
        
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
                # Inventory result
                if result["available"]:
                    variants = result.get("variants", [])
                    in_stock = [v for v in variants if v["in_stock"]]
                    context_parts.append(
                        f"{result['product_name']} is available. "
                        f"{len(in_stock)} variants in stock."
                    )
                else:
                    context_parts.append(f"{result.get('product_name', 'Product')} is out of stock.")
            elif "products" in result:
                # Search result
                products = result["products"]
                context_parts.append(f"Found {len(products)} products.")
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

    def _handle_product_context(
        self,
        question: str,
        product_context: dict,
        size: Optional[str] = None,
    ) -> dict[str, Any]:
        """Answer inventory/product questions from caller-supplied catalog data."""
        answer = self._format_product_context_answer(question, product_context, size)
        return {
            "answer": answer,
            "sources": [],
            "route_type": RouteType.TOOL_ONLY.value,
            "rag_used": False,
            "tools_used": ["caller_catalog"],
            "tool_results": [product_context],
        }

    def _format_product_context_answer(
        self,
        question: str,
        ctx: dict,
        size: Optional[str] = None,
    ) -> str:
        name = ctx.get("name") or "this product"
        brand = ctx.get("brand") or ""
        label = f"{brand} {name}".strip()
        sizes = ctx.get("availableSizes") or ctx.get("available_sizes") or []
        price = ctx.get("retailPrice", ctx.get("retail_price"))
        in_stock = ctx.get("inStock") if "inStock" in ctx else ctx.get("in_stock")

        target = size
        if not target:
            match = re.search(r"\b\d+(?:\.\d+)?\b", question)
            target = match.group(0) if match else None

        size_labels = [str(s) for s in sizes]
        price_bit = f" at ${float(price):.2f}" if price is not None else ""

        if target:
            if str(target) in size_labels:
                return f"Yes, size {target} for {label} is in stock{price_bit}."
            available = ", ".join(size_labels) or "none"
            return (
                f"Size {target} for {label} is currently out of stock. "
                f"Available sizes: {available}."
            )

        if in_stock is False or not size_labels:
            return f"{label} is currently out of stock."

        available = ", ".join(size_labels)
        return f"{label} is in stock{price_bit}. Available sizes: {available}."
    
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
