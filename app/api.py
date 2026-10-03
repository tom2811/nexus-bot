"""API endpoints for Nexus-bot assistant."""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.database import DatabaseConnection
from app.rag import RAGPipeline
from app.router import HybridRouter, RouteType
from app.vector_store import VectorStore


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for assistant queries."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question or message",
        examples=["What is the return policy?"]
    )
    route_type: Optional[str] = Field(
        None,
        description="Force specific routing (rag_only, tool_only, hybrid)",
        examples=["rag_only", "tool_only", "hybrid"]
    )
    product_id: Optional[int] = Field(
        None,
        description="Product ID for inventory/product queries",
        examples=[1]
    )
    size: Optional[str] = Field(
        None,
        description="Product size for inventory queries",
        examples=["10"]
    )
    order_id: Optional[int] = Field(
        None,
        description="Order ID for order status queries",
        examples=[1001]
    )
    search_query: Optional[str] = Field(
        None,
        description="Search query for product search",
        examples=["Nike"]
    )


class Source(BaseModel):
    """Source information for retrieved documents."""
    
    source: str = Field(..., description="Source document filename")
    chunk_id: str = Field(..., description="Chunk identifier")


class QueryResponse(BaseModel):
    """Response model for assistant queries."""
    
    answer: str = Field(..., description="Assistant's answer")
    route_type: str = Field(..., description="Routing type used")
    rag_used: bool = Field(..., description="Whether RAG was used")
    tools_used: list[str] = Field(default_factory=list, description="Tools used")
    sources: list[Source] = Field(default_factory=list, description="RAG sources")
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw tool results"
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")


# Router setup
router = APIRouter(prefix="/api/v1", tags=["assistant"])

# Global state (will be initialized in main.py)
_hybrid_router: Optional[HybridRouter] = None


def init_router(vector_persist_dir: str = "./chroma_db") -> None:
    """Initialize the hybrid router.
    
    Args:
        vector_persist_dir: Directory for vector store persistence
    """
    global _hybrid_router
    
    # Initialize components
    db = DatabaseConnection.from_env()
    vector_store = VectorStore(persist_directory=vector_persist_dir)
    rag_pipeline = RAGPipeline(vector_store=vector_store)
    
    # Create router
    _hybrid_router = HybridRouter(rag_pipeline=rag_pipeline, db=db)


def get_router() -> HybridRouter:
    """Get the initialized hybrid router."""
    if _hybrid_router is None:
        raise RuntimeError("Router not initialized. Call init_router() first.")
    return _hybrid_router


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Query the assistant",
    description="Send a question to the hybrid AI assistant"
)
def query_assistant(request: QueryRequest) -> QueryResponse:
    """Query the hybrid assistant.
    
    The assistant can answer questions using:
    - RAG (document retrieval) for policy/guide questions
    - Database tools for inventory/orders/product search
    - Both sources combined for hybrid questions
    
    Args:
        request: Query request with message and optional parameters
        
    Returns:
        Query response with answer, sources, and routing info
        
    Raises:
        HTTPException: If query fails or parameters are invalid
    """
    try:
        hybrid_router = get_router()
        
        # Parse route type if provided
        route_type = None
        if request.route_type:
            try:
                route_type = RouteType(request.route_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid route_type: {request.route_type}. "
                           f"Must be one of: rag_only, tool_only, hybrid"
                )
        
        # Build kwargs for router
        kwargs = {}
        if route_type:
            kwargs["route_type"] = route_type
        if request.product_id is not None:
            kwargs["product_id"] = request.product_id
        if request.size is not None:
            kwargs["size"] = request.size
        if request.order_id is not None:
            kwargs["order_id"] = request.order_id
        if request.search_query is not None:
            kwargs["search_query"] = request.search_query
        
        # Query the router
        result = hybrid_router.query(request.message, **kwargs)
        
        # Convert sources to response model
        sources = [
            Source(source=src["source"], chunk_id=src["chunk_id"])
            for src in result.get("sources", [])
        ]
        
        # Build response
        return QueryResponse(
            answer=result["answer"],
            route_type=result["route_type"],
            rag_used=result["rag_used"],
            tools_used=result["tools_used"],
            sources=sources,
            tool_results=result.get("tool_results", [])
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ConnectionError as e:
        # LLM connection issues
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM service unavailable: {str(e)}"
        )
    except TimeoutError as e:
        # LLM timeout
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"LLM request timeout: {str(e)}"
        )
    except ValueError as e:
        # Invalid parameters
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/status",
    status_code=status.HTTP_200_OK,
    summary="Check assistant status",
    description="Check if the assistant is properly initialized"
)
def assistant_status() -> dict[str, Any]:
    """Check assistant initialization status.
    
    Returns:
        Status information including whether router is initialized
    """
    return {
        "status": "ok" if _hybrid_router is not None else "not_initialized",
        "router_initialized": _hybrid_router is not None
    }
