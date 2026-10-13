"""Tests for hybrid router."""

import os
from pathlib import Path

import pytest

from app.database import DatabaseConnection, initialize_database
from app.llm import LLMService
from app.rag import RAGPipeline
from app.router import HybridRouter, RouteType
from app.vector_store import VectorStore


# Test database configuration
TEST_DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "nexus_bot_test"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


def is_postgres_available():
    """Check if PostgreSQL is available for testing."""
    try:
        db = DatabaseConnection(**TEST_DB_CONFIG)
        db.close()
        return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not is_postgres_available(),
    reason="PostgreSQL not available or credentials not configured"
)


@pytest.fixture
def db():
    """Fixture providing a test database connection."""
    db = DatabaseConnection(**TEST_DB_CONFIG)
    
    # Initialize schema
    schema_file = Path(__file__).parent.parent / "app" / "schema.sql"
    data_file = Path(__file__).parent.parent / "app" / "sample_data.sql"
    
    try:
        initialize_database(db, str(schema_file), str(data_file))
    except Exception:
        pytest.skip("Cannot initialize test database")
    
    yield db
    
    db.close()


@pytest.fixture
def vector_store():
    """Fixture providing a test vector store."""
    import tempfile
    import shutil
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    store = VectorStore(
        persist_directory=temp_dir,
        collection_name="test_router"
    )
    
    # Add sample documents
    docs = [
        {
            "content": "Our return policy allows returns within 30 days of purchase with original receipt.",
            "source": "policy.txt",
            "type": "txt",
            "chunk_id": "0"
        },
        {
            "content": "Shipping is free for orders over $50. Standard shipping takes 5-7 business days.",
            "source": "shipping.txt",
            "type": "txt",
            "chunk_id": "0"
        }
    ]
    store.add_documents(docs)
    
    yield store
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def rag_pipeline(vector_store, monkeypatch):
    """Fixture providing RAG pipeline with mocked LLM."""
    from unittest.mock import Mock
    
    # Create mock LLM
    mock_llm = Mock(spec=LLMService)
    mock_llm.generate.return_value = "This is a RAG answer about 30 days return policy and shipping information."
    
    pipeline = RAGPipeline(vector_store=vector_store)
    # Replace the LLM service with mock
    pipeline.llm_service = mock_llm
    
    return pipeline


@pytest.fixture
def router(rag_pipeline, db, monkeypatch):
    """Fixture providing hybrid router with mocked LLM."""
    from unittest.mock import Mock
    
    # Create mock LLM that returns predictable responses
    mock_llm = Mock(spec=LLMService)
    mock_llm.generate.return_value = "This is a generated answer combining the information."
    
    router = HybridRouter(rag_pipeline=rag_pipeline, db=db)
    # Replace the LLM service with mock
    router.llm = mock_llm
    
    return router


# ============================================================================
# Intent Detection Tests
# ============================================================================

def test_detect_rag_intent(router):
    """Test detecting RAG-only intent."""
    questions = [
        "What is the return policy?",
        "Tell me about shipping",
        "How do I return an item?",
        "Explain the refund process"
    ]
    
    for question in questions:
        route_type = router.detect_intent(question)
        assert route_type == RouteType.RAG_ONLY


def test_detect_tool_intent(router):
    """Test detecting tool-only intent."""
    questions = [
        "Is product 123 available?",
        "Check inventory for Air Max",
        "What's my order status?",
        "Search for Nike shoes",
        "Find products in stock"
    ]
    
    for question in questions:
        route_type = router.detect_intent(question)
        assert route_type == RouteType.TOOL_ONLY


def test_detect_hybrid_intent(router):
    """Test detecting hybrid intent."""
    questions = [
        "Is product 123 available and what's the return policy?",
        "Can I return items that are in stock?",
        "Check inventory and tell me about shipping",
    ]
    
    for question in questions:
        route_type = router.detect_intent(question)
        assert route_type == RouteType.HYBRID


def test_detect_intent_case_insensitive(router):
    """Test intent detection is case-insensitive."""
    assert router.detect_intent("RETURN POLICY") == RouteType.RAG_ONLY
    assert router.detect_intent("check inventory") == RouteType.TOOL_ONLY
    assert router.detect_intent("AVAILABLE and POLICY") == RouteType.HYBRID


# ============================================================================
# RAG-Only Routing Tests
# ============================================================================

@requires_postgres
def test_route_rag_only(router):
    """Test RAG-only routing."""
    result = router.query("What is the return policy?")
    
    assert result["route_type"] == RouteType.RAG_ONLY.value
    assert result["rag_used"] is True
    assert len(result["tools_used"]) == 0
    assert "answer" in result
    assert "30 days" in result["answer"].lower() or "return" in result["answer"].lower()


@requires_postgres
def test_route_rag_with_sources(router):
    """Test RAG routing includes sources."""
    result = router.query("What is the shipping policy?")
    
    assert result["route_type"] == RouteType.RAG_ONLY.value
    assert "sources" in result
    # Should have source from shipping document
    assert any("shipping" in str(source).lower() for source in result["sources"])


# ============================================================================
# Tool-Only Routing Tests
# ============================================================================

@requires_postgres
def test_route_tool_inventory(router):
    """Test tool-only routing for inventory check."""
    result = router.query(
        "Is product available?",
        product_id=1,
        size="10",
        color="White"
    )
    
    assert result["route_type"] == RouteType.TOOL_ONLY.value
    assert result["rag_used"] is False
    assert "inventory" in result["tools_used"]
    assert "answer" in result


@requires_postgres
def test_route_tool_search(router):
    """Test tool-only routing for product search."""
    result = router.query(
        "Search for Nike products",
        search_query="Nike"
    )
    
    assert result["route_type"] == RouteType.TOOL_ONLY.value
    assert "search" in result["tools_used"]
    assert "tool_results" in result


@requires_postgres
def test_route_tool_order_status(router):
    """Test tool-only routing for order status."""
    result = router.query(
        "What's my order status?",
        order_id=1
    )
    
    assert result["route_type"] == RouteType.TOOL_ONLY.value
    assert "order_status" in result["tools_used"]
    assert "answer" in result


# ============================================================================
# Hybrid Routing Tests
# ============================================================================

@requires_postgres
def test_route_hybrid(router):
    """Test hybrid routing with both RAG and tools."""
    result = router.query(
        "Is product available and what's the return policy?",
        product_id=1,
        size="10"
    )
    
    assert result["route_type"] == RouteType.HYBRID.value
    assert result["rag_used"] is True
    assert len(result["tools_used"]) > 0
    assert "answer" in result


@requires_postgres
def test_route_hybrid_combines_sources(router):
    """Test hybrid routing combines sources from RAG and tools."""
    result = router.query(
        "Check inventory and shipping policy",
        product_id=1
    )
    
    assert result["route_type"] == RouteType.HYBRID.value
    # Should have RAG sources
    assert len(result.get("sources", [])) > 0
    # Should have tool results
    assert len(result.get("tool_results", [])) > 0


# ============================================================================
# Manual Route Type Tests
# ============================================================================

@requires_postgres
def test_manual_route_type_rag(router):
    """Test manually specifying RAG route."""
    result = router.query(
        "Some question",
        route_type=RouteType.RAG_ONLY
    )
    
    assert result["route_type"] == RouteType.RAG_ONLY.value
    assert result["rag_used"] is True


@requires_postgres
def test_manual_route_type_tool(router):
    """Test manually specifying tool route."""
    result = router.query(
        "Some question",
        route_type=RouteType.TOOL_ONLY,
        product_id=1
    )
    
    assert result["route_type"] == RouteType.TOOL_ONLY.value
    assert result["rag_used"] is False


@requires_postgres
def test_manual_route_type_hybrid(router):
    """Test manually specifying hybrid route."""
    result = router.query(
        "Some question",
        route_type=RouteType.HYBRID,
        product_id=1
    )
    
    assert result["route_type"] == RouteType.HYBRID.value
    assert result["rag_used"] is True


# ============================================================================
# Error Handling Tests
# ============================================================================

@requires_postgres
def test_route_tool_invalid_product(router):
    """Test tool routing with invalid product."""
    result = router.query(
        "Is product available?",
        product_id=99999
    )
    
    assert result["route_type"] == RouteType.TOOL_ONLY.value
    assert "answer" in result
    # Should handle gracefully


@requires_postgres
def test_route_no_parameters(router):
    """Test tool routing without required parameters."""
    result = router.query("Check inventory")
    
    # Should fallback to RAG or return helpful message
    assert "answer" in result


# ============================================================================
# Integration Tests
# ============================================================================

@requires_postgres
def test_router_initialization(router):
    """Test router initializes all components."""
    assert router.rag_pipeline is not None
    assert router.db is not None
    assert router.llm is not None
    assert router.inventory_tool is not None
    assert router.search_tool is not None
    assert router.order_tool is not None


@requires_postgres
def test_router_with_custom_llm(rag_pipeline, db):
    """Test router with custom LLM service."""
    custom_llm = LLMService(
        base_url="http://localhost:11434",
        model="llama3.2"
    )
    
    router = HybridRouter(
        rag_pipeline=rag_pipeline,
        db=db,
        llm=custom_llm
    )
    
    assert router.llm == custom_llm


def test_product_context_answers_without_database():
    """Reusable hosts supply catalog data; do not require local product IDs."""
    from unittest.mock import MagicMock

    rag = MagicMock()
    router = HybridRouter(rag_pipeline=rag, db=MagicMock())
    result = router.query(
        "Is size 10 in stock?",
        product_context={
            "name": "Air Max",
            "brand": "Nike",
            "retailPrice": 180,
            "availableSizes": ["9", "10"],
            "inStock": True,
        },
        size="10",
    )

    assert result["route_type"] == RouteType.TOOL_ONLY.value
    assert "caller_catalog" in result["tools_used"]
    assert "10" in result["answer"]
    assert "Air Max" in result["answer"]
    rag.query.assert_not_called()


def test_hybrid_query_uses_tenant_collection():
    """Hybrid RAG must search the caller's tenant, not the default collection."""
    from unittest.mock import MagicMock

    rag = MagicMock()
    rag.query.return_value = {
        "answer": "14-day returns",
        "sources": [{"source": "return_policy.md"}],
        "context_used": True,
    }
    llm = MagicMock()
    llm.generate.return_value = "Size 10 is in stock. Returns are 14 days."
    router = HybridRouter(rag_pipeline=rag, db=MagicMock(), llm=llm)

    router.query(
        "Is size 10 in stock and what's the return policy?",
        tenant_id="kickskraze",
        size="10",
        product_context={
            "name": "Air Max",
            "brand": "Nike",
            "availableSizes": ["10"],
            "inStock": True,
            "retailPrice": 180,
        },
    )

    rag.query.assert_called_with(
        "Is size 10 in stock and what's the return policy?",
        collection_name="kickskraze",
    )
