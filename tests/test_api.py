"""Tests for API endpoints."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_router():
    """Create mock hybrid router."""
    mock = MagicMock()
    mock.query.return_value = {
        "answer": "Test answer",
        "sources": [{"source": "test.txt", "chunk_id": "chunk_1"}],
        "route_type": "rag_only",
        "rag_used": True,
        "tools_used": [],
        "tool_results": []
    }
    return mock


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self, client):
        """Test health endpoint returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAssistantStatus:
    """Tests for assistant status endpoint."""
    
    @patch("app.api._hybrid_router", None)
    def test_status_not_initialized(self, client):
        """Test status when router not initialized."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_initialized"
        assert data["router_initialized"] is False
    
    @patch("app.api._hybrid_router", MagicMock())
    def test_status_initialized(self, client):
        """Test status when router is initialized."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["router_initialized"] is True


class TestQueryEndpoint:
    """Tests for query endpoint."""
    
    @patch("app.api.get_router")
    def test_query_rag_only(self, mock_get_router, client, mock_router):
        """Test RAG-only query."""
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={"message": "What is the return policy?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["route_type"] == "rag_only"
        assert data["rag_used"] is True
        assert data["tools_used"] == []
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "test.txt"
    
    @patch("app.api.get_router")
    def test_query_tool_only(self, mock_get_router, client, mock_router):
        """Test tool-only query."""
        mock_router.query.return_value = {
            "answer": "Product is available",
            "sources": [],
            "route_type": "tool_only",
            "rag_used": False,
            "tools_used": ["inventory"],
            "tool_results": [{"available": True, "quantity": 12}]
        }
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={
                "message": "Is Air Max available?",
                "product_id": 1,
                "size": "10"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Product is available"
        assert data["route_type"] == "tool_only"
        assert data["rag_used"] is False
        assert data["tools_used"] == ["inventory"]
        assert len(data["tool_results"]) == 1
    
    @patch("app.api.get_router")
    def test_query_hybrid(self, mock_get_router, client, mock_router):
        """Test hybrid query."""
        mock_router.query.return_value = {
            "answer": "Product is available. Return policy allows 30 days.",
            "sources": [{"source": "policy.txt", "chunk_id": "chunk_1"}],
            "route_type": "hybrid",
            "rag_used": True,
            "tools_used": ["inventory"],
            "tool_results": [{"available": True, "quantity": 12}]
        }
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={
                "message": "Is Air Max available and what's the return policy?",
                "product_id": 1,
                "size": "10"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["route_type"] == "hybrid"
        assert data["rag_used"] is True
        assert data["tools_used"] == ["inventory"]
        assert len(data["sources"]) == 1
        assert len(data["tool_results"]) == 1
    
    @patch("app.api.get_router")
    def test_query_with_manual_routing(self, mock_get_router, client, mock_router):
        """Test query with manual route type."""
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={
                "message": "Test question",
                "route_type": "rag_only"
            }
        )
        
        assert response.status_code == 200
        # Verify router was called with route_type
        mock_router.query.assert_called_once()
        call_kwargs = mock_router.query.call_args[1]
        assert "route_type" in call_kwargs

    @patch("app.api.get_router")
    def test_query_accepts_sources_without_chunk_id(self, mock_get_router, client, mock_router):
        """RAG sources omit chunk_id; the API must still return 200."""
        mock_router.query.return_value = {
            "answer": "Returns within 14 days.",
            "sources": [{"source": "return_policy.md"}],
            "route_type": "rag_only",
            "rag_used": True,
            "tools_used": [],
            "tool_results": [],
        }
        mock_get_router.return_value = mock_router

        response = client.post(
            "/api/v1/query",
            json={"message": "What is the return policy?", "tenant_id": "kickskraze"},
        )

        assert response.status_code == 200
        assert response.json()["sources"][0]["source"] == "return_policy.md"
        assert response.json()["sources"][0]["chunk_id"] is None

    @patch("app.api.get_router")
    def test_query_forwards_product_context(self, mock_get_router, client, mock_router):
        """Host apps pass live catalog context instead of local product IDs."""
        mock_get_router.return_value = mock_router
        product_context = {
            "name": "Air Max",
            "brand": "Nike",
            "availableSizes": ["10"],
            "inStock": True,
            "retailPrice": 180,
        }

        response = client.post(
            "/api/v1/query",
            json={
                "message": "Is size 10 in stock?",
                "product_id": "fbca06da-b5ca-4b2c-ad08-5b3e943777fe",
                "size": "10",
                "tenant_id": "kickskraze",
                "product_context": product_context,
            },
        )

        assert response.status_code == 200
        kwargs = mock_router.query.call_args.kwargs
        assert kwargs["tenant_id"] == "kickskraze"
        assert kwargs["product_context"]["name"] == "Air Max"
        assert kwargs["size"] == "10"
    
    def test_query_empty_message(self, client):
        """Test query with empty message."""
        response = client.post(
            "/api/v1/query",
            json={"message": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_query_message_too_long(self, client):
        """Test query with message exceeding max length."""
        response = client.post(
            "/api/v1/query",
            json={"message": "x" * 1001}
        )
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_message(self, client):
        """Test query without message field."""
        response = client.post(
            "/api/v1/query",
            json={}
        )
        assert response.status_code == 422  # Validation error
    
    @patch("app.api.get_router")
    def test_query_invalid_route_type(self, mock_get_router, client, mock_router):
        """Test query with invalid route type."""
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={
                "message": "Test question",
                "route_type": "invalid_type"
            }
        )
        
        assert response.status_code == 400
        assert "Invalid route_type" in response.json()["detail"]
    
    @patch("app.api.get_router")
    def test_query_router_not_initialized(self, mock_get_router, client):
        """Test query when router not initialized."""
        mock_get_router.side_effect = RuntimeError("Router not initialized")
        
        response = client.post(
            "/api/v1/query",
            json={"message": "Test question"}
        )
        
        assert response.status_code == 500
    
    @patch("app.api.get_router")
    def test_query_connection_error(self, mock_get_router, client, mock_router):
        """Test query with LLM connection error."""
        mock_router.query.side_effect = ConnectionError("Ollama not available")
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={"message": "Test question"}
        )
        
        assert response.status_code == 503
        assert "LLM service unavailable" in response.json()["detail"]
    
    @patch("app.api.get_router")
    def test_query_timeout_error(self, mock_get_router, client, mock_router):
        """Test query with LLM timeout."""
        mock_router.query.side_effect = TimeoutError("Request timeout")
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={"message": "Test question"}
        )
        
        assert response.status_code == 504
        assert "LLM request timeout" in response.json()["detail"]
    
    @patch("app.api.get_router")
    def test_query_value_error(self, mock_get_router, client, mock_router):
        """Test query with invalid parameters."""
        mock_router.query.side_effect = ValueError("Invalid product_id")
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={
                "message": "Test question",
                "product_id": -1
            }
        )
        
        assert response.status_code == 400
        assert "Invalid product_id" in response.json()["detail"]
    
    @patch("app.api.get_router")
    def test_query_with_all_parameters(self, mock_get_router, client, mock_router):
        """Test query with all optional parameters."""
        mock_get_router.return_value = mock_router
        
        response = client.post(
            "/api/v1/query",
            json={
                "message": "Test question",
                "route_type": "hybrid",
                "product_id": 1,
                "size": "10",
                "order_id": 1001,
                "search_query": "Nike"
            }
        )
        
        assert response.status_code == 200
        # Verify all parameters passed to router
        call_kwargs = mock_router.query.call_args[1]
        assert call_kwargs["product_id"] == 1
        assert call_kwargs["size"] == "10"
        assert call_kwargs["order_id"] == 1001
        assert call_kwargs["search_query"] == "Nike"


class TestAPIDocumentation:
    """Tests for API documentation."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Nexus-bot"
        assert schema["info"]["version"] == "0.2.0"
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
