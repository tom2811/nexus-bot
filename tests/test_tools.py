"""Tests for database tools."""

import os
from pathlib import Path

import pytest

from app.database import DatabaseConnection, initialize_database
from app.tools import (
    InventoryTool,
    OrderStatusTool,
    ProductSearchTool,
    ToolError,
    ValidationError,
)


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
def inventory_tool(db):
    """Fixture providing inventory tool."""
    return InventoryTool(db)


@pytest.fixture
def search_tool(db):
    """Fixture providing product search tool."""
    return ProductSearchTool(db)


@pytest.fixture
def order_tool(db):
    """Fixture providing order status tool."""
    return OrderStatusTool(db)


# ============================================================================
# Inventory Tool Tests
# ============================================================================

@requires_postgres
def test_check_inventory_available(inventory_tool):
    """Test checking inventory for available product."""
    result = inventory_tool.check_inventory(product_id=1, size="10", color="White")
    
    assert result["available"] is True
    assert result["product_id"] == 1
    assert result["product_name"] == "Air Max 90"
    assert result["brand"] == "Nike"
    assert len(result["variants"]) == 1
    assert result["variants"][0]["size"] == "10"
    assert result["variants"][0]["color"] == "White"
    assert result["variants"][0]["quantity"] > 0
    assert result["variants"][0]["in_stock"] is True


@requires_postgres
def test_check_inventory_out_of_stock(inventory_tool):
    """Test checking inventory for out-of-stock product."""
    result = inventory_tool.check_inventory(product_id=1, size="10", color="Black")
    
    assert result["available"] is False
    assert result["product_id"] == 1
    assert len(result["variants"]) == 1
    assert result["variants"][0]["quantity"] == 0
    assert result["variants"][0]["in_stock"] is False


@requires_postgres
def test_check_inventory_all_variants(inventory_tool):
    """Test checking inventory without size/color filters."""
    result = inventory_tool.check_inventory(product_id=1)
    
    assert result["product_id"] == 1
    assert result["product_name"] == "Air Max 90"
    assert len(result["variants"]) > 1  # Multiple variants
    # Check structure of variants
    for variant in result["variants"]:
        assert "size" in variant
        assert "color" in variant
        assert "quantity" in variant
        assert "in_stock" in variant


@requires_postgres
def test_check_inventory_product_not_found(inventory_tool):
    """Test checking inventory for non-existent product."""
    result = inventory_tool.check_inventory(product_id=99999)
    
    assert result["available"] is False
    assert result["product_id"] == 99999
    assert result["product_name"] is None
    assert len(result["variants"]) == 0
    assert "message" in result


@requires_postgres
def test_check_inventory_invalid_product_id(inventory_tool):
    """Test validation of product_id."""
    with pytest.raises(ValidationError, match="positive integer"):
        inventory_tool.check_inventory(product_id=-1)
    
    with pytest.raises(ValidationError, match="positive integer"):
        inventory_tool.check_inventory(product_id=0)


@requires_postgres
def test_check_inventory_invalid_size(inventory_tool):
    """Test validation of size parameter."""
    with pytest.raises(ValidationError, match="size must be a string"):
        inventory_tool.check_inventory(product_id=1, size=123)


@requires_postgres
def test_check_inventory_invalid_color(inventory_tool):
    """Test validation of color parameter."""
    with pytest.raises(ValidationError, match="color must be a string"):
        inventory_tool.check_inventory(product_id=1, color=123)


# ============================================================================
# Product Search Tool Tests
# ============================================================================

@requires_postgres
def test_search_products_by_name(search_tool):
    """Test searching products by name."""
    results = search_tool.search_products(query="Air Max")
    
    assert len(results) >= 1
    assert any(result["name"] == "Air Max 90" for result in results)
    
    # Check structure
    for result in results:
        assert "id" in result
        assert "name" in result
        assert "brand" in result
        assert "category" in result
        assert "price" in result


@requires_postgres
def test_search_products_by_brand(search_tool):
    """Test searching products by brand."""
    results = search_tool.search_products(query="Nike")
    
    assert len(results) >= 2
    assert all(result["brand"] == "Nike" for result in results)


@requires_postgres
def test_search_products_with_category_filter(search_tool):
    """Test searching products with category filter."""
    results = search_tool.search_products(query="Nike", category="Footwear")
    
    assert len(results) >= 1
    assert all(result["category"] == "Footwear" for result in results)
    assert all(result["brand"] == "Nike" for result in results)


@requires_postgres
def test_search_products_case_insensitive(search_tool):
    """Test search is case-insensitive."""
    results_lower = search_tool.search_products(query="nike")
    results_upper = search_tool.search_products(query="NIKE")
    results_mixed = search_tool.search_products(query="NiKe")
    
    assert len(results_lower) == len(results_upper) == len(results_mixed)


@requires_postgres
def test_search_products_no_results(search_tool):
    """Test searching with no matching products."""
    results = search_tool.search_products(query="NonexistentProduct12345")
    
    assert len(results) == 0


@requires_postgres
def test_search_products_max_results(search_tool):
    """Test max_results limit."""
    results = search_tool.search_products(query="a", max_results=2)
    
    assert len(results) <= 2


@requires_postgres
def test_search_products_invalid_query(search_tool):
    """Test validation of query parameter."""
    with pytest.raises(ValidationError, match="non-empty string"):
        search_tool.search_products(query="")
    
    with pytest.raises(ValidationError, match="non-empty string"):
        search_tool.search_products(query="   ")


@requires_postgres
def test_search_products_invalid_max_results(search_tool):
    """Test validation of max_results parameter."""
    with pytest.raises(ValidationError, match="between 1 and 100"):
        search_tool.search_products(query="test", max_results=0)
    
    with pytest.raises(ValidationError, match="between 1 and 100"):
        search_tool.search_products(query="test", max_results=101)


# ============================================================================
# Order Status Tool Tests
# ============================================================================

@requires_postgres
def test_get_order_status(order_tool):
    """Test getting order status."""
    result = order_tool.get_order_status(order_id=1)
    
    assert result["order_id"] == 1
    assert result["status"] == "shipped"
    assert result["customer_name"] == "John Smith"
    assert result["customer_email"] == "john@example.com"
    assert result["total_amount"] == 159.98
    assert len(result["items"]) >= 2
    
    # Check item structure
    for item in result["items"]:
        assert "product_id" in item
        assert "product_name" in item
        assert "quantity" in item
        assert "price" in item


@requires_postgres
def test_get_order_status_with_email_verification(order_tool):
    """Test getting order status with email verification."""
    result = order_tool.get_order_status(
        order_id=1,
        customer_email="john@example.com"
    )
    
    assert result["order_id"] == 1
    assert result["customer_email"] == "john@example.com"


@requires_postgres
def test_get_order_status_wrong_email(order_tool):
    """Test getting order status with wrong email."""
    with pytest.raises(ToolError, match="email does not match"):
        order_tool.get_order_status(
            order_id=1,
            customer_email="wrong@example.com"
        )


@requires_postgres
def test_get_order_status_not_found(order_tool):
    """Test getting status for non-existent order."""
    with pytest.raises(ToolError, match="Order not found"):
        order_tool.get_order_status(order_id=99999)


@requires_postgres
def test_get_order_status_invalid_order_id(order_tool):
    """Test validation of order_id."""
    with pytest.raises(ValidationError, match="positive integer"):
        order_tool.get_order_status(order_id=-1)
    
    with pytest.raises(ValidationError, match="positive integer"):
        order_tool.get_order_status(order_id=0)


@requires_postgres
def test_get_order_status_invalid_email(order_tool):
    """Test validation of email parameter."""
    with pytest.raises(ValidationError, match="customer_email must be a string"):
        order_tool.get_order_status(order_id=1, customer_email=123)


@requires_postgres
def test_get_order_status_multiple_orders(order_tool):
    """Test getting status for multiple orders."""
    order1 = order_tool.get_order_status(order_id=1)
    order2 = order_tool.get_order_status(order_id=2)
    order3 = order_tool.get_order_status(order_id=3)
    
    assert order1["status"] == "shipped"
    assert order2["status"] == "processing"
    assert order3["status"] == "delivered"
    
    assert order1["customer_name"] != order2["customer_name"]
