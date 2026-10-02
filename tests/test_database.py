"""Tests for database connection and operations."""

import os
from pathlib import Path

import pytest

from app.database import DatabaseConnection, DatabaseError, initialize_database


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
    except DatabaseError:
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
    except DatabaseError:
        pytest.skip("Cannot initialize test database")
    
    yield db
    
    db.close()


@requires_postgres
def test_connection_pool_creation():
    """Test that connection pool is created successfully."""
    db = DatabaseConnection(**TEST_DB_CONFIG)
    
    assert db.pool is not None
    db.close()


@requires_postgres
def test_get_connection(db):
    """Test getting a connection from the pool."""
    with db.get_connection() as conn:
        assert conn is not None
        assert not conn.closed


@requires_postgres
def test_execute_query(db):
    """Test executing a SELECT query."""
    result = db.execute_query("SELECT * FROM products WHERE id = %s", (1,))
    
    assert len(result) == 1
    assert result[0]["name"] == "Air Max 90"
    assert result[0]["brand"] == "Nike"


@requires_postgres
def test_execute_query_multiple_rows(db):
    """Test query returning multiple rows."""
    result = db.execute_query("SELECT * FROM products WHERE brand = %s", ("Nike",))
    
    assert len(result) >= 2
    assert all(row["brand"] == "Nike" for row in result)


@requires_postgres
def test_execute_query_no_results(db):
    """Test query with no matching results."""
    result = db.execute_query(
        "SELECT * FROM products WHERE name = %s",
        ("Nonexistent Product",)
    )
    
    assert len(result) == 0


@requires_postgres
def test_execute_update(db):
    """Test executing an UPDATE query."""
    affected = db.execute_update(
        "UPDATE inventory SET quantity = %s WHERE product_id = %s AND size = %s AND color = %s",
        (100, 1, "10", "White")
    )
    
    assert affected == 1
    
    # Verify the update
    result = db.execute_query(
        "SELECT quantity FROM inventory WHERE product_id = %s AND size = %s AND color = %s",
        (1, "10", "White")
    )
    assert result[0]["quantity"] == 100


@requires_postgres
def test_product_with_inventory(db):
    """Test joining products with inventory."""
    query = """
        SELECT p.name, p.brand, i.size, i.color, i.quantity
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        WHERE p.id = %s AND i.size = %s
    """
    
    result = db.execute_query(query, (1, "10"))
    
    assert len(result) >= 1
    assert result[0]["name"] == "Air Max 90"
    assert result[0]["size"] == "10"


@requires_postgres
def test_order_with_items(db):
    """Test joining orders with order items."""
    query = """
        SELECT o.id, o.customer_name, o.status, oi.product_id, oi.quantity
        FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
        WHERE o.id = %s
    """
    
    result = db.execute_query(query, (1,))
    
    assert len(result) >= 1
    assert result[0]["customer_name"] == "John Smith"
    assert result[0]["status"] == "shipped"


@requires_postgres
def test_connection_error_handling():
    """Test handling of connection errors."""
    with pytest.raises(DatabaseError):
        DatabaseConnection(
            host="invalid_host",
            database="invalid_db",
            user="invalid_user",
            password="invalid_pass",
        )


@requires_postgres
def test_from_env():
    """Test creating connection from environment variables."""
    # Use actual environment credentials
    db = DatabaseConnection.from_env()
    assert db is not None
    
    # Verify connection works
    result = db.execute_query("SELECT 1 as test")
    assert result[0]["test"] == 1
    
    db.close()
