"""Controlled database tools for safe LLM access.

These tools provide validated, safe access to the database without
exposing arbitrary SQL generation to the LLM.
"""

from typing import Any, Optional

from app.database import DatabaseConnection, DatabaseError


class ToolError(Exception):
    """Base exception for tool errors."""
    pass


class ValidationError(ToolError):
    """Raised when tool input validation fails."""
    pass


class InventoryTool:
    """Check product inventory availability."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize inventory tool.
        
        Args:
            db: Database connection
        """
        self.db = db
    
    def check_inventory(
        self,
        product_id: int,
        size: Optional[str] = None,
        color: Optional[str] = None
    ) -> dict[str, Any]:
        """Check if a product is in stock.
        
        Args:
            product_id: Product ID to check
            size: Optional size filter
            color: Optional color filter
            
        Returns:
            Dictionary with availability information:
            {
                "available": bool,
                "product_id": int,
                "product_name": str,
                "variants": [
                    {
                        "size": str,
                        "color": str,
                        "quantity": int
                    }
                ]
            }
            
        Raises:
            ValidationError: If inputs are invalid
            ToolError: If database query fails
        """
        # Validate inputs
        if not isinstance(product_id, int) or product_id <= 0:
            raise ValidationError("product_id must be a positive integer")
        
        if size is not None and not isinstance(size, str):
            raise ValidationError("size must be a string")
        
        if color is not None and not isinstance(color, str):
            raise ValidationError("color must be a string")
        
        try:
            # Get product info
            product_query = "SELECT id, name, brand, price FROM products WHERE id = %s"
            products = self.db.execute_query(product_query, (product_id,))
            
            if not products:
                return {
                    "available": False,
                    "product_id": product_id,
                    "product_name": None,
                    "variants": [],
                    "message": "Product not found"
                }
            
            product = products[0]
            
            # Build inventory query with filters
            inventory_query = """
                SELECT size, color, quantity
                FROM inventory
                WHERE product_id = %s
            """
            params = [product_id]
            
            if size:
                inventory_query += " AND size = %s"
                params.append(size)
            
            if color:
                inventory_query += " AND color = %s"
                params.append(color)
            
            inventory_query += " ORDER BY size, color"
            
            # Get inventory
            inventory = self.db.execute_query(inventory_query, tuple(params))
            
            # Check if any variant is available
            available = any(item["quantity"] > 0 for item in inventory)
            
            return {
                "available": available,
                "product_id": product_id,
                "product_name": product["name"],
                "brand": product["brand"],
                "price": float(product["price"]),
                "variants": [
                    {
                        "size": item["size"],
                        "color": item["color"],
                        "quantity": item["quantity"],
                        "in_stock": item["quantity"] > 0
                    }
                    for item in inventory
                ]
            }
            
        except DatabaseError as e:
            raise ToolError(f"Failed to check inventory: {e}")


class ProductSearchTool:
    """Search products by name, brand, or category."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize product search tool.
        
        Args:
            db: Database connection
        """
        self.db = db
    
    def search_products(
        self,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Search for products by name or brand.
        
        Args:
            query: Search query string
            category: Optional category filter
            max_results: Maximum number of results (default: 10)
            
        Returns:
            List of matching products with basic info
            
        Raises:
            ValidationError: If inputs are invalid
            ToolError: If database query fails
        """
        # Validate inputs
        if not isinstance(query, str) or not query.strip():
            raise ValidationError("query must be a non-empty string")
        
        if max_results <= 0 or max_results > 100:
            raise ValidationError("max_results must be between 1 and 100")
        
        if category is not None and not isinstance(category, str):
            raise ValidationError("category must be a string")
        
        try:
            # Build search query
            search_query = """
                SELECT id, name, description, brand, category, price
                FROM products
                WHERE (
                    LOWER(name) LIKE LOWER(%s)
                    OR LOWER(brand) LIKE LOWER(%s)
                    OR LOWER(description) LIKE LOWER(%s)
                )
            """
            search_pattern = f"%{query}%"
            params = [search_pattern, search_pattern, search_pattern]
            
            if category:
                search_query += " AND LOWER(category) = LOWER(%s)"
                params.append(category)
            
            search_query += " ORDER BY name LIMIT %s"
            params.append(max_results)
            
            # Execute search
            results = self.db.execute_query(search_query, tuple(params))
            
            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "description": row["description"],
                    "brand": row["brand"],
                    "category": row["category"],
                    "price": float(row["price"])
                }
                for row in results
            ]
            
        except DatabaseError as e:
            raise ToolError(f"Failed to search products: {e}")


class OrderStatusTool:
    """Get order status and details."""
    
    def __init__(self, db: DatabaseConnection):
        """Initialize order status tool.
        
        Args:
            db: Database connection
        """
        self.db = db
    
    def get_order_status(
        self,
        order_id: int,
        customer_email: Optional[str] = None
    ) -> dict[str, Any]:
        """Get order status and details.
        
        Args:
            order_id: Order ID to check
            customer_email: Optional email for verification
            
        Returns:
            Dictionary with order information:
            {
                "order_id": int,
                "status": str,
                "customer_name": str,
                "customer_email": str,
                "total_amount": float,
                "created_at": str,
                "items": [...]
            }
            
        Raises:
            ValidationError: If inputs are invalid
            ToolError: If database query fails or order not found
        """
        # Validate inputs
        if not isinstance(order_id, int) or order_id <= 0:
            raise ValidationError("order_id must be a positive integer")
        
        if customer_email is not None and not isinstance(customer_email, str):
            raise ValidationError("customer_email must be a string")
        
        try:
            # Get order
            order_query = """
                SELECT id, customer_name, customer_email, status,
                       total_amount, created_at, updated_at
                FROM orders
                WHERE id = %s
            """
            params = [order_id]
            
            # Add email verification if provided
            if customer_email:
                order_query += " AND LOWER(customer_email) = LOWER(%s)"
                params.append(customer_email)
            
            orders = self.db.execute_query(order_query, tuple(params))
            
            if not orders:
                if customer_email:
                    raise ToolError("Order not found or email does not match")
                raise ToolError("Order not found")
            
            order = orders[0]
            
            # Get order items
            items_query = """
                SELECT oi.id, oi.product_id, p.name as product_name,
                       oi.quantity, oi.price, oi.size, oi.color
                FROM order_items oi
                JOIN products p ON oi.product_id = p.id
                WHERE oi.order_id = %s
            """
            items = self.db.execute_query(items_query, (order_id,))
            
            return {
                "order_id": order["id"],
                "status": order["status"],
                "customer_name": order["customer_name"],
                "customer_email": order["customer_email"],
                "total_amount": float(order["total_amount"]),
                "created_at": str(order["created_at"]),
                "updated_at": str(order["updated_at"]),
                "items": [
                    {
                        "product_id": item["product_id"],
                        "product_name": item["product_name"],
                        "quantity": item["quantity"],
                        "price": float(item["price"]),
                        "size": item["size"],
                        "color": item["color"]
                    }
                    for item in items
                ]
            }
            
        except DatabaseError as e:
            raise ToolError(f"Failed to get order status: {e}")
