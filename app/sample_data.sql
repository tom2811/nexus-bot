-- Sample data for testing

-- Clear existing data
TRUNCATE TABLE order_items, orders, inventory, products RESTART IDENTITY CASCADE;

-- Insert sample products
INSERT INTO products (name, description, brand, category, price) VALUES
('Air Max 90', 'Classic running shoe with visible Air cushioning', 'Nike', 'Footwear', 129.99),
('UltraBoost 22', 'Energy-returning running shoe', 'Adidas', 'Footwear', 189.99),
('Classic T-Shirt', 'Cotton crew neck t-shirt', 'Nike', 'Apparel', 29.99),
('Training Shorts', 'Lightweight athletic shorts', 'Adidas', 'Apparel', 39.99),
('Sport Watch', 'GPS fitness tracker', 'Garmin', 'Electronics', 299.99);

-- Insert inventory
INSERT INTO inventory (product_id, size, color, quantity) VALUES
-- Air Max 90
(1, '8', 'White', 15),
(1, '9', 'White', 8),
(1, '10', 'White', 12),
(1, '11', 'White', 5),
(1, '8', 'Black', 10),
(1, '9', 'Black', 3),
(1, '10', 'Black', 0),
-- UltraBoost 22
(2, '8', 'Grey', 20),
(2, '9', 'Grey', 15),
(2, '10', 'Grey', 18),
(2, '11', 'Grey', 10),
-- Classic T-Shirt
(3, 'S', 'White', 50),
(3, 'M', 'White', 45),
(3, 'L', 'White', 30),
(3, 'XL', 'White', 20),
(3, 'M', 'Black', 25),
(3, 'L', 'Black', 15),
-- Training Shorts
(4, 'S', 'Black', 30),
(4, 'M', 'Black', 40),
(4, 'L', 'Black', 25),
-- Sport Watch
(5, 'One Size', 'Black', 12);

-- Insert sample orders
INSERT INTO orders (customer_name, customer_email, status, total_amount) VALUES
('John Smith', 'john@example.com', 'shipped', 159.98),
('Jane Doe', 'jane@example.com', 'processing', 219.98),
('Bob Johnson', 'bob@example.com', 'delivered', 329.98);

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, price, size, color) VALUES
-- Order 1
(1, 1, 1, 129.99, '10', 'White'),
(1, 3, 1, 29.99, 'M', 'White'),
-- Order 2
(2, 2, 1, 189.99, '9', 'Grey'),
(2, 3, 1, 29.99, 'L', 'Black'),
-- Order 3
(3, 5, 1, 299.99, 'One Size', 'Black'),
(3, 3, 1, 29.99, 'M', 'White');
