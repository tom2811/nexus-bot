#!/bin/bash
# Setup test database for nexus-bot

echo "Setting up test database..."

# Check if PostgreSQL is running
if ! psql -U postgres -c "SELECT 1" > /dev/null 2>&1; then
    echo "Error: PostgreSQL is not accessible or requires password"
    echo "Please ensure PostgreSQL is running and credentials are correct"
    exit 1
fi

# Create test database if it doesn't exist
psql -U postgres -c "CREATE DATABASE nexus_bot_test;" 2>/dev/null
echo "✓ Database nexus_bot_test ready"

# Initialize schema
psql -U postgres -d nexus_bot_test -f app/schema.sql > /dev/null 2>&1
echo "✓ Schema initialized"

# Load sample data
psql -U postgres -d nexus_bot_test -f app/sample_data.sql > /dev/null 2>&1
echo "✓ Sample data loaded"

echo ""
echo "Test database ready! Run tests with:"
echo "  uv run pytest tests/test_database.py -v"
