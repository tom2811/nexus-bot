"""Database connection and operations for structured data."""

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor


class DatabaseError(Exception):
    """Base exception for database errors."""
    pass


class DatabaseConnection:
    """Manages PostgreSQL database connections with connection pooling."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "nexus_bot",
        user: str = "postgres",
        password: str = "",
        min_connections: int = 1,
        max_connections: int = 5,
    ):
        """Initialize database connection pool.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        self.config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                min_connections,
                max_connections,
                **self.config
            )
        except psycopg2.Error as e:
            raise DatabaseError(f"Failed to create connection pool: {e}")

    @contextmanager
    def get_connection(self) -> Generator:
        """Get a connection from the pool.
        
        Yields:
            Database connection
            
        Raises:
            DatabaseError: If connection cannot be obtained
        """
        conn = None
        try:
            conn = self.pool.getconn()
            if conn is None:
                raise DatabaseError("No connection available")
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database error: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)

    @contextmanager
    def get_cursor(self, connection=None) -> Generator:
        """Get a cursor for executing queries.
        
        Args:
            connection: Optional existing connection. If None, gets one from pool.
            
        Yields:
            Database cursor (returns rows as dictionaries)
        """
        if connection:
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            try:
                yield cursor
            finally:
                cursor.close()
        else:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                try:
                    yield cursor
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    cursor.close()

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query and return results.
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: Whether to fetch results
            
        Returns:
            List of rows as dictionaries
            
        Raises:
            DatabaseError: If query execution fails
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            if fetch:
                return [dict(row) for row in cursor.fetchall()]
            return []

    def execute_update(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """Execute an INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            Number of affected rows
            
        Raises:
            DatabaseError: If query execution fails
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()

    @classmethod
    def from_env(cls) -> "DatabaseConnection":
        """Create database connection from environment variables.
        
        Environment variables:
            DB_HOST: Database host (default: localhost)
            DB_PORT: Database port (default: 5432)
            DB_NAME: Database name (default: nexus_bot)
            DB_USER: Database user (default: postgres)
            DB_PASSWORD: Database password (default: empty)
            
        Returns:
            DatabaseConnection instance
        """
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "nexus_bot"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )


def initialize_database(db: DatabaseConnection, schema_file: str, data_file: Optional[str] = None):
    """Initialize database with schema and optional sample data.
    
    Args:
        db: Database connection
        schema_file: Path to schema SQL file
        data_file: Optional path to sample data SQL file
        
    Raises:
        DatabaseError: If initialization fails
    """
    try:
        # Load and execute schema
        with open(schema_file, "r") as f:
            schema_sql = f.read()
        
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
                conn.commit()
        
        # Load and execute sample data if provided
        if data_file:
            with open(data_file, "r") as f:
                data_sql = f.read()
            
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(data_sql)
                    conn.commit()
                    
    except (psycopg2.Error, IOError) as e:
        raise DatabaseError(f"Failed to initialize database: {e}")
