"""Tests for security features."""

from unittest.mock import Mock

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from app.security import verify_api_key


class TestAPIKeyAuth:
    """Test API key authentication."""
    
    def test_verify_api_key_no_env_key(self, monkeypatch):
        """Should allow access when no API_KEY env var set."""
        monkeypatch.delenv("API_KEY", raising=False)
        
        result = verify_api_key(None)
        assert result is True
    
    def test_verify_api_key_valid(self, monkeypatch):
        """Should allow access with valid API key."""
        monkeypatch.setenv("API_KEY", "test-key-123")
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="test-key-123"
        )
        
        result = verify_api_key(credentials)
        assert result is True
    
    def test_verify_api_key_invalid(self, monkeypatch):
        """Should deny access with invalid API key."""
        monkeypatch.setenv("API_KEY", "test-key-123")
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="wrong-key"
        )
        
        result = verify_api_key(credentials)
        assert result is False
    
    def test_verify_api_key_missing(self, monkeypatch):
        """Should deny access when key required but not provided."""
        monkeypatch.setenv("API_KEY", "test-key-123")
        
        result = verify_api_key(None)
        assert result is False
