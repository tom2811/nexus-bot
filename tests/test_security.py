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

    def test_verify_api_key_from_x_api_key_header(self, monkeypatch):
        """Host apps may send X-API-Key instead of Bearer."""
        monkeypatch.setenv("API_KEY", "test-key-123")

        assert verify_api_key(None, header_key="test-key-123") is True
        assert verify_api_key(None, header_key="wrong-key") is False

    def test_verify_multiple_api_keys(self, monkeypatch):
        """Should support multiple API keys for different clients."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("API_KEYS", "kickskraze_key,other_app_key,third_app_key")

        # Valid keys should pass
        assert verify_api_key(None, header_key="kickskraze_key") is True
        assert verify_api_key(None, header_key="other_app_key") is True
        assert verify_api_key(None, header_key="third_app_key") is True
        
        # Invalid key should fail
        assert verify_api_key(None, header_key="wrong_key") is False

    def test_verify_multiple_api_keys_with_whitespace(self, monkeypatch):
        """Should handle whitespace in API_KEYS list."""
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("API_KEYS", "key1 , key2  ,  key3")

        assert verify_api_key(None, header_key="key1") is True
        assert verify_api_key(None, header_key="key2") is True
        assert verify_api_key(None, header_key="key3") is True

    def test_api_keys_takes_precedence_over_api_key(self, monkeypatch):
        """API_KEYS should take precedence over API_KEY when both are set."""
        monkeypatch.setenv("API_KEY", "old_single_key")
        monkeypatch.setenv("API_KEYS", "new_key1,new_key2")

        # New keys should work
        assert verify_api_key(None, header_key="new_key1") is True
        assert verify_api_key(None, header_key="new_key2") is True
        
        # Old single key should not work (API_KEYS overrides)
        assert verify_api_key(None, header_key="old_single_key") is False

    def test_backward_compatible_single_api_key(self, monkeypatch):
        """Should still work with single API_KEY for backward compatibility."""
        monkeypatch.delenv("API_KEYS", raising=False)
        monkeypatch.setenv("API_KEY", "single_key")

        assert verify_api_key(None, header_key="single_key") is True
        assert verify_api_key(None, header_key="wrong_key") is False
