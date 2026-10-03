"""Tests for LLM service."""

import os
import pytest

from app.llm import LLMService


def test_llm_service_initialization():
    """Test LLM service initializes with defaults."""
    # Clear env vars for this test
    old_url = os.environ.pop("OLLAMA_BASE_URL", None)
    old_model = os.environ.pop("OLLAMA_MODEL", None)
    
    try:
        llm = LLMService()
        assert llm.base_url == "http://localhost:11434"
        assert llm.model == "llama3.2"
        assert llm.timeout == 60
    finally:
        # Restore env vars
        if old_url:
            os.environ["OLLAMA_BASE_URL"] = old_url
        if old_model:
            os.environ["OLLAMA_MODEL"] = old_model


def test_llm_service_from_env():
    """Test LLM service reads from environment variables."""
    os.environ["OLLAMA_BASE_URL"] = "http://test:9999"
    os.environ["OLLAMA_MODEL"] = "test-model"
    
    llm = LLMService()
    assert llm.base_url == "http://test:9999"
    assert llm.model == "test-model"
    
    # Cleanup
    os.environ.pop("OLLAMA_BASE_URL", None)
    os.environ.pop("OLLAMA_MODEL", None)


def test_llm_service_custom_config():
    """Test LLM service with explicit configuration overrides env."""
    os.environ["OLLAMA_BASE_URL"] = "http://env:1111"
    os.environ["OLLAMA_MODEL"] = "env-model"
    
    llm = LLMService(
        base_url="http://custom:8080",
        model="llama2",
        timeout=120,
    )
    assert llm.base_url == "http://custom:8080"
    assert llm.model == "llama2"
    assert llm.timeout == 120
    
    # Cleanup
    os.environ.pop("OLLAMA_BASE_URL", None)
    os.environ.pop("OLLAMA_MODEL", None)


@pytest.mark.skipif(
    not os.getenv("OLLAMA_RUNNING", "").lower() == "true",
    reason="Ollama not running (set OLLAMA_RUNNING=true to enable)"
)
def test_llm_generate():
    """Test LLM generation (requires Ollama running)."""
    llm = LLMService()
    response = llm.generate("Say 'test successful' and nothing else.")
    assert isinstance(response, str)
    assert len(response) > 0
