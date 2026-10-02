"""Tests for LLM service."""

import pytest

from app.llm import LLMService


def test_llm_service_initialization():
    """Test LLM service can be initialized."""
    llm = LLMService()
    assert llm.base_url == "http://localhost:11434"
    assert llm.model == "llama3.2"
    assert llm.timeout == 60


def test_llm_service_custom_config():
    """Test LLM service with custom configuration."""
    llm = LLMService(
        base_url="http://custom:8080",
        model="llama2",
        timeout=120,
    )
    assert llm.base_url == "http://custom:8080"
    assert llm.model == "llama2"
    assert llm.timeout == 120


def test_llm_generate():
    """Test LLM generation (requires Ollama)."""
    llm = LLMService()
    response = llm.generate("Say 'test successful' and nothing else.")
    assert isinstance(response, str)
    assert len(response) > 0
