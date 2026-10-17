"""LLM service facade over the resilient provider chain.

Public attributes (`provider`, `base_url`, `model`, `timeout`) are preserved for
callers and tests; `generate()` delegates retry/fallback to `ProviderChain`.
"""

import os
from typing import Optional

from app.providers import (
    GroqProvider,
    OllamaProvider,
    ProviderChain,
    build_default_chain,
)


class LLMService:
    """LLM inference via a retrying, fallback-capable provider chain."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        chain: Optional[ProviderChain] = None,
    ):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            self.provider = "groq"
            self.base_url = (base_url or "https://api.groq.com/openai/v1").strip()
            self.model = (model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")).strip()
        else:
            self.provider = "ollama"
            self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip()
            self.model = (model or os.getenv("OLLAMA_MODEL", "llama3.2")).strip()

        self.timeout = timeout if timeout is not None else int(os.getenv("LLM_TIMEOUT", "60"))

        if chain is not None:
            self.chain = chain
        elif base_url or model:
            # Explicit single target override (caller pinned a provider/model).
            if self.provider == "groq":
                self.chain = ProviderChain([GroqProvider(self.model, base_url=self.base_url)])
            else:
                self.chain = ProviderChain([OllamaProvider(self.model, base_url=self.base_url)])
        else:
            self.chain = ProviderChain(build_default_chain())

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
    ) -> str:
        """Generate a response, retrying/falling back across providers."""
        request_timeout = timeout if timeout is not None else self.timeout
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = self.chain.complete(messages, temperature, request_timeout)
        return result.text.strip()
