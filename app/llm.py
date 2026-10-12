"""LLM service supporting Groq (cloud) and Ollama (local)."""

import os
import httpx
from typing import Optional


class LLMService:
    """Service for LLM inference using Groq or Ollama."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize LLM client.

        If GROQ_API_KEY is present, defaults to Groq Cloud API.
        Otherwise, falls back to local Ollama.
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            self.provider = "groq"
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            self.provider = "ollama"
            self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/") + "/api/generate"
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")

        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM."""
        request_timeout = timeout if timeout is not None else self.timeout

        if self.provider == "groq":
            return self._generate_groq(prompt, system_prompt, temperature, request_timeout)
        return self._generate_ollama(prompt, system_prompt, temperature, request_timeout)

    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        timeout: int,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise ConnectionError("Cannot connect to Groq API.")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Groq API error ({e.response.status_code}): {e.response.text}")
        except httpx.TimeoutException:
            raise TimeoutError(f"Groq request timed out after {timeout}s")

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        timeout: int,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(self.base_url, json=payload)
                response.raise_for_status()
                return response.json()["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running? Try: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except httpx.TimeoutException:
            raise TimeoutError(f"LLM request timed out after {timeout}s")

