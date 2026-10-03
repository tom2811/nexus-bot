"""LLM service using Ollama."""

import os
import httpx
from typing import Optional


class LLMService:
    """Service for LLM inference using Ollama."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize Ollama client.
        
        Args:
            base_url: Ollama API URL (defaults to env var or localhost)
            model: Model name (defaults to env var or llama3.2)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        request_timeout = timeout if timeout is not None else self.timeout
        
        try:
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running? Try: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except httpx.TimeoutException:
            raise TimeoutError(f"LLM request timed out after {request_timeout}s")
