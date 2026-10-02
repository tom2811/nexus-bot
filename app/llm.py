"""LLM service using Ollama."""

import httpx
from typing import Optional


class LLMService:
    """Service for LLM inference using Ollama."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: int = 60,
    ):
        """Initialize Ollama client."""
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
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
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                return response.json()["response"]
        except httpx.ConnectError:
            raise ConnectionError(
                "Cannot connect to Ollama. Is it running? Try: ollama serve"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e}")
