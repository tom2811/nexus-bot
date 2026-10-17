"""Resilient, swappable LLM providers.

A provider chain tries each target in order: retry with exponential backoff on
429/5xx/timeout honoring Retry-After, skip targets whose circuit breaker is
open, and record every fallback into the active trace. No API-key rotation:
rate limits are per-account, so key rotation does not raise quota.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.tracing import record_fallback, record_served, record_usage

logger = logging.getLogger("nexus.provider")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _parse_retry_after(headers: httpx.Headers) -> Optional[float]:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class ProviderError(Exception):
    """A provider call failed. `retryable` decides whether the chain retries."""

    def __init__(self, message: str, retryable: bool = False, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class ProviderResult:
    text: str
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; half-opens after `cooldown`s."""

    def __init__(self, threshold: int = 5, cooldown: float = 60.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._failures = 0
        self._opened_at = 0.0

    def allow(self) -> bool:
        if self._failures < self.threshold:
            return True
        if time.monotonic() - self._opened_at >= self.cooldown:
            self._failures = 0  # half-open: let one request probe
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._opened_at = time.monotonic()


class ChatProvider:
    name = "base"

    def __init__(self, model: str, breaker: Optional[CircuitBreaker] = None):
        self.model = model
        self.breaker = breaker or CircuitBreaker()

    def complete(self, messages: list[dict], temperature: float, timeout: int) -> ProviderResult:
        raise NotImplementedError


class GroqProvider(ChatProvider):
    name = "groq"

    def __init__(self, model: str, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, breaker: Optional[CircuitBreaker] = None):
        super().__init__(model, breaker)
        self.base_url = (base_url or "https://api.groq.com/openai/v1").strip().rstrip("/")
        self.api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()

    def complete(self, messages: list[dict], temperature: float, timeout: int) -> ProviderResult:
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        except httpx.TimeoutException as e:
            raise ProviderError(f"groq timeout: {e}", retryable=True)
        except httpx.ConnectError as e:
            raise ProviderError(f"groq connect: {e}", retryable=True)

        if r.status_code != 200:
            raise ProviderError(
                f"groq {r.status_code}: {r.text[:200]}",
                retryable=r.status_code in RETRYABLE_STATUS,
                retry_after=_parse_retry_after(r.headers),
            )
        data = r.json()
        usage = data.get("usage") or {}
        return ProviderResult(
            text=data["choices"][0]["message"]["content"],
            provider=self.name,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class OllamaProvider(ChatProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: Optional[str] = None,
                 breaker: Optional[CircuitBreaker] = None):
        super().__init__(model, breaker)
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip().rstrip("/")

    def complete(self, messages: list[dict], temperature: float, timeout: int) -> ProviderResult:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        prompt = "\n".join(m["content"] for m in messages if m.get("role") == "user")
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False,
                         "options": {"temperature": temperature}}
        if system:
            payload["system"] = system
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(f"{self.base_url}/api/generate", json=payload)
        except httpx.TimeoutException as e:
            raise ProviderError(f"ollama timeout: {e}", retryable=True)
        except httpx.ConnectError as e:
            raise ProviderError(f"ollama connect: {e}", retryable=True)

        if r.status_code != 200:
            raise ProviderError(
                f"ollama {r.status_code}",
                retryable=r.status_code in RETRYABLE_STATUS,
                retry_after=_parse_retry_after(r.headers),
            )
        return ProviderResult(text=r.json()["response"], provider=self.name, model=self.model)


def build_default_chain() -> list[ChatProvider]:
    """Build the provider chain from env.

    Production default: Groq primary -> Groq cheaper model -> Ollama (if configured).
    `LLM_PROVIDER_CHAIN=groq:openai/gpt-oss-120b,groq:openai/gpt-oss-20b` overrides it.
    """
    explicit = os.getenv("LLM_PROVIDER_CHAIN")
    if explicit:
        chain: list[ChatProvider] = []
        for target in explicit.split(","):
            target = target.strip()
            if not target:
                continue
            provider, _, model = target.partition(":")
            model = model.strip() or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            if provider == "groq":
                chain.append(GroqProvider(model))
            elif provider == "ollama":
                chain.append(OllamaProvider(model))
            else:
                logger.warning("Unknown provider in LLM_PROVIDER_CHAIN: %s", provider)
        return chain

    has_groq = bool(os.getenv("GROQ_API_KEY"))
    if has_groq:
        primary = (os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b").strip()
        fallback = (os.getenv("GROQ_FALLBACK_MODEL") or "openai/gpt-oss-20b").strip()
        models = [primary] + ([fallback] if fallback and fallback != primary else [])
        chain = [GroqProvider(m) for m in models]
        if os.getenv("OLLAMA_BASE_URL"):
            chain.append(OllamaProvider((os.getenv("OLLAMA_MODEL") or "llama3.2").strip()))
        return chain

    return [OllamaProvider((os.getenv("OLLAMA_MODEL") or "llama3.2").strip())]


class ProviderChain:
    """Ordered providers with retry/backoff and circuit breaking."""

    def __init__(self, providers: list[ChatProvider], max_retries: int = 2,
                 backoff_base: float = 0.5):
        self.providers = providers
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    @property
    def primary(self) -> Optional[ChatProvider]:
        return self.providers[0] if self.providers else None

    def complete(self, messages: list[dict], temperature: float, timeout: int) -> ProviderResult:
        # Per-request budget: retries and backoff must fit inside `timeout`,
        # so a slow primary can't stretch worst-case latency past the caller's
        # (and the client's) deadline.
        deadline = time.monotonic() + max(0.0, float(timeout))
        last: Optional[ProviderError] = None
        for provider in self.providers:
            if not provider.breaker.allow():
                record_fallback(provider.name, provider.model, "circuit_open")
                continue
            for attempt in range(self.max_retries + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last = ProviderError("request time budget exhausted", retryable=False)
                    record_fallback(provider.name, provider.model, "budget_exhausted")
                    break
                try:
                    result = provider.complete(messages, temperature, remaining)
                    provider.breaker.record_success()
                    record_served(result.provider, result.model)
                    record_usage(result.prompt_tokens, result.completion_tokens)
                    return result
                except ProviderError as e:
                    last = e
                    if not e.retryable or attempt == self.max_retries:
                        provider.breaker.record_failure()
                        record_fallback(provider.name, provider.model, str(e))
                        break
                    delay = e.retry_after if e.retry_after is not None else self.backoff_base * (2 ** attempt)
                    if time.monotonic() + delay >= deadline:
                        record_fallback(provider.name, provider.model, "budget_exhausted")
                        break
                    logger.warning("retrying %s/%s in %.2fs: %s", provider.name, provider.model, delay, e)
                    time.sleep(delay)
        raise last or ProviderError("no LLM providers configured", retryable=False)
