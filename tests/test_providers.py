"""Tests for the resilient provider chain."""

import pytest

from app import providers as providers_mod
from app.providers import (
    ChatProvider,
    CircuitBreaker,
    GroqProvider,
    OllamaProvider,
    ProviderChain,
    ProviderError,
    ProviderResult,
    _parse_retry_after,
    build_default_chain,
)


class FakeProvider(ChatProvider):
    def __init__(self, name, model="m", outcomes=None, breaker=None):
        super().__init__(model, breaker)
        self.name = name
        self.outcomes = list(outcomes or [])
        self.calls = 0

    def complete(self, messages, temperature, timeout):
        self.calls += 1
        outcome = (
            self.outcomes.pop(0)
            if self.outcomes
            else ProviderResult("ok", self.name, self.model)
        )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(providers_mod.time, "sleep", lambda _seconds: None)


class TestCircuitBreaker:
    def test_allows_until_threshold(self):
        breaker = CircuitBreaker(threshold=2)
        assert breaker.allow() is True
        breaker.record_failure()
        assert breaker.allow() is True
        breaker.record_failure()
        assert breaker.allow() is False

    def test_half_opens_after_cooldown(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr(providers_mod.time, "monotonic", lambda: clock["t"])
        breaker = CircuitBreaker(threshold=1, cooldown=60)
        breaker.record_failure()
        assert breaker.allow() is False
        clock["t"] = 61.0
        assert breaker.allow() is True

    def test_success_resets(self):
        breaker = CircuitBreaker(threshold=2)
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        assert breaker.allow() is True


class TestProviderChain:
    def test_primary_success(self):
        primary = FakeProvider("a", outcomes=[ProviderResult("hi", "a", "m")])
        chain = ProviderChain([primary])
        assert chain.complete([], 0.0, 5).text == "hi"
        assert primary.calls == 1

    def test_falls_back_to_next_provider(self):
        primary = FakeProvider("a", outcomes=[ProviderError("boom", retryable=True)])
        secondary = FakeProvider("b", outcomes=[ProviderResult("rescue", "b", "m")])
        chain = ProviderChain([primary, secondary], max_retries=0)
        assert chain.complete([], 0.0, 5).text == "rescue"
        assert primary.calls == 1
        assert secondary.calls == 1

    def test_retries_retryable_before_fallback(self):
        primary = FakeProvider(
            "a",
            outcomes=[ProviderError("flaky", retryable=True), ProviderResult("ok", "a", "m")],
        )
        chain = ProviderChain([primary], max_retries=1)
        assert chain.complete([], 0.0, 5).text == "ok"
        assert primary.calls == 2

    def test_non_retryable_moves_on_without_retry(self):
        primary = FakeProvider("a", outcomes=[ProviderError("bad", retryable=False)])
        secondary = FakeProvider("b")
        chain = ProviderChain([primary, secondary], max_retries=3)
        chain.complete([], 0.0, 5)
        assert primary.calls == 1

    def test_open_circuit_is_skipped(self):
        primary = FakeProvider(
            "a",
            outcomes=[ProviderError("boom", retryable=False)],
            breaker=CircuitBreaker(threshold=1),
        )
        secondary = FakeProvider("b")
        chain = ProviderChain([primary, secondary], max_retries=0)
        chain.complete([], 0.0, 5)   # primary fails, breaker opens
        chain.complete([], 0.0, 5)   # primary skipped
        assert primary.calls == 1
        assert secondary.calls == 2

    def test_all_fail_raises_last_error(self):
        primary = FakeProvider("a", outcomes=[ProviderError("first", retryable=False)])
        secondary = FakeProvider("b", outcomes=[ProviderError("second", retryable=False)])
        chain = ProviderChain([primary, secondary], max_retries=0)
        with pytest.raises(ProviderError, match="second"):
            chain.complete([], 0.0, 5)

    def test_budget_blocks_retry(self):
        primary = FakeProvider("a", outcomes=[ProviderError("slow", retryable=True)])
        chain = ProviderChain([primary], max_retries=3, backoff_base=100)
        with pytest.raises(ProviderError):
            chain.complete([], 0.0, 1)  # 1s budget < 100s backoff
        assert primary.calls == 1

    def test_empty_chain_raises(self):
        with pytest.raises(ProviderError):
            ProviderChain([]).complete([], 0.0, 5)

    def test_records_fallback_and_served_in_trace(self):
        from app.tracing import current_trace, start_trace

        trace = start_trace()
        primary = FakeProvider("a", outcomes=[ProviderError("boom", retryable=True)])
        secondary = FakeProvider("b", outcomes=[ProviderResult("ok", "b", "m2")])
        ProviderChain([primary, secondary], max_retries=0).complete([], 0.0, 5)
        assert trace.provider == "b"
        assert trace.fallbacks and trace.fallbacks[0]["provider"] == "a"
        assert current_trace() is trace


class TestRetryAfter:
    def test_parses_seconds(self):
        import httpx

        assert _parse_retry_after(httpx.Headers({"retry-after": "2"})) == 2.0

    def test_missing_or_invalid(self):
        import httpx

        assert _parse_retry_after(httpx.Headers({})) is None
        assert _parse_retry_after(httpx.Headers({"retry-after": "later"})) is None


class TestBuildDefaultChain:
    def test_explicit_chain_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER_CHAIN", "groq:model-a,ollama:model-b")
        chain = build_default_chain()
        assert [type(p) for p in chain] == [GroqProvider, OllamaProvider]
        assert chain[0].model == "model-a"
        assert chain[1].model == "model-b"

    def test_groq_primary_with_fallback(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER_CHAIN", raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "k")
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        monkeypatch.delenv("GROQ_FALLBACK_MODEL", raising=False)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        chain = build_default_chain()
        assert [p.model for p in chain] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

    def test_ollama_when_no_groq(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER_CHAIN", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
        chain = build_default_chain()
        assert len(chain) == 1 and isinstance(chain[0], OllamaProvider)
