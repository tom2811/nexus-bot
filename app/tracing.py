"""Per-request structured tracing.

One JSON line per request to stdout (Railway/log aggregators capture stdout).
Fields: route, retrieval scores, tools, provider/model, tokens, latency, fallbacks.
"""

import contextvars
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

logger = logging.getLogger("nexus.trace")

# Fields whose text can contain raw user/model content — never logged verbatim.
_MAX_LIST = 25


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    api_version: str = "1"
    route: Optional[str] = None
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["retrieval"] = d["retrieval"][:_MAX_LIST]
        return d


_current: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "nexus_trace", default=None
)


def start_trace(api_version: str = "1") -> Trace:
    t = Trace(api_version=api_version)
    _current.set(t)
    return t


def current_trace() -> Optional[Trace]:
    return _current.get()


def record_served(provider: str, model: str) -> None:
    t = _current.get()
    if t:
        t.provider = provider
        t.model = model


def record_fallback(provider: str, model: str, reason: str) -> None:
    t = _current.get()
    if t:
        t.fallbacks.append({"provider": provider, "model": model, "reason": reason[:160]})


def record_usage(prompt_tokens: Optional[int], completion_tokens: Optional[int]) -> None:
    t = _current.get()
    if t and (prompt_tokens is not None or completion_tokens is not None):
        t.prompt_tokens = prompt_tokens
        t.completion_tokens = completion_tokens


def emit_trace(trace: Trace) -> None:
    logger.info("request_trace %s", json.dumps(trace.to_dict(), default=str))


class Timer:
    """Context manager measuring wall-clock latency in milliseconds."""

    def __init__(self) -> None:
        self.ms: Optional[float] = None
        self._t0 = time.perf_counter()

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = round((time.perf_counter() - self._t0) * 1000, 1)
