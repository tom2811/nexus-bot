"""Tests for per-request tracing."""

import json

from app import tracing
from app.tracing import Timer, current_trace, emit_trace, start_trace


def test_start_trace_sets_current():
    trace = start_trace(api_version="2")
    assert current_trace() is trace
    assert trace.api_version == "2"
    assert trace.trace_id


def test_record_served_updates_provider_and_model():
    trace = start_trace()
    tracing.record_served("groq", "openai/gpt-oss-120b")
    assert trace.provider == "groq"
    assert trace.model == "openai/gpt-oss-120b"


def test_record_fallback_appends_and_truncates_reason():
    trace = start_trace()
    tracing.record_fallback("groq", "m", "x" * 500)
    assert len(trace.fallbacks) == 1
    assert len(trace.fallbacks[0]["reason"]) == 160


def test_record_usage_ignores_all_none():
    trace = start_trace()
    tracing.record_usage(None, None)
    assert trace.prompt_tokens is None
    tracing.record_usage(10, 20)
    assert (trace.prompt_tokens, trace.completion_tokens) == (10, 20)


def test_to_dict_truncates_retrieval():
    trace = start_trace()
    trace.retrieval = [{"source": str(i)} for i in range(40)]
    assert len(trace.to_dict()["retrieval"]) == 25


def test_emit_trace_logs_json_line(caplog):
    trace = start_trace()
    trace.route = "rag_only"
    with caplog.at_level("INFO", logger="nexus.trace"):
        emit_trace(trace)
    assert "request_trace" in caplog.text
    payload = json.loads(caplog.text.split("request_trace", 1)[1].strip())
    assert payload["route"] == "rag_only"


def test_timer_measures_elapsed():
    with Timer() as timer:
        pass
    assert timer.ms is not None and timer.ms >= 0
