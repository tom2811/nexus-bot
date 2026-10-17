"""Golden routing evaluation against the (dependency-free) intent detector.

Runs in CI without Postgres or an LLM: `detect_intent` is pure, so we exercise
it directly against `eval/golden.jsonl`.
"""

import json
from pathlib import Path

import pytest

from app.router import HybridRouter

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"


def load_golden() -> list[dict]:
    cases = []
    for line in GOLDEN_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def _intent(question: str) -> str:
    # detect_intent needs no DB/LLM; bypass __init__ to avoid a live Postgres.
    return HybridRouter.__new__(HybridRouter).detect_intent(question).value


@pytest.mark.parametrize(
    "case",
    [c for c in load_golden() if not c.get("known_mismatch")],
    ids=lambda c: c["id"],
)
def test_golden_route(case):
    assert _intent(case["question"]) == case["expected_route"]


def test_golden_routing_accuracy():
    gated = [c for c in load_golden() if not c.get("known_mismatch")]
    correct = sum(1 for c in gated if _intent(c["question"]) == c["expected_route"])
    accuracy = correct / len(gated)
    assert accuracy >= 0.95, f"routing accuracy {accuracy:.1%} below 95%"
