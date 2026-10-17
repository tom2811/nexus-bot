#!/usr/bin/env python3
"""Run the Nexus-bot evaluation harness.

Backends
--------
live (default when a database is reachable): real Postgres + vector store + LLM.
fixture: deterministic in-process router that answers straight from the dataset.
    This is a HARNESS SELF-TEST (dataset loading, metrics, thresholds, JSON
    output) -- it does not measure the model. Use it in CI; use `live` (or the
    mocked unit tests) to judge answer quality.

Exit codes: 0 = passed, 1 = thresholds failed, 2 = environment unavailable.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.evaluation import (  # noqa: E402
    EvaluationDataset,
    Evaluator,
    EmbeddingJudge,
    ModelJudge,
    load_dataset,
)

DEFAULT_DATASET = Path(__file__).parent / "eval" / "golden.jsonl"


class FixtureRouter:
    """Answers from the dataset's own expectations (harness smoke test only)."""

    def __init__(self, questions):
        self._by_text = {q.question: q for q in questions}

    def query(self, question, **kwargs):
        expected = self._by_text.get(question)
        if expected is None:
            return {
                "answer": "(fixture)",
                "route_type": "rag_only",
                "rag_used": True,
                "tools_used": [],
                "sources": [],
                "tool_results": [],
                "retrieval": [],
                "context": "fixture",
            }
        sources = (
            [{"source": s} for s in (expected.relevant_sources or [])]
            if expected.expected_sources
            else []
        )
        retrieval = [{"source": s} for s in (expected.relevant_sources or [])]
        return {
            "answer": f"(fixture answer for {expected.id})",
            "route_type": expected.expected_route,
            "rag_used": expected.expected_sources,
            "tools_used": list(expected.expected_tools),
            "sources": sources,
            "tool_results": [],
            "retrieval": retrieval,
            "context": "fixture context",
        }


def build_live_router():
    """Build the real router. Raises on unreachable Postgres."""
    from app.database import DatabaseConnection
    from app.llm import LLMService
    from app.rag import RAGPipeline
    from app.router import HybridRouter
    from app.vector_store import VectorStore

    db = DatabaseConnection(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "nexus_bot"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    vector_store = VectorStore(
        persist_directory=os.getenv("VECTOR_PERSIST_DIR", "./chroma_db")
    )
    llm = LLMService()
    rag = RAGPipeline(vector_store=vector_store, llm_service=llm)
    return HybridRouter(rag_pipeline=rag, db=db, llm=llm)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Nexus-bot evaluation")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                        help="JSONL dataset path")
    parser.add_argument("--backend", choices=["auto", "live", "fixture"],
                        default="auto")
    parser.add_argument("--judge", choices=["embedding", "model", "none"],
                        default="embedding", help="faithfulness judge")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--show-correct", action="store_true")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write full results to this JSON file")
    parser.add_argument("--min-route", type=float, default=0.95)
    parser.add_argument("--min-tools", type=float, default=None)
    parser.add_argument("--min-sources", type=float, default=None)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    questions = (
        load_dataset(args.dataset) if Path(args.dataset).exists()
        else EvaluationDataset.get_questions()
    )
    print(f"\nNEXUS-BOT EVALUATION  ({len(questions)} questions from {args.dataset})")
    print("=" * 80)

    backend = args.backend
    if backend == "auto":
        try:
            router = build_live_router()
            backend = "live"
        except Exception as e:  # noqa: BLE001 - report any bootstrap failure
            print(f"live backend unavailable ({type(e).__name__}: {e})")
            router = FixtureRouter(questions)
            backend = "fixture"
    elif backend == "live":
        try:
            router = build_live_router()
        except Exception as e:  # noqa: BLE001
            print(f"live backend unavailable: {type(e).__name__}: {e}")
            return 2
    else:
        router = FixtureRouter(questions)

    if backend == "fixture":
        print("BACKEND: fixture -- HARNESS SELF-TEST, does NOT measure the model")
    else:
        print("BACKEND: live")

    judge = None
    if args.judge == "embedding":
        judge = EmbeddingJudge()
    elif args.judge == "model":
        judge = ModelJudge(router.llm if backend == "live" else None)

    evaluator = Evaluator(router, judge=judge)
    evaluation = evaluator.evaluate_all(questions)
    evaluator.print_summary(evaluation)

    if args.detailed:
        evaluator.print_detailed_results(evaluation, show_correct=args.show_correct)

    if args.json_path:
        payload = {
            k: v for k, v in evaluation.items() if k != "results"
        }
        payload["results"] = [vars(r) for r in evaluation["results"]]
        Path(args.json_path).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {args.json_path}")

    thresholds = {"route_accuracy": args.min_route}
    if backend == "live":
        thresholds.update({
            "tools_accuracy": args.min_tools,
            "sources_accuracy": args.min_sources,
            "faithfulness": args.min_faithfulness,
        })

    failures = Evaluator.check_thresholds(evaluation, thresholds)
    if failures:
        print("\nFAILED thresholds:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll thresholds passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
