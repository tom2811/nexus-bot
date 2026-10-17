"""Tests for evaluation metrics, dataset loading, and thresholds."""

from pathlib import Path

from app.evaluation import (
    EvaluationQuestion,
    Evaluator,
    QuestionType,
    load_dataset,
)

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"


class TestRetrievalMetrics:
    def test_precision_and_recall(self):
        precision, recall = Evaluator._retrieval_metrics(
            ["a.md", "b.md"],
            [{"source": "a.md"}, {"source": "c.md"}],
        )
        assert precision == 0.5
        assert recall == 0.5

    def test_perfect(self):
        precision, recall = Evaluator._retrieval_metrics(
            ["a.md"], [{"source": "a.md"}]
        )
        assert (precision, recall) == (1.0, 1.0)

    def test_none_when_no_expectation(self):
        assert Evaluator._retrieval_metrics(None, [{"source": "a.md"}]) == (None, None)

    def test_empty_retrieval_scores_zero(self):
        assert Evaluator._retrieval_metrics(["a.md"], []) == (0.0, 0.0)


class TestDatasetLoading:
    def test_loads_golden_jsonl(self):
        questions = load_dataset(GOLDEN)
        assert len(questions) >= 15
        assert all(isinstance(q, EvaluationQuestion) for q in questions)
        assert all(q.question_type in QuestionType for q in questions)

    def test_relevant_sources_parsed(self):
        by_id = {q.id: q for q in load_dataset(GOLDEN)}
        assert by_id["rag_01"].relevant_sources == ["return_policy.md"]


class TestThresholds:
    def test_reports_below_threshold(self):
        evaluation = {"route_accuracy": 0.5}
        failures = Evaluator.check_thresholds(evaluation, {"route_accuracy": 0.9})
        assert len(failures) == 1 and "route_accuracy" in failures[0]

    def test_reports_missing_metric(self):
        failures = Evaluator.check_thresholds({}, {"faithfulness": 0.8})
        assert "not measured" in failures[0]

    def test_passes_when_meeting_threshold(self):
        assert Evaluator.check_thresholds({"route_accuracy": 0.95}, {"route_accuracy": 0.9}) == []
