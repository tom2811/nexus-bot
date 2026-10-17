"""Evaluation system for Nexus-bot hybrid assistant.

Measures routing accuracy, tool selection, source attribution, retrieval
precision/recall, and (optionally) answer faithfulness via a judge.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from app.router import HybridRouter


class QuestionType(str, Enum):
    """Type of evaluation question."""
    RAG_ONLY = "rag_only"
    TOOL_ONLY = "tool_only"
    HYBRID = "hybrid"
    NO_INFO = "no_info"


class Judge(Protocol):
    """Scores how well an answer is supported by its context (0..1)."""

    def score(self, question: str, answer: str, context: str) -> float: ...


class EmbeddingJudge:
    """Faithfulness proxy via answer<->context embedding similarity.

    Dependency-light default: uses a local sentence-transformers model, so
    evaluation needs no external API. It is a proxy, not a verdict -- a
    semantic overlap score, not entailment.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, question: str, answer: str, context: str) -> float:
        import numpy as np

        if not answer.strip() or not context.strip():
            return 0.0
        model = self._load()
        answer_vec, context_vec = model.encode(
            [answer, context], normalize_embeddings=True
        )
        return float(np.clip(float(answer_vec @ context_vec), 0.0, 1.0))


class ModelJudge:
    """Faithfulness judge backed by the configured LLM (self-reported)."""

    def __init__(self, llm):
        self.llm = llm

    def score(self, question: str, answer: str, context: str) -> float:
        prompt = (
            "Rate from 0.0 to 1.0 how fully the answer is supported by the "
            "context. Reply with the number only.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\nScore:"
        )
        raw = self.llm.generate(prompt, temperature=0.0, timeout=60)
        match = re.search(r"\d*\.?\d+", raw or "")
        if not match:
            return 0.0
        return max(0.0, min(1.0, float(match.group())))


def _grounding_text(response: dict) -> str:
    """Assemble the context an answer should be grounded in."""
    parts: list[str] = []
    if response.get("context"):
        parts.append(str(response["context"]))
    for tool_result in response.get("tool_results", []) or []:
        parts.append(json.dumps(tool_result, default=str))
    return "\n".join(parts)


def load_dataset(path: str | Path) -> list["EvaluationQuestion"]:
    """Load an evaluation dataset from a JSONL file (one question per line)."""
    questions: list[EvaluationQuestion] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        raw = json.loads(line)
        questions.append(
            EvaluationQuestion(
                id=raw["id"],
                question=raw["question"],
                question_type=QuestionType(raw.get("question_type", "rag_only")),
                expected_route=raw["expected_route"],
                expected_tools=raw.get("expected_tools", []),
                expected_sources=raw.get("expected_sources", False),
                parameters=raw.get("parameters", {}),
                ground_truth=raw.get("ground_truth"),
                relevant_sources=raw.get("relevant_sources"),
                notes=raw.get("notes"),
            )
        )
    return questions


@dataclass
class EvaluationQuestion:
    """Single evaluation question with expected behavior."""
    id: str
    question: str
    question_type: QuestionType
    expected_route: str
    expected_tools: list[str]
    expected_sources: bool
    parameters: dict[str, Any]
    ground_truth: str | None = None
    relevant_sources: list[str] | None = None
    notes: str | None = None


@dataclass
class EvaluationResult:
    """Results from evaluating a single question."""
    question_id: str
    question: str
    actual_route: str
    expected_route: str
    route_correct: bool
    actual_tools: list[str]
    expected_tools: list[str]
    tools_correct: bool
    has_sources: bool
    expected_sources: bool
    sources_correct: bool
    answer: str
    sources: list[dict]
    tool_results: list[dict]
    ground_truth: str | None = None
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    faithfulness: float | None = None
    notes: str | None = None


class EvaluationDataset:
    """Collection of evaluation questions."""
    
    @staticmethod
    def get_questions() -> list[EvaluationQuestion]:
        """Get 45 evaluation questions covering RAG, tool, hybrid, and no-info cases."""
        return [
            # RAG-ONLY QUESTIONS
            EvaluationQuestion(
                id="rag_01",
                question="What is the return policy?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Basic policy retrieval"
            ),
            EvaluationQuestion(
                id="rag_02",
                question="How long do I have to return an item?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Specific return policy detail"
            ),
            EvaluationQuestion(
                id="rag_03",
                question="What is the shipping policy?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Shipping information retrieval"
            ),
            EvaluationQuestion(
                id="rag_04",
                question="Do you offer international shipping?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Specific shipping detail"
            ),
            EvaluationQuestion(
                id="rag_05",
                question="What payment methods do you accept?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Payment policy"
            ),
            EvaluationQuestion(
                id="rag_06",
                question="How do I track my order?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="General tracking information"
            ),
            EvaluationQuestion(
                id="rag_07",
                question="What is your warranty policy?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Warranty information"
            ),
            EvaluationQuestion(
                id="rag_08",
                question="Can I exchange items?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Exchange policy"
            ),
            EvaluationQuestion(
                id="rag_09",
                question="What are your business hours?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Business information"
            ),
            EvaluationQuestion(
                id="rag_10",
                question="How do I contact customer support?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={},
                notes="Support contact information"
            ),
            
            # TOOL-ONLY QUESTIONS (Database Queries)
            EvaluationQuestion(
                id="tool_01",
                question="Is Air Max available in size 10?",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["inventory"],
                expected_sources=False,
                parameters={"product_id": 1, "size": "10"},
                notes="Inventory check with product ID"
            ),
            EvaluationQuestion(
                id="tool_02",
                question="Check inventory for product 2 in size 9",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["inventory"],
                expected_sources=False,
                parameters={"product_id": 2, "size": "9"},
                notes="Direct inventory query"
            ),
            EvaluationQuestion(
                id="tool_03",
                question="Do you have Jordan 1 in stock?",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["search"],
                expected_sources=False,
                parameters={"search_query": "Jordan 1"},
                notes="Product search query"
            ),
            EvaluationQuestion(
                id="tool_04",
                question="Search for Nike products",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["search"],
                expected_sources=False,
                parameters={"search_query": "Nike"},
                notes="Brand search"
            ),
            EvaluationQuestion(
                id="tool_05",
                question="What is the status of order 1001?",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["order"],
                expected_sources=False,
                parameters={"order_id": 1001},
                notes="Order status lookup"
            ),
            EvaluationQuestion(
                id="tool_06",
                question="Check my order 1002",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["order"],
                expected_sources=False,
                parameters={"order_id": 1002},
                notes="Order status with order ID"
            ),
            EvaluationQuestion(
                id="tool_07",
                question="Is product 3 available in size 8.5?",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["inventory"],
                expected_sources=False,
                parameters={"product_id": 3, "size": "8.5"},
                notes="Inventory with half size"
            ),
            EvaluationQuestion(
                id="tool_08",
                question="Find running shoes",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["search"],
                expected_sources=False,
                parameters={"search_query": "running"},
                notes="Category search"
            ),
            EvaluationQuestion(
                id="tool_09",
                question="How much inventory for product 1 size 11?",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["inventory"],
                expected_sources=False,
                parameters={"product_id": 1, "size": "11"},
                notes="Quantity-focused inventory question"
            ),
            EvaluationQuestion(
                id="tool_10",
                question="Where is order 1003?",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["order"],
                expected_sources=False,
                parameters={"order_id": 1003},
                notes="Order tracking question"
            ),
            
            # HYBRID QUESTIONS (RAG + Tools)
            EvaluationQuestion(
                id="hybrid_01",
                question="Is Air Max available in size 10 and what is your return policy?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["inventory"],
                expected_sources=True,
                parameters={"product_id": 1, "size": "10"},
                notes="Inventory + return policy"
            ),
            EvaluationQuestion(
                id="hybrid_02",
                question="Check inventory for product 2 size 9 and tell me about shipping",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["inventory"],
                expected_sources=True,
                parameters={"product_id": 2, "size": "9"},
                notes="Inventory + shipping policy"
            ),
            EvaluationQuestion(
                id="hybrid_03",
                question="What's the status of order 1001 and can I return it?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["order"],
                expected_sources=True,
                parameters={"order_id": 1001},
                notes="Order status + return policy"
            ),
            EvaluationQuestion(
                id="hybrid_04",
                question="Find Nike products and tell me about your warranty",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["search"],
                expected_sources=True,
                parameters={"search_query": "Nike"},
                notes="Product search + warranty policy"
            ),
            EvaluationQuestion(
                id="hybrid_05",
                question="Is product 3 in stock and what payment methods do you accept?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["inventory"],
                expected_sources=True,
                parameters={"product_id": 3, "size": "9"},
                notes="Inventory + payment policy"
            ),
            EvaluationQuestion(
                id="hybrid_06",
                question="Check order 1002 and how do I track it?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["order"],
                expected_sources=True,
                parameters={"order_id": 1002},
                notes="Order status + tracking information"
            ),
            EvaluationQuestion(
                id="hybrid_07",
                question="Search for running shoes and tell me about international shipping",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["search"],
                expected_sources=True,
                parameters={"search_query": "running"},
                notes="Product search + international shipping"
            ),
            EvaluationQuestion(
                id="hybrid_08",
                question="Is Air Max available and what are your business hours?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["inventory"],
                expected_sources=True,
                parameters={"product_id": 1, "size": "10"},
                notes="Inventory + business hours"
            ),
            EvaluationQuestion(
                id="hybrid_09",
                question="Find Jordan products and can I exchange them?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["search"],
                expected_sources=True,
                parameters={"search_query": "Jordan"},
                notes="Product search + exchange policy"
            ),
            EvaluationQuestion(
                id="hybrid_10",
                question="Check product 1 availability and how do I contact support?",
                question_type=QuestionType.HYBRID,
                expected_route="hybrid",
                expected_tools=["inventory"],
                expected_sources=True,
                parameters={"product_id": 1, "size": "10"},
                notes="Inventory + contact information"
            ),
            
            # NO-INFORMATION QUESTIONS (Should handle gracefully)
            EvaluationQuestion(
                id="no_info_01",
                question="What is the weather today?",
                question_type=QuestionType.NO_INFO,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=False,
                parameters={},
                notes="Out of domain - should indicate no information"
            ),
            EvaluationQuestion(
                id="no_info_02",
                question="Who won the World Cup?",
                question_type=QuestionType.NO_INFO,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=False,
                parameters={},
                notes="Out of domain - sports"
            ),
            EvaluationQuestion(
                id="no_info_03",
                question="What is the meaning of life?",
                question_type=QuestionType.NO_INFO,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=False,
                parameters={},
                notes="Philosophical - out of scope"
            ),
            EvaluationQuestion(
                id="no_info_04",
                question="Tell me a joke",
                question_type=QuestionType.NO_INFO,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=False,
                parameters={},
                notes="Entertainment - out of scope"
            ),
            EvaluationQuestion(
                id="no_info_05",
                question="What is product 999 availability?",
                question_type=QuestionType.NO_INFO,
                expected_route="tool_only",
                expected_tools=["inventory"],
                expected_sources=False,
                parameters={"product_id": 999, "size": "10"},
                notes="Non-existent product - should handle gracefully"
            ),
        ]


class Evaluator:
    """Evaluates hybrid assistant performance."""

    def __init__(self, router: HybridRouter, judge: Judge | None = None):
        """Initialize evaluator.

        Args:
            router: HybridRouter instance to evaluate
            judge: Optional faithfulness judge (EmbeddingJudge/ModelJudge)
        """
        self.router = router
        self.judge = judge

    def evaluate_question(self, question: EvaluationQuestion) -> EvaluationResult:
        """Evaluate a single question.

        Args:
            question: Question to evaluate

        Returns:
            Evaluation result with routing and quality metrics
        """
        response = self.router.query(question.question, **question.parameters)

        route_correct = response["route_type"] == question.expected_route

        actual_tools = set(response.get("tools_used", []))
        expected_tools = set(question.expected_tools)
        tools_correct = actual_tools == expected_tools

        has_sources = len(response.get("sources", [])) > 0
        sources_correct = has_sources == question.expected_sources

        precision, recall = self._retrieval_metrics(
            question.relevant_sources, response.get("retrieval", [])
        )

        faithfulness = None
        if self.judge is not None:
            faithfulness = self.judge.score(
                question.question,
                response.get("answer", ""),
                _grounding_text(response),
            )

        return EvaluationResult(
            question_id=question.id,
            question=question.question,
            actual_route=response["route_type"],
            expected_route=question.expected_route,
            route_correct=route_correct,
            actual_tools=list(actual_tools),
            expected_tools=list(expected_tools),
            tools_correct=tools_correct,
            has_sources=has_sources,
            expected_sources=question.expected_sources,
            sources_correct=sources_correct,
            answer=response["answer"],
            sources=response.get("sources", []),
            tool_results=response.get("tool_results", []),
            ground_truth=question.ground_truth,
            retrieval_precision=precision,
            retrieval_recall=recall,
            faithfulness=faithfulness,
            notes=question.notes,
        )

    @staticmethod
    def _retrieval_metrics(
        relevant: list[str] | None, retrieval: list[dict]
    ) -> tuple[float | None, float | None]:
        """Precision/recall of retrieved sources vs. an expected set."""
        if relevant is None:
            return None, None
        expected = set(relevant)
        retrieved = [r.get("source") for r in retrieval if r.get("source")]
        hits = sum(1 for source in retrieved if source in expected)
        precision = hits / len(retrieved) if retrieved else 0.0
        recall = hits / len(expected) if expected else 0.0
        return precision, recall

    def evaluate_all(self, questions: list[EvaluationQuestion] | None = None) -> dict:
        """Evaluate all questions and return summary metrics.

        Args:
            questions: Optional list of questions (uses default dataset if None)

        Returns:
            Dictionary with evaluation metrics and results
        """
        if questions is None:
            questions = EvaluationDataset.get_questions()

        paired: list[tuple[EvaluationQuestion, EvaluationResult]] = []
        for question in questions:
            try:
                paired.append((question, self.evaluate_question(question)))
            except Exception as e:
                print(f"Error evaluating {question.id}: {e}")
                continue

        results = [result for _, result in paired]
        total = len(results)

        def count(pred) -> int:
            return sum(1 for r in results if pred(r))

        route_correct = count(lambda r: r.route_correct)
        tools_correct = count(lambda r: r.tools_correct)
        sources_correct = count(lambda r: r.sources_correct)

        def mean(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        retrieval_precision = mean(
            [r.retrieval_precision for r in results if r.retrieval_precision is not None]
        )
        retrieval_recall = mean(
            [r.retrieval_recall for r in results if r.retrieval_recall is not None]
        )
        faithfulness = mean(
            [r.faithfulness for r in results if r.faithfulness is not None]
        )

        by_type = {}
        for q_type in QuestionType:
            type_results = [r for q, r in paired if q.question_type == q_type]
            if type_results:
                n = len(type_results)
                by_type[q_type.value] = {
                    "total": n,
                    "route_correct": sum(1 for r in type_results if r.route_correct),
                    "tools_correct": sum(1 for r in type_results if r.tools_correct),
                    "sources_correct": sum(1 for r in type_results if r.sources_correct),
                    "route_accuracy": sum(1 for r in type_results if r.route_correct) / n,
                    "tools_accuracy": sum(1 for r in type_results if r.tools_correct) / n,
                    "sources_accuracy": sum(1 for r in type_results if r.sources_correct) / n,
                }

        return {
            "total_questions": total,
            "route_correct": route_correct,
            "tools_correct": tools_correct,
            "sources_correct": sources_correct,
            "route_accuracy": route_correct / total if total > 0 else 0,
            "tools_accuracy": tools_correct / total if total > 0 else 0,
            "sources_accuracy": sources_correct / total if total > 0 else 0,
            "retrieval_precision": retrieval_precision,
            "retrieval_recall": retrieval_recall,
            "faithfulness": faithfulness,
            "by_type": by_type,
            "results": results,
        }

    @staticmethod
    def check_thresholds(
        evaluation: dict, thresholds: dict[str, float]
    ) -> list[str]:
        """Return human-readable failures for any metric below its threshold."""
        failures = []
        for metric, minimum in thresholds.items():
            if minimum is None:
                continue
            value = evaluation.get(metric)
            if value is None:
                failures.append(f"{metric}: not measured (threshold {minimum:.0%})")
            elif value < minimum:
                failures.append(
                    f"{metric}: {value:.1%} < required {minimum:.0%}"
                )
        return failures

    def print_summary(self, evaluation: dict):
        """Print evaluation summary.
        
        Args:
            evaluation: Evaluation results from evaluate_all()
        """
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nTotal Questions: {evaluation['total_questions']}")
        print(f"Route Accuracy: {evaluation['route_accuracy']:.1%} ({evaluation['route_correct']}/{evaluation['total_questions']})")
        print(f"Tool Selection: {evaluation['tools_accuracy']:.1%} ({evaluation['tools_correct']}/{evaluation['total_questions']})")
        print(f"Source Correctness: {evaluation['sources_accuracy']:.1%} ({evaluation['sources_correct']}/{evaluation['total_questions']})")

        if evaluation.get("retrieval_precision") is not None:
            print(f"Retrieval Precision: {evaluation['retrieval_precision']:.1%}")
        if evaluation.get("retrieval_recall") is not None:
            print(f"Retrieval Recall: {evaluation['retrieval_recall']:.1%}")
        if evaluation.get("faithfulness") is not None:
            print(f"Faithfulness (judge proxy): {evaluation['faithfulness']:.1%}")
        
        print("\n" + "-"*80)
        print("BY QUESTION TYPE")
        print("-"*80)
        
        for q_type, metrics in evaluation['by_type'].items():
            print(f"\n{q_type.upper().replace('_', ' ')} ({metrics['total']} questions):")
            print(f"  Route: {metrics['route_accuracy']:.1%}")
            print(f"  Tools: {metrics['tools_accuracy']:.1%}")
            print(f"  Sources: {metrics['sources_accuracy']:.1%}")
        
        print("\n" + "="*80)
    
    def print_detailed_results(self, evaluation: dict, show_correct: bool = False):
        """Print detailed results for each question.
        
        Args:
            evaluation: Evaluation results from evaluate_all()
            show_correct: If True, show correct results too (default: only errors)
        """
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for result in evaluation['results']:
            if not show_correct and result.route_correct and result.tools_correct and result.sources_correct:
                continue
            
            print(f"\n[{result.question_id}] {result.question}")
            print(f"Notes: {result.notes}")
            
            issues = []
            if not result.route_correct:
                issues.append(f"Route: expected {result.expected_route}, got {result.actual_route}")
            if not result.tools_correct:
                issues.append(f"Tools: expected {result.expected_tools}, got {result.actual_tools}")
            if not result.sources_correct:
                issues.append(f"Sources: expected {result.expected_sources}, got {result.has_sources}")
            
            if issues:
                print("FAILED:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("PASSED")
            
            answer = result.answer[:100] + "..." if len(result.answer) > 100 else result.answer
            print(f"Answer: {answer}")
        
        print("\n" + "="*80)
