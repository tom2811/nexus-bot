"""Evaluation system for Nexus-bot hybrid assistant.

Measures routing accuracy, tool selection, and source attribution.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.router import HybridRouter


class QuestionType(str, Enum):
    """Type of evaluation question."""
    RAG_ONLY = "rag_only"
    TOOL_ONLY = "tool_only"
    HYBRID = "hybrid"
    NO_INFO = "no_info"


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
    
    def __init__(self, router: HybridRouter):
        """Initialize evaluator.
        
        Args:
            router: HybridRouter instance to evaluate
        """
        self.router = router
    
    def evaluate_question(self, question: EvaluationQuestion) -> EvaluationResult:
        """Evaluate a single question.
        
        Args:
            question: Question to evaluate
            
        Returns:
            Evaluation result with routing and quality metrics
        """
        response = self.router.query(
            message=question.question,
            **question.parameters
        )
        
        route_correct = response["route_type"] == question.expected_route
        
        actual_tools = set(response.get("tools_used", []))
        expected_tools = set(question.expected_tools)
        tools_correct = actual_tools == expected_tools
        
        has_sources = len(response.get("sources", [])) > 0
        sources_correct = has_sources == question.expected_sources
        
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
            notes=question.notes
        )
    
    def evaluate_all(self, questions: list[EvaluationQuestion] | None = None) -> dict:
        """Evaluate all questions and return summary metrics.
        
        Args:
            questions: Optional list of questions (uses default dataset if None)
            
        Returns:
            Dictionary with evaluation metrics and results
        """
        if questions is None:
            questions = EvaluationDataset.get_questions()
        
        results = []
        for question in questions:
            try:
                result = self.evaluate_question(question)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {question.id}: {e}")
                continue
        
        total = len(results)
        route_correct = sum(1 for r in results if r.route_correct)
        tools_correct = sum(1 for r in results if r.tools_correct)
        sources_correct = sum(1 for r in results if r.sources_correct)
        
        by_type = {}
        for q_type in QuestionType:
            type_results = [r for r, q in zip(results, questions) if q.question_type == q_type]
            if type_results:
                by_type[q_type.value] = {
                    "total": len(type_results),
                    "route_correct": sum(1 for r in type_results if r.route_correct),
                    "tools_correct": sum(1 for r in type_results if r.tools_correct),
                    "sources_correct": sum(1 for r in type_results if r.sources_correct),
                    "route_accuracy": sum(1 for r in type_results if r.route_correct) / len(type_results),
                    "tools_accuracy": sum(1 for r in type_results if r.tools_correct) / len(type_results),
                    "sources_accuracy": sum(1 for r in type_results if r.sources_correct) / len(type_results),
                }
        
        return {
            "total_questions": total,
            "route_correct": route_correct,
            "tools_correct": tools_correct,
            "sources_correct": sources_correct,
            "route_accuracy": route_correct / total if total > 0 else 0,
            "tools_accuracy": tools_correct / total if total > 0 else 0,
            "sources_accuracy": sources_correct / total if total > 0 else 0,
            "by_type": by_type,
            "results": results,
        }
    
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
