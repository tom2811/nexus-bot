"""Tests for the evaluation system."""

import pytest
from unittest.mock import Mock

from app.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    Evaluator,
    QuestionType,
)


class TestEvaluationDataset:
    """Test evaluation dataset."""
    
    def test_get_questions_returns_list(self):
        """Should return a list of questions."""
        questions = EvaluationDataset.get_questions()
        assert isinstance(questions, list)
        assert len(questions) > 0
    
    def test_all_questions_have_required_fields(self):
        """All questions should have required fields."""
        questions = EvaluationDataset.get_questions()
        
        for q in questions:
            assert q.id
            assert q.question
            assert q.question_type in QuestionType
            assert q.expected_route in ["rag_only", "tool_only", "hybrid"]
            assert isinstance(q.expected_tools, list)
            assert isinstance(q.expected_sources, bool)
            assert isinstance(q.parameters, dict)
    
    def test_question_ids_are_unique(self):
        """Question IDs should be unique."""
        questions = EvaluationDataset.get_questions()
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids))
    
    def test_has_rag_only_questions(self):
        """Should have RAG-only questions."""
        questions = EvaluationDataset.get_questions()
        rag_questions = [q for q in questions if q.question_type == QuestionType.RAG_ONLY]
        assert len(rag_questions) >= 5
    
    def test_has_tool_only_questions(self):
        """Should have tool-only questions."""
        questions = EvaluationDataset.get_questions()
        tool_questions = [q for q in questions if q.question_type == QuestionType.TOOL_ONLY]
        assert len(tool_questions) >= 5
    
    def test_has_hybrid_questions(self):
        """Should have hybrid questions."""
        questions = EvaluationDataset.get_questions()
        hybrid_questions = [q for q in questions if q.question_type == QuestionType.HYBRID]
        assert len(hybrid_questions) >= 5
    
    def test_has_no_info_questions(self):
        """Should have no-information questions."""
        questions = EvaluationDataset.get_questions()
        no_info_questions = [q for q in questions if q.question_type == QuestionType.NO_INFO]
        assert len(no_info_questions) >= 3
    
    def test_rag_questions_expect_sources(self):
        """RAG questions should expect sources."""
        questions = EvaluationDataset.get_questions()
        rag_questions = [q for q in questions if q.question_type == QuestionType.RAG_ONLY]
        
        for q in rag_questions:
            if q.question_type != QuestionType.NO_INFO:
                assert q.expected_sources is True
                assert q.expected_tools == []
    
    def test_tool_questions_have_parameters(self):
        """Tool questions should have required parameters."""
        questions = EvaluationDataset.get_questions()
        tool_questions = [q for q in questions if q.question_type == QuestionType.TOOL_ONLY]
        
        for q in tool_questions:
            assert len(q.parameters) > 0
            assert len(q.expected_tools) > 0
    
    def test_hybrid_questions_expect_both(self):
        """Hybrid questions should expect both sources and tools."""
        questions = EvaluationDataset.get_questions()
        hybrid_questions = [q for q in questions if q.question_type == QuestionType.HYBRID]
        
        for q in hybrid_questions:
            assert q.expected_sources is True
            assert len(q.expected_tools) > 0
            assert len(q.parameters) > 0


class TestEvaluator:
    """Test evaluator."""
    
    def test_evaluate_question_rag_correct(self):
        """Should correctly evaluate RAG-only question."""
        # Mock router
        router = Mock()
        router.query.return_value = {
            "answer": "Our return policy allows returns within 30 days.",
            "route_type": "rag_only",
            "rag_used": True,
            "tools_used": [],
            "sources": [{"source": "policy.txt", "chunk_id": "chunk_1"}],
            "tool_results": []
        }
        
        evaluator = Evaluator(router)
        
        question = EvaluationQuestion(
            id="test_01",
            question="What is the return policy?",
            question_type=QuestionType.RAG_ONLY,
            expected_route="rag_only",
            expected_tools=[],
            expected_sources=True,
            parameters={}
        )
        
        result = evaluator.evaluate_question(question)
        
        assert result.route_correct is True
        assert result.tools_correct is True
        assert result.sources_correct is True
        assert result.actual_route == "rag_only"
        assert result.has_sources is True
    
    def test_evaluate_question_tool_correct(self):
        """Should correctly evaluate tool-only question."""
        router = Mock()
        router.query.return_value = {
            "answer": "Air Max 90 is available in size 10.",
            "route_type": "tool_only",
            "rag_used": False,
            "tools_used": ["inventory"],
            "sources": [],
            "tool_results": [{"available": True, "quantity": 12}]
        }
        
        evaluator = Evaluator(router)
        
        question = EvaluationQuestion(
            id="test_02",
            question="Is Air Max available in size 10?",
            question_type=QuestionType.TOOL_ONLY,
            expected_route="tool_only",
            expected_tools=["inventory"],
            expected_sources=False,
            parameters={"product_id": 1, "size": "10"}
        )
        
        result = evaluator.evaluate_question(question)
        
        assert result.route_correct is True
        assert result.tools_correct is True
        assert result.sources_correct is True
        assert result.actual_route == "tool_only"
        assert "inventory" in result.actual_tools
    
    def test_evaluate_question_hybrid_correct(self):
        """Should correctly evaluate hybrid question."""
        router = Mock()
        router.query.return_value = {
            "answer": "Air Max is available. Return policy is 30 days.",
            "route_type": "hybrid",
            "rag_used": True,
            "tools_used": ["inventory"],
            "sources": [{"source": "policy.txt", "chunk_id": "chunk_1"}],
            "tool_results": [{"available": True, "quantity": 12}]
        }
        
        evaluator = Evaluator(router)
        
        question = EvaluationQuestion(
            id="test_03",
            question="Is Air Max available and what's the return policy?",
            question_type=QuestionType.HYBRID,
            expected_route="hybrid",
            expected_tools=["inventory"],
            expected_sources=True,
            parameters={"product_id": 1, "size": "10"}
        )
        
        result = evaluator.evaluate_question(question)
        
        assert result.route_correct is True
        assert result.tools_correct is True
        assert result.sources_correct is True
        assert result.has_sources is True
        assert len(result.actual_tools) > 0
    
    def test_evaluate_question_wrong_route(self):
        """Should detect incorrect routing."""
        router = Mock()
        router.query.return_value = {
            "answer": "Answer",
            "route_type": "rag_only",
            "rag_used": True,
            "tools_used": [],
            "sources": [{"source": "test.txt", "chunk_id": "chunk_1"}],
            "tool_results": []
        }
        
        evaluator = Evaluator(router)
        
        question = EvaluationQuestion(
            id="test_04",
            question="Check inventory",
            question_type=QuestionType.TOOL_ONLY,
            expected_route="tool_only",
            expected_tools=["inventory"],
            expected_sources=False,
            parameters={"product_id": 1, "size": "10"}
        )
        
        result = evaluator.evaluate_question(question)
        
        assert result.route_correct is False
        assert result.actual_route == "rag_only"
        assert result.expected_route == "tool_only"
    
    def test_evaluate_question_wrong_tools(self):
        """Should detect incorrect tool selection."""
        router = Mock()
        router.query.return_value = {
            "answer": "Answer",
            "route_type": "tool_only",
            "rag_used": False,
            "tools_used": ["search"],
            "sources": [],
            "tool_results": []
        }
        
        evaluator = Evaluator(router)
        
        question = EvaluationQuestion(
            id="test_05",
            question="Check inventory",
            question_type=QuestionType.TOOL_ONLY,
            expected_route="tool_only",
            expected_tools=["inventory"],
            expected_sources=False,
            parameters={"product_id": 1, "size": "10"}
        )
        
        result = evaluator.evaluate_question(question)
        
        assert result.tools_correct is False
        assert "search" in result.actual_tools
        assert "inventory" not in result.actual_tools
    
    def test_evaluate_question_wrong_sources(self):
        """Should detect incorrect source presence."""
        router = Mock()
        router.query.return_value = {
            "answer": "Answer",
            "route_type": "tool_only",
            "rag_used": False,
            "tools_used": ["inventory"],
            "sources": [{"source": "policy.txt", "chunk_id": "chunk_1"}],
            "tool_results": []
        }
        
        evaluator = Evaluator(router)
        
        question = EvaluationQuestion(
            id="test_06",
            question="Check inventory",
            question_type=QuestionType.TOOL_ONLY,
            expected_route="tool_only",
            expected_tools=["inventory"],
            expected_sources=False,
            parameters={"product_id": 1, "size": "10"}
        )
        
        result = evaluator.evaluate_question(question)
        
        assert result.sources_correct is False
        assert result.has_sources is True
        assert result.expected_sources is False
    
    def test_evaluate_all_returns_metrics(self):
        """Should return evaluation metrics."""
        router = Mock()
        router.query.return_value = {
            "answer": "Answer",
            "route_type": "rag_only",
            "rag_used": True,
            "tools_used": [],
            "sources": [{"source": "test.txt", "chunk_id": "chunk_1"}],
            "tool_results": []
        }
        
        evaluator = Evaluator(router)
        
        questions = [
            EvaluationQuestion(
                id="test_01",
                question="What is the return policy?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
            EvaluationQuestion(
                id="test_02",
                question="What is shipping?",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
        ]
        
        evaluation = evaluator.evaluate_all(questions)
        
        assert "total_questions" in evaluation
        assert "route_correct" in evaluation
        assert "tools_correct" in evaluation
        assert "sources_correct" in evaluation
        assert "route_accuracy" in evaluation
        assert "tools_accuracy" in evaluation
        assert "sources_accuracy" in evaluation
        assert "by_type" in evaluation
        assert "results" in evaluation
        
        assert evaluation["total_questions"] == 2
        assert evaluation["route_correct"] == 2
        assert evaluation["route_accuracy"] == 1.0
    
    def test_evaluate_all_calculates_accuracy(self):
        """Should calculate accuracy correctly."""
        router = Mock()
        
        # Mock correct and incorrect responses
        responses = [
            {  # Correct
                "answer": "Answer",
                "route_type": "rag_only",
                "rag_used": True,
                "tools_used": [],
                "sources": [{"source": "test.txt", "chunk_id": "chunk_1"}],
                "tool_results": []
            },
            {  # Wrong route
                "answer": "Answer",
                "route_type": "tool_only",
                "rag_used": False,
                "tools_used": [],
                "sources": [],
                "tool_results": []
            },
        ]
        router.query.side_effect = responses
        
        evaluator = Evaluator(router)
        
        questions = [
            EvaluationQuestion(
                id="test_01",
                question="Question 1",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
            EvaluationQuestion(
                id="test_02",
                question="Question 2",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
        ]
        
        evaluation = evaluator.evaluate_all(questions)
        
        assert evaluation["total_questions"] == 2
        assert evaluation["route_correct"] == 1
        assert evaluation["route_accuracy"] == 0.5
    
    def test_evaluate_all_groups_by_type(self):
        """Should group results by question type."""
        router = Mock()
        router.query.return_value = {
            "answer": "Answer",
            "route_type": "rag_only",
            "rag_used": True,
            "tools_used": [],
            "sources": [{"source": "test.txt", "chunk_id": "chunk_1"}],
            "tool_results": []
        }
        
        evaluator = Evaluator(router)
        
        questions = [
            EvaluationQuestion(
                id="test_01",
                question="RAG question",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
            EvaluationQuestion(
                id="test_02",
                question="Tool question",
                question_type=QuestionType.TOOL_ONLY,
                expected_route="tool_only",
                expected_tools=["inventory"],
                expected_sources=False,
                parameters={"product_id": 1, "size": "10"}
            ),
        ]
        
        evaluation = evaluator.evaluate_all(questions)
        
        assert "by_type" in evaluation
        assert "rag_only" in evaluation["by_type"]
        assert "tool_only" in evaluation["by_type"]
        
        assert evaluation["by_type"]["rag_only"]["total"] == 1
        assert evaluation["by_type"]["tool_only"]["total"] == 1
    
    def test_evaluate_all_handles_errors_gracefully(self):
        """Should continue evaluation even if some questions fail."""
        router = Mock()
        router.query.side_effect = [
            Exception("Error"),
            {
                "answer": "Answer",
                "route_type": "rag_only",
                "rag_used": True,
                "tools_used": [],
                "sources": [{"source": "test.txt", "chunk_id": "chunk_1"}],
                "tool_results": []
            }
        ]
        
        evaluator = Evaluator(router)
        
        questions = [
            EvaluationQuestion(
                id="test_01",
                question="Question 1",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
            EvaluationQuestion(
                id="test_02",
                question="Question 2",
                question_type=QuestionType.RAG_ONLY,
                expected_route="rag_only",
                expected_tools=[],
                expected_sources=True,
                parameters={}
            ),
        ]
        
        evaluation = evaluator.evaluate_all(questions)
        
        # Should have 1 result (second one succeeded)
        assert evaluation["total_questions"] == 1
        assert len(evaluation["results"]) == 1
