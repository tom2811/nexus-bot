#!/usr/bin/env python3
"""Run evaluation of Nexus-bot hybrid assistant."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import get_db_connection
from app.embeddings import get_embedding_function
from app.evaluation import EvaluationDataset, Evaluator
from app.llm import LLMService
from app.rag import RAGPipeline
from app.router import HybridRouter
from app.tools import InventoryTool, OrderStatusTool, ProductSearchTool
from app.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Run Nexus-bot evaluation")
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--show-correct", action="store_true")
    
    args = parser.parse_args()
    
    print("\nNEXUS-BOT EVALUATION")
    print("=" * 80)
    print("Initializing components...")
    
    try:
        db_conn = get_db_connection()
        embedding_fn = get_embedding_function()
        
        vector_persist_dir = os.getenv("VECTOR_PERSIST_DIR", "./chroma_db")
        vector_store = VectorStore(
            embedding_function=embedding_fn,
            persist_directory=vector_persist_dir
        )
        
        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        llm_service = LLMService(host=ollama_host, model=ollama_model)
        
        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            llm_service=llm_service
        )
        
        tools = {
            "inventory": InventoryTool(db_conn),
            "search": ProductSearchTool(db_conn),
            "order": OrderStatusTool(db_conn),
        }
        
        router = HybridRouter(
            rag_pipeline=rag_pipeline,
            tools=tools,
            llm_service=llm_service
        )
        
        print("Components initialized")
        
        questions = EvaluationDataset.get_questions()
        print(f"Loaded {len(questions)} evaluation questions")
        
        evaluator = Evaluator(router)
        
        print("\nRunning evaluation...\n")
        
        evaluation = evaluator.evaluate_all(questions)
        
        evaluator.print_summary(evaluation)
        
        if args.detailed:
            evaluator.print_detailed_results(
                evaluation,
                show_correct=args.show_correct
            )
        
        threshold = 0.8
        if evaluation["route_accuracy"] >= threshold:
            print(f"\nEvaluation passed (route accuracy >= {threshold:.0%})")
            return 0
        else:
            print(f"\nRoute accuracy below threshold: {evaluation['route_accuracy']:.1%}")
            return 1
    
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
