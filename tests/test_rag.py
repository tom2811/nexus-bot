"""Tests for RAG pipeline."""

import pytest

from app.rag import RAGPipeline
from app.vector_store import VectorStore
from app.llm import LLMService


@pytest.fixture
def vector_store():
    """Create a test vector store."""
    store = VectorStore(persist_directory="./test_chroma_rag", collection_name="test_rag")
    yield store
    store.clear()


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "Our return policy allows returns within 30 days of purchase with original receipt.",
            "source": "policy.txt",
            "type": "txt",
            "chunk_id": "0",
        },
        {
            "content": "Shipping is free for orders over $50. Standard shipping takes 5-7 business days.",
            "source": "shipping.txt",
            "type": "txt",
            "chunk_id": "0",
        },
        {
            "content": "We accept Visa, Mastercard, American Express, and PayPal as payment methods.",
            "source": "payment.txt",
            "type": "txt",
            "chunk_id": "0",
        },
    ]


def test_rag_pipeline_initialization():
    """Test RAG pipeline can be initialized."""
    pipeline = RAGPipeline()
    assert pipeline.vector_store is not None
    assert pipeline.llm_service is not None
    assert pipeline.n_results == 5


def test_rag_pipeline_custom_components(vector_store):
    """Test RAG pipeline with custom components."""
    llm = LLMService()
    pipeline = RAGPipeline(vector_store=vector_store, llm_service=llm, n_results=3)
    
    assert pipeline.vector_store is vector_store
    assert pipeline.llm_service is llm
    assert pipeline.n_results == 3


def test_query_with_empty_store(vector_store):
    """Test querying with no documents returns appropriate response."""
    pipeline = RAGPipeline(vector_store=vector_store)
    result = pipeline.query("What is the return policy?")
    
    assert "answer" in result
    assert "sources" in result
    assert "context_used" in result
    assert result["context_used"] is False
    assert len(result["sources"]) == 0
    assert "don't have enough information" in result["answer"].lower()


def test_query_with_documents(vector_store, sample_documents):
    """Test querying with documents returns answer and sources."""
    vector_store.add_documents(sample_documents)
    pipeline = RAGPipeline(vector_store=vector_store)
    
    result = pipeline.query("What is the return policy?")
    
    assert "answer" in result
    assert "sources" in result
    assert "context_used" in result
    assert result["context_used"] is True
    assert len(result["sources"]) > 0
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_build_context(vector_store, sample_documents):
    """Test context building from retrieved documents."""
    vector_store.add_documents(sample_documents)
    pipeline = RAGPipeline(vector_store=vector_store)
    
    retrieved = vector_store.search("return policy", n_results=2)
    context = pipeline._build_context(retrieved)
    
    assert isinstance(context, str)
    assert len(context) > 0
    assert "[Source" in context


def test_build_prompt(vector_store):
    """Test prompt building."""
    pipeline = RAGPipeline(vector_store=vector_store)
    
    question = "What is the return policy?"
    context = "Returns are allowed within 30 days."
    prompt = pipeline._build_prompt(question, context)
    
    assert question in prompt
    assert context in prompt
    assert "Context" in prompt


def test_format_sources(vector_store, sample_documents):
    """Test source formatting."""
    vector_store.add_documents(sample_documents)
    pipeline = RAGPipeline(vector_store=vector_store)
    
    retrieved = vector_store.search("payment", n_results=2)
    sources = pipeline._format_sources(retrieved)
    
    assert isinstance(sources, list)
    assert len(sources) > 0
    for source in sources:
        assert "source" in source
        assert "type" in source


def test_format_sources_with_pages(vector_store):
    """Test source formatting with page numbers."""
    docs_with_pages = [
        {
            "content": "Chapter 1 content.",
            "source": "book.pdf",
            "type": "pdf",
            "page": 1,
            "chunk_id": "0",
        },
        {
            "content": "Chapter 2 content.",
            "source": "book.pdf",
            "type": "pdf",
            "page": 2,
            "chunk_id": "1",
        },
    ]
    vector_store.add_documents(docs_with_pages)
    pipeline = RAGPipeline(vector_store=vector_store)
    
    retrieved = vector_store.search("chapter", n_results=2)
    sources = pipeline._format_sources(retrieved)
    
    assert len(sources) == 2
    for source in sources:
        assert "page" in source


def test_format_sources_deduplication(vector_store):
    """Test that duplicate sources are removed."""
    duplicate_docs = [
        {
            "content": "First mention of policy.",
            "source": "policy.txt",
            "type": "txt",
            "chunk_id": "0",
        },
        {
            "content": "Second mention of policy.",
            "source": "policy.txt",
            "type": "txt",
            "chunk_id": "1",
        },
    ]
    vector_store.add_documents(duplicate_docs)
    pipeline = RAGPipeline(vector_store=vector_store)
    
    retrieved = vector_store.search("policy", n_results=2)
    sources = pipeline._format_sources(retrieved)
    
    # Should deduplicate same source without page numbers
    assert len(sources) == 1
    assert sources[0]["source"] == "policy.txt"
