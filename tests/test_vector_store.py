"""Tests for vector store."""

import tempfile
import shutil
from pathlib import Path

import pytest

from app.vector_store import VectorStore


@pytest.fixture
def temp_store():
    """Create temporary vector store."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(persist_directory=temp_dir, collection_name="test_docs")
    yield store
    shutil.rmtree(temp_dir)


def test_add_and_search(temp_store):
    """Test adding documents and searching."""
    docs = [
        {"content": "The quick brown fox jumps over the lazy dog.", "source": "test1.txt", "type": "txt"},
        {"content": "Python is a programming language.", "source": "test2.txt", "type": "txt"},
    ]
    
    temp_store.add_documents(docs)
    results = temp_store.search("fox animal", n_results=2)
    
    assert len(results) == 2
    assert "fox" in results[0]["content"].lower()
    assert results[0]["metadata"]["source"] == "test1.txt"


def test_search_empty_store(temp_store):
    """Test searching empty store returns empty list."""
    results = temp_store.search("anything", n_results=5)
    assert results == []


def test_metadata_preservation(temp_store):
    """Test metadata is preserved."""
    docs = [
        {
            "content": "Page one content.",
            "source": "doc.pdf",
            "type": "pdf",
            "page": 1,
            "chunk_id": 0,
        }
    ]
    
    temp_store.add_documents(docs)
    results = temp_store.search("page", n_results=1)
    
    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "doc.pdf"
    assert results[0]["metadata"]["type"] == "pdf"
    assert results[0]["metadata"]["page"] == 1
    assert results[0]["metadata"]["chunk_id"] == 0
