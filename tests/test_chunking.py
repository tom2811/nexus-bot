"""Tests for text chunking."""

from app.chunking import chunk_text, chunk_documents


def test_chunk_text_small():
    """Test chunking text smaller than chunk size."""
    text = "Short text"
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_large():
    """Test chunking large text with overlap."""
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    
    assert len(chunks) > 1
    assert all(len(chunk) == 100 for chunk in chunks[:-1])


def test_chunk_documents():
    """Test chunking documents with metadata preservation."""
    docs = [
        {"content": "a" * 200, "source": "test.txt", "type": "txt"},
    ]
    
    chunked = chunk_documents(docs, chunk_size=100, overlap=10)
    
    assert len(chunked) > 1
    assert all(doc["source"] == "test.txt" for doc in chunked)
    assert all(doc["type"] == "txt" for doc in chunked)
    assert all("chunk_id" in doc for doc in chunked)
