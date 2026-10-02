"""Tests for embedding service."""

from app.embeddings import EmbeddingService


def test_embed_text():
    """Test embedding a single text."""
    service = EmbeddingService()
    embedding = service.embed_text("This is a test sentence.")
    
    assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
    assert all(isinstance(float(x), float) for x in embedding)


def test_embed_texts():
    """Test embedding multiple texts."""
    service = EmbeddingService()
    texts = ["First sentence.", "Second sentence."]
    embeddings = service.embed_texts(texts)
    
    assert len(embeddings) == 2
    assert all(len(emb) == 384 for emb in embeddings)
