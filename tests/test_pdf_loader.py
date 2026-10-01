"""Tests for PDF document loader."""

from app.loaders import load_pdf, load_document


def test_load_pdf():
    """Test loading a PDF file."""
    file_path = "tests/test_documents/sample.pdf"
    docs = load_pdf(file_path)
    
    assert len(docs) == 2  # 2 pages
    assert all(doc["type"] == "pdf" for doc in docs)
    assert all(doc["source"] == file_path for doc in docs)
    assert docs[0]["page"] == 1
    assert docs[1]["page"] == 2
    assert "page 1" in docs[0]["content"].lower()
    assert "page 2" in docs[1]["content"].lower()


def test_load_document_pdf():
    """Test generic document loader with PDF file."""
    file_path = "tests/test_documents/sample.pdf"
    docs = load_document(file_path)
    
    assert len(docs) == 2
    assert docs[0]["type"] == "pdf"
