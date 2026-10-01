"""Tests for document loaders."""

from pathlib import Path

from app.loaders import load_text, load_markdown, load_document


def test_load_text():
    """Test loading a plain text file."""
    file_path = "tests/test_documents/sample.txt"
    docs = load_text(file_path)
    
    assert len(docs) == 1
    assert docs[0]["type"] == "txt"
    assert docs[0]["source"] == file_path
    assert "sample text document" in docs[0]["content"]


def test_load_markdown():
    """Test loading a markdown file."""
    file_path = "tests/test_documents/sample.md"
    docs = load_markdown(file_path)
    
    assert len(docs) == 1
    assert docs[0]["type"] == "markdown"
    assert docs[0]["source"] == file_path
    assert "# Sample Markdown" in docs[0]["content"]


def test_load_document_txt():
    """Test generic document loader with txt file."""
    file_path = "tests/test_documents/sample.txt"
    docs = load_document(file_path)
    
    assert len(docs) == 1
    assert docs[0]["type"] == "txt"


def test_load_document_markdown():
    """Test generic document loader with markdown file."""
    file_path = "tests/test_documents/sample.md"
    docs = load_document(file_path)
    
    assert len(docs) == 1
    assert docs[0]["type"] == "markdown"
