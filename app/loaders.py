"""Document loaders for various file formats."""

from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def load_text(file_path: str) -> List[Dict[str, str]]:
    """Load plain text file."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return [{"content": text, "source": file_path, "type": "txt"}]


def load_markdown(file_path: str) -> List[Dict[str, str]]:
    """Load markdown file."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    return [{"content": text, "source": file_path, "type": "markdown"}]


def load_pdf(file_path: str) -> List[Dict[str, str]]:
    """Load PDF file, one chunk per page."""
    reader = PdfReader(file_path)
    chunks = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            chunks.append({
                "content": text,
                "source": file_path,
                "type": "pdf",
                "page": page_num,
            })
    
    return chunks


def load_document(file_path: str) -> List[Dict[str, str]]:
    """Load document based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".md":
        return load_markdown(file_path)
    elif suffix == ".txt":
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
