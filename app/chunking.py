"""Text chunking utilities."""

from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def chunk_documents(documents: List[Dict[str, str]], 
                    chunk_size: int = 500, 
                    overlap: int = 50) -> List[Dict[str, str]]:
    """Chunk documents and preserve metadata."""
    chunked = []
    
    for doc in documents:
        text = doc["content"]
        chunks = chunk_text(text, chunk_size, overlap)
        
        for idx, chunk in enumerate(chunks):
            chunked_doc = doc.copy()
            chunked_doc["content"] = chunk
            chunked_doc["chunk_id"] = idx
            chunked.append(chunked_doc)
    
    return chunked
