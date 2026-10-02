"""Vector store using ChromaDB."""

from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from app.embeddings import EmbeddingService


class VectorStore:
    """ChromaDB vector store for document retrieval."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        """Initialize ChromaDB with persistent storage."""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_service = EmbeddingService()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_service.embedding_function,
        )
    
    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
        
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc["content"] for doc in documents]
        
        # Build metadata
        metadatas = []
        for doc in documents:
            metadata = {
                "source": doc["source"],
                "type": doc["type"],
            }
            if "page" in doc:
                metadata["page"] = doc["page"]
            if "chunk_id" in doc:
                metadata["chunk_id"] = doc["chunk_id"]
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, any]]:
        """Search for similar documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        # Format results
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        
        return formatted
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_service.embedding_function,
        )
