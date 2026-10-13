"""RAG pipeline combining retrieval and LLM generation."""

from typing import Dict, List, Optional

from app.vector_store import VectorStore
from app.llm import LLMService


class RAGPipeline:
    """RAG pipeline for question answering with source citations."""
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_service: Optional[LLMService] = None,
        n_results: int = 5,
    ):
        """Initialize RAG pipeline."""
        self.vector_store = vector_store or VectorStore()
        self.llm_service = llm_service or LLMService()
        self.n_results = n_results
    
    def query(self, question: str, collection_name: Optional[str] = None) -> Dict[str, any]:
        """Answer a question using RAG."""
        # Retrieve relevant documents from tenant collection
        retrieved = self.vector_store.search(question, n_results=self.n_results, collection_name=collection_name)
        
        # Handle empty retrieval
        if not retrieved:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "context_used": False,
            }
        
        # Build context from retrieved documents
        context = self._build_context(retrieved)
        
        # Generate answer
        prompt = self._build_prompt(question, context)
        answer = self.llm_service.generate(
            prompt=prompt,
            system_prompt="You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain the answer, say so.",
            temperature=0.7,
        )
        
        # Format sources
        sources = self._format_sources(retrieved)
        
        return {
            "answer": answer.strip(),
            "sources": sources,
            "context_used": True,
        }
    
    def _build_context(self, retrieved: List[Dict]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(retrieved, 1):
            source_info = f"[Source {i}: {doc['metadata'].get('source', 'unknown')}"
            if 'page' in doc['metadata']:
                source_info += f", page {doc['metadata']['page']}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM."""
        return f"""Context information:

{context}

Question: {question}

Answer the question based on the context above. Be concise and accurate."""
    
    def _format_sources(self, retrieved: List[Dict]) -> List[Dict[str, str]]:
        """Format source information for the response."""
        sources = []
        seen = set()
        
        for doc in retrieved:
            metadata = doc['metadata']
            source = metadata.get('source', 'unknown')
            
            # Build unique source identifier
            source_id = source
            if 'page' in metadata:
                source_id += f"_page_{metadata['page']}"
            
            # Avoid duplicates
            if source_id in seen:
                continue
            seen.add(source_id)
            
            source_info = {
                "source": source,
                "type": metadata.get('type', 'unknown'),
            }
            if 'page' in metadata:
                source_info['page'] = metadata['page']
            if 'chunk_id' in metadata:
                source_info['chunk_id'] = str(metadata['chunk_id'])
            
            sources.append(source_info)
        
        return sources
