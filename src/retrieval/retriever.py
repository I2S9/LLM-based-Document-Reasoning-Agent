# Context retrieval for queries

from typing import List, Tuple
import numpy as np
from src.retrieval.embedder import Embedder
from src.retrieval.vectorstore import VectorStore


class Retriever:
    """Retriever that combines embedding and vector store."""
    
    def __init__(self, embedder: Embedder = None, store: VectorStore = None, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize retriever with embedder and store."""
        if embedder is None:
            embedder = Embedder(model_name)
        if store is None:
            store = VectorStore(embedder.dimension)
        
        self.embedder = embedder
        self.store = store
    
    def retrieve(self, query: str, k: int = 5, min_similarity: float = 0.0) -> List[str]:
        """Retrieve k most relevant chunks for a query with optional filtering."""
        q_emb = self.embedder.encode([query])
        chunks = self.store.search(q_emb, k)
        
        if min_similarity > 0.0:
            chunks = self._filter_chunks(query, chunks, min_similarity)
        
        return chunks
    
    def _filter_chunks(self, query: str, chunks: List[str], min_similarity: float) -> List[str]:
        """Filter chunks based on similarity threshold."""
        if len(chunks) == 0:
            return chunks
        
        query_emb = self.embedder.embed_query(query)
        chunk_embs = self.embedder.embed(chunks)
        
        filtered = []
        for i, chunk in enumerate(chunks):
            similarity = np.dot(query_emb[0], chunk_embs[i]) / (
                np.linalg.norm(query_emb[0]) * np.linalg.norm(chunk_embs[i])
            )
            if similarity >= min_similarity:
                filtered.append(chunk)
        
        return filtered
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve chunks with similarity scores."""
        q_emb = self.embedder.encode([query])
        chunks = self.store.search(q_emb, k)
        
        chunk_embs = self.embedder.embed(chunks)
        results = []
        for i, chunk in enumerate(chunks):
            similarity = np.dot(q_emb[0], chunk_embs[i]) / (
                np.linalg.norm(q_emb[0]) * np.linalg.norm(chunk_embs[i])
            )
            results.append((chunk, float(similarity)))
        
        return results
    
    def index_chunks(self, chunks: List[str]):
        """Index a list of text chunks."""
        embeddings = self.embedder.embed(chunks)
        self.store.add(embeddings, chunks)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Alias for retrieve() for backward compatibility."""
        return self.retrieve(query, k)
