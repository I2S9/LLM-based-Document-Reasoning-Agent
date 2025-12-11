# Vector indexing (FAISS or similar)

import faiss
import numpy as np
from typing import List


class VectorStore:
    """Simple FAISS-based vector store for semantic search."""
    
    def __init__(self, dim: int):
        """Initialize vector store with given dimension."""
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
    
    def add(self, embeddings: np.ndarray, chunks: List[str]):
        """Add embeddings and corresponding chunks to the index."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        embeddings = embeddings.astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)
    
    def search(self, embedding: np.ndarray, k: int = 5) -> List[str]:
        """Search for k most similar chunks."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        embedding = embedding.astype("float32")
        scores, indices = self.index.search(embedding, k)
        
        results = [self.chunks[i] for i in indices[0]]
        return results
