# Context retrieval for queries

from typing import List
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
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve k most relevant chunks for a query."""
        q_emb = self.embedder.encode([query])
        return self.store.search(q_emb, k)
    
    def index_chunks(self, chunks: List[str]):
        """Index a list of text chunks."""
        embeddings = self.embedder.embed(chunks)
        self.store.add(embeddings, chunks)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Alias for retrieve() for backward compatibility."""
        return self.retrieve(query, k)
