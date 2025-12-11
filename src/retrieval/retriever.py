# Context retrieval for queries

from typing import List
from src.retrieval.embedder import Embedder
from src.retrieval.vectorstore import VectorStore


class Retriever:
    """Retriever that combines embedding and vector store."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize retriever with embedding model."""
        self.embedder = Embedder(model_name)
        self.vectorstore = None
    
    def index_chunks(self, chunks: List[str]):
        """Index a list of text chunks."""
        embeddings = self.embedder.embed(chunks)
        self.vectorstore = VectorStore(self.embedder.dimension)
        self.vectorstore.add(embeddings, chunks)
    
    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for k most relevant chunks for a query."""
        if self.vectorstore is None:
            raise ValueError("No chunks indexed. Call index_chunks() first.")
        
        query_embedding = self.embedder.embed_query(query)
        results = self.vectorstore.search(query_embedding, k)
        return results
