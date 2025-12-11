# Example: Phase 4 Retriever usage

from src.retrieval.retriever import Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.vectorstore import VectorStore
from src.retrieval.chunker import chunk_text

# Create embedder and store
embedder = Embedder()
store = VectorStore(embedder.dimension)

# Create retriever with embedder and store (Phase 4 API)
retriever = Retriever(embedder, store)

# Prepare chunks
text = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks. Natural language processing handles text data."
chunks = chunk_text(text, size=40)

# Index chunks
retriever.index_chunks(chunks)

# Retrieve relevant chunks
query = "neural networks"
results = retriever.retrieve(query, k=3)

print(f"Query: '{query}'")
print(f"\nRetrieved {len(results)} relevant chunks:")
for i, chunk in enumerate(results, 1):
    print(f"\n{i}. {chunk}")

