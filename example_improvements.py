# Example: Using Phase 10 improvements

from src.retrieval.chunker import chunk_text, semantic_chunk_text
from src.retrieval.retriever import Retriever
from src.agent.worker import Worker
from src.llm.local_model_client import LocalModelClient

print("Phase 10 Improvements Example")
print("=" * 60)

# Example 1: Semantic Chunking
print("\n1. Semantic Chunking")
text = "Machine learning is AI. Deep learning uses neural networks. Natural language processing handles text. Computer vision processes images."
fixed = chunk_text(text, size=50)
semantic = semantic_chunk_text(text, max_chunk_size=100)
print(f"   Fixed-size: {len(fixed)} chunks")
print(f"   Semantic: {len(semantic)} chunks")

# Example 2: Chunk Filtering
print("\n2. Chunk Filtering")
retriever = Retriever()
retriever.index_chunks([
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "The weather is sunny today",
    "Natural language processing handles text"
])
filtered = retriever.retrieve("machine learning", k=4, min_similarity=0.3)
print(f"   Filtered results: {len(filtered)} chunks")

# Example 3: Enhanced Prompting
print("\n3. Enhanced Prompting")
worker = Worker(retriever, LocalModelClient())
context = ["Machine learning is AI", "Deep learning uses neural networks"]
enhanced = worker._build_prompt("What is ML?", context, enhanced=True)
print(f"   Enhanced prompt: {len(enhanced)} characters")
print(f"   Contains instructions: {'Instructions:' in enhanced}")

# Example 4: Context Merging
print("\n4. Context Merging")
contexts = [
    ["Context 1: Machine learning"],
    ["Context 2: Deep learning"],
    ["Context 1: Machine learning"]
]
merged = worker._merge_contexts(contexts, strategy="deduplicate")
print(f"   Merged contexts: {len(merged)} chunks")

print("\n✓ All improvements demonstrated!")

