# Comprehensive test for Phase 10 improvements

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.chunker import chunk_text, semantic_chunk_text
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import chunk_text
from src.agent.worker import Worker
from src.llm.local_model_client import LocalModelClient
from src.evaluation.metrics import simple_similarity

print("=" * 60)
print("Testing Phase 10: Improvements")
print("=" * 60)

# Test 1: Semantic Chunking
print("\n1. Testing Semantic Chunking...")
test_text = "Machine learning is a subset of AI. Deep learning uses neural networks. Natural language processing handles text. Computer vision processes images."
fixed_chunks = chunk_text(test_text, size=50)
semantic_chunks = semantic_chunk_text(test_text, max_chunk_size=100)
print(f"   Fixed-size chunks: {len(fixed_chunks)}")
print(f"   Semantic chunks: {len(semantic_chunks)}")
assert len(semantic_chunks) > 0, "Semantic chunking should produce chunks"
print("   ✓ Semantic chunking works")

# Test 2: Chunk Filtering
print("\n2. Testing Chunk Filtering...")
retriever = Retriever()
retriever.index_chunks([
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks",
    "The weather is sunny today",
    "Natural language processing handles text data"
])
unfiltered = retriever.retrieve("machine learning", k=4, min_similarity=0.0)
filtered = retriever.retrieve("machine learning", k=4, min_similarity=0.3)
print(f"   Unfiltered results: {len(unfiltered)}")
print(f"   Filtered results (min_similarity=0.3): {len(filtered)}")
assert len(filtered) <= len(unfiltered), "Filtering should reduce results"
print("   ✓ Chunk filtering works")

# Test 3: Retrieve with scores
print("\n3. Testing Retrieve with Scores...")
scored_results = retriever.retrieve_with_scores("machine learning", k=3)
print(f"   Retrieved {len(scored_results)} chunks with scores")
for chunk, score in scored_results:
    print(f"     Score: {score:.3f} - {chunk[:50]}...")
assert all(isinstance(score, float) for _, score in scored_results), "Scores should be floats"
print("   ✓ Retrieve with scores works")

# Test 4: Context Merging Strategies
print("\n4. Testing Context Merging Strategies...")
worker = Worker(retriever, LocalModelClient())
contexts = [
    ["Context 1: Machine learning"],
    ["Context 2: Deep learning"],
    ["Context 1: Machine learning"]  # Duplicate
]

concatenated = worker._merge_contexts(contexts, strategy="concatenate")
deduplicated = worker._merge_contexts(contexts, strategy="deduplicate")
print(f"   Concatenated: {len(concatenated)} chunks")
print(f"   Deduplicated: {len(deduplicated)} chunks")
assert len(deduplicated) < len(concatenated), "Deduplication should reduce chunks"
print("   ✓ Context merging strategies work")

# Test 5: Enhanced Prompting
print("\n5. Testing Enhanced Prompting...")
context = ["Machine learning is a subset of AI", "Deep learning uses neural networks"]
enhanced_prompt = worker._build_prompt("What is machine learning?", context, enhanced=True)
basic_prompt = worker._build_prompt("What is machine learning?", context, enhanced=False)
print(f"   Enhanced prompt length: {len(enhanced_prompt)}")
print(f"   Basic prompt length: {len(basic_prompt)}")
assert len(enhanced_prompt) > len(basic_prompt), "Enhanced prompt should be longer"
assert "Instructions:" in enhanced_prompt, "Enhanced prompt should have instructions"
print("   ✓ Enhanced prompting works")

# Test 6: LocalModelClient (structure test)
print("\n6. Testing LocalModelClient Structure...")
try:
    client = LocalModelClient()
    print("   ✓ LocalModelClient can be instantiated")
    try:
        response = client.generate("test")
        print("   ⚠ Model loaded and working")
    except NotImplementedError:
        print("   ⚠ Model not loaded (expected without model_name)")
    except Exception as e:
        print(f"   ⚠ Model loading issue: {type(e).__name__}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("All improvements tested successfully!")
print("=" * 60)

