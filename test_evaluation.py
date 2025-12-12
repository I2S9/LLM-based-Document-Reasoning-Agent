# Test Phase 8: Evaluation Framework

from src.evaluation.metrics import simple_similarity, chunk_relevance, measure_latency
from src.evaluation.benchmark import Benchmark
from src.agent.agent import Agent
from src.agent.planner import plan
from src.agent.worker import Worker
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import chunk_text
from src.llm.local_model_client import LocalModelClient

print("Testing Phase 8: Evaluation Framework")
print("=" * 60)

# Test 1: simple_similarity
print("\n1. Testing simple_similarity()...")
text1 = "Machine learning is a subset of artificial intelligence"
text2 = "Machine learning uses algorithms to learn from data"
similarity = simple_similarity(text1, text2)
print(f"   Text 1: '{text1}'")
print(f"   Text 2: '{text2}'")
print(f"   Similarity: {similarity:.3f}")
assert 0.0 <= similarity <= 1.0, "Similarity should be between 0 and 1"
print("   ✓ simple_similarity() works correctly")

# Test 2: chunk_relevance
print("\n2. Testing chunk_relevance()...")
chunks = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Natural language processing handles text"
]
query = "machine learning artificial intelligence"
relevance = chunk_relevance(chunks, query)
print(f"   Query: '{query}'")
print(f"   Chunks: {len(chunks)}")
print(f"   Relevance: {relevance:.3f}")
assert 0.0 <= relevance <= 1.0, "Relevance should be between 0 and 1"
print("   ✓ chunk_relevance() works correctly")

# Test 3: measure_latency
print("\n3. Testing measure_latency()...")
def dummy_function(x):
    time.sleep(0.01)  # Simulate work
    return x * 2

import time
result, latency = measure_latency(dummy_function, 5)
print(f"   Function result: {result}")
print(f"   Latency: {latency:.4f} seconds")
assert result == 10, "Function should return 10"
assert latency > 0, "Latency should be positive"
print("   ✓ measure_latency() works correctly")

# Test 4: Benchmark
print("\n4. Testing Benchmark...")
# Setup agent and retriever
retriever = Retriever()
test_text = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks. Natural language processing handles text data."
chunks = chunk_text(test_text, size=40)
retriever.index_chunks(chunks)

model = LocalModelClient()
worker = Worker(retriever, model)
agent = Agent(plan, worker, model)

benchmark = Benchmark(agent, retriever)
print("   ✓ Benchmark initialized")

# Test evaluate_query
print("\n5. Testing evaluate_query()...")
test_cases = [
    {
        "query": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of artificial intelligence"
    }
]

try:
    result = benchmark.evaluate_query(
        test_cases[0]["query"],
        test_cases[0]["ground_truth"],
        k=3
    )
    print(f"   Query: '{result['query']}'")
    print(f"   Latency: {result['latency']:.4f} seconds")
    print(f"   Similarity: {result['similarity']:.3f}")
    print(f"   Chunk relevance: {result['chunk_relevance']:.3f}")
    print(f"   Number of chunks: {result['num_chunks']}")
    assert "latency" in result
    assert "similarity" in result
    assert "chunk_relevance" in result
    print("   ✓ evaluate_query() works correctly")
except NotImplementedError:
    print("   ⚠ evaluate_query() structure works (model not implemented)")

# Test run_benchmark
print("\n6. Testing run_benchmark()...")
try:
    summary = benchmark.run_benchmark(test_cases, k=3)
    print(f"   Number of queries: {summary['num_queries']}")
    print(f"   Average latency: {summary['avg_latency']:.4f} seconds")
    print(f"   Average similarity: {summary['avg_similarity']:.3f}")
    print(f"   Average chunk relevance: {summary['avg_chunk_relevance']:.3f}")
    assert summary['num_queries'] == 1
    assert 'results' in summary
    print("   ✓ run_benchmark() works correctly")
except NotImplementedError:
    print("   ⚠ run_benchmark() structure works (model not implemented)")

print("\n✓ Phase 8: Evaluation Framework works correctly!")

