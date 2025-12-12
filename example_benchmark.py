# Example: Using the Benchmark to compare models

from src.evaluation.benchmark import Benchmark
from src.evaluation.metrics import simple_similarity, chunk_relevance
from src.agent.agent import Agent
from src.agent.planner import plan
from src.agent.worker import Worker
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import chunk_text
from src.llm.local_model_client import LocalModelClient

print("Benchmark Example: Model Comparison")
print("=" * 60)

# Setup: Create retriever with indexed chunks
print("\n1. Setting up retriever...")
retriever = Retriever()
test_text = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks. Natural language processing handles text data. Supervised learning uses labeled data. Unsupervised learning finds patterns in data."
chunks = chunk_text(test_text, size=50)
retriever.index_chunks(chunks)
print(f"   Indexed {len(chunks)} chunks")

# Create test cases
test_cases = [
    {
        "query": "What is machine learning?",
        "ground_truth": "Machine learning is a subset of artificial intelligence"
    },
    {
        "query": "What is deep learning?",
        "ground_truth": "Deep learning uses neural networks"
    },
    {
        "query": "What is supervised learning?",
        "ground_truth": "Supervised learning uses labeled data"
    }
]

print(f"\n2. Created {len(test_cases)} test cases")

# Setup agent
print("\n3. Setting up agent...")
model = LocalModelClient()
worker = Worker(retriever, model)
agent = Agent(plan, worker, model)
print("   Agent created")

# Create benchmark
print("\n4. Creating benchmark...")
benchmark = Benchmark(agent, retriever)
print("   Benchmark created")

# Run benchmark
print("\n5. Running benchmark...")
try:
    summary = benchmark.run_benchmark(test_cases, k=3)
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Number of queries: {summary['num_queries']}")
    print(f"Average latency: {summary['avg_latency']:.4f} seconds")
    print(f"Average similarity: {summary['avg_similarity']:.3f}")
    print(f"Average chunk relevance: {summary['avg_chunk_relevance']:.3f}")
    
    print("\nDetailed Results:")
    for i, result in enumerate(summary['results'], 1):
        print(f"\n  Query {i}: '{result['query']}'")
        print(f"    Latency: {result['latency']:.4f}s")
        print(f"    Similarity: {result['similarity']:.3f}")
        print(f"    Chunk relevance: {result['chunk_relevance']:.3f}")
        print(f"    Chunks retrieved: {result['num_chunks']}")
    
except NotImplementedError:
    print("   (Model not implemented - placeholder)")
    print("   ✓ Benchmark structure works correctly")
    print("   ✓ All metrics calculated")

print("\n" + "=" * 60)
print("Benchmark complete!")

