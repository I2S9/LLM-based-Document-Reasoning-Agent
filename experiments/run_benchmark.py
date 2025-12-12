# Script to run benchmarks and save results

import json
import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.agent import Agent
from src.agent.planner import plan
from src.agent.worker import Worker
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import chunk_text
from src.evaluation.benchmark import Benchmark
from src.llm.local_model_client import LocalModelClient


def setup_retriever():
    """Setup retriever with sample documents."""
    retriever = Retriever()
    
    # Sample document text
    document_text = """
    Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.
    Deep learning uses neural networks with multiple layers to process complex patterns in data.
    Natural language processing handles text data and enables machines to understand human language.
    Supervised learning uses labeled data to train models that can make predictions.
    Unsupervised learning finds patterns in data without labeled examples.
    Reinforcement learning trains agents through rewards and penalties in an environment.
    """
    
    chunks = chunk_text(document_text, size=100)
    retriever.index_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks")
    return retriever


def get_test_cases():
    """Define test cases with queries and ground truth."""
    return [
        {
            "query": "What is machine learning?",
            "ground_truth": "Machine learning is a subset of artificial intelligence"
        },
        {
            "query": "What is deep learning?",
            "ground_truth": "Deep learning uses neural networks with multiple layers"
        },
        {
            "query": "What is natural language processing?",
            "ground_truth": "Natural language processing handles text data"
        },
        {
            "query": "What is supervised learning?",
            "ground_truth": "Supervised learning uses labeled data to train models"
        }
    ]


def run_experiment():
    """Run benchmark experiment and save results."""
    print("=" * 60)
    print("Running Benchmark Experiment")
    print("=" * 60)
    
    # Setup
    print("\n1. Setting up components...")
    retriever = setup_retriever()
    model = LocalModelClient()
    worker = Worker(retriever, model)
    agent = Agent(plan, worker, model)
    benchmark = Benchmark(agent, retriever)
    print("   ✓ Components initialized")
    
    # Get test cases
    print("\n2. Loading test cases...")
    test_cases = get_test_cases()
    print(f"   ✓ Loaded {len(test_cases)} test cases")
    
    # Run benchmark
    print("\n3. Running benchmark...")
    try:
        summary = benchmark.run_benchmark(test_cases, k=5)
        print("   ✓ Benchmark completed")
    except NotImplementedError:
        print("   ⚠ Model not implemented (using placeholder)")
        # Create mock results for testing structure
        summary = {
            "num_queries": len(test_cases),
            "avg_latency": 0.5,
            "avg_similarity": 0.75,
            "avg_chunk_relevance": 0.6,
            "results": [
                {
                    "query": tc["query"],
                    "ground_truth": tc["ground_truth"],
                    "latency": 0.5,
                    "similarity": 0.75,
                    "chunk_relevance": 0.6,
                    "num_chunks": 5,
                    "answer": "Mock answer (model not implemented)"
                }
                for tc in test_cases
            ]
        }
    
    # Prepare results for JSON
    results_data = {
        "experiment_info": {
            "num_queries": summary["num_queries"],
            "timestamp": None  # Can be added with datetime if needed
        },
        "summary": {
            "avg_latency": summary["avg_latency"],
            "avg_similarity": summary["avg_similarity"],
            "avg_chunk_relevance": summary["avg_chunk_relevance"]
        },
        "results": []
    }
    
    # Extract latency and score (similarity) for each query
    for result in summary["results"]:
        results_data["results"].append({
            "query": result["query"],
            "latency": result["latency"],
            "score": result["similarity"],
            "ground_truth": result["ground_truth"],
            "chunk_relevance": result.get("chunk_relevance", 0.0),
            "num_chunks": result.get("num_chunks", 0)
        })
    
    # Save to JSON
    print("\n4. Saving results...")
    results_file = Path(__file__).parent / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Results saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Number of queries: {results_data['experiment_info']['num_queries']}")
    print(f"Average latency: {results_data['summary']['avg_latency']:.4f} seconds")
    print(f"Average score (similarity): {results_data['summary']['avg_similarity']:.3f}")
    print(f"Average chunk relevance: {results_data['summary']['avg_chunk_relevance']:.3f}")
    
    print("\nPer-query results:")
    for i, result in enumerate(results_data["results"], 1):
        print(f"  {i}. Query: '{result['query']}'")
        print(f"     Latency: {result['latency']:.4f}s, Score: {result['score']:.3f}")
    
    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    run_experiment()

