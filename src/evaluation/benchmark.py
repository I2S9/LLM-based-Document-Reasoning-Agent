# Evaluation benchmark suite

import time
from typing import List, Dict, Any, Callable
from src.evaluation.metrics import simple_similarity, chunk_relevance, measure_latency
from src.agent.agent import Agent
from src.retrieval.retriever import Retriever


class Benchmark:
    """Benchmark suite for comparing models."""
    
    def __init__(self, agent: Agent, retriever: Retriever):
        """Initialize benchmark with agent and retriever."""
        self.agent = agent
        self.retriever = retriever
    
    def evaluate_query(
        self, 
        query: str, 
        ground_truth: str,
        k: int = 5
    ) -> Dict[str, Any]:
        """Evaluate a single query and return metrics."""
        results = {}
        
        # Measure latency
        start_time = time.time()
        answer = self.agent.run(query)
        latency = time.time() - start_time
        results["latency"] = latency
        
        # Calculate answer similarity
        similarity = simple_similarity(answer, ground_truth)
        results["similarity"] = similarity
        
        # Get retrieved chunks and calculate relevance
        chunks = self.retriever.retrieve(query, k=k)
        relevance = chunk_relevance(chunks, query)
        results["chunk_relevance"] = relevance
        results["num_chunks"] = len(chunks)
        
        # Store results
        results["query"] = query
        results["answer"] = answer
        results["ground_truth"] = ground_truth
        
        return results
    
    def run_benchmark(
        self,
        test_cases: List[Dict[str, str]],
        k: int = 5
    ) -> Dict[str, Any]:
        """Run benchmark on multiple test cases."""
        all_results = []
        total_latency = 0.0
        total_similarity = 0.0
        total_relevance = 0.0
        
        for test_case in test_cases:
            query = test_case["query"]
            ground_truth = test_case["ground_truth"]
            
            result = self.evaluate_query(query, ground_truth, k)
            all_results.append(result)
            
            total_latency += result["latency"]
            total_similarity += result["similarity"]
            total_relevance += result["chunk_relevance"]
        
        num_cases = len(test_cases)
        summary = {
            "num_queries": num_cases,
            "avg_latency": total_latency / num_cases if num_cases > 0 else 0.0,
            "avg_similarity": total_similarity / num_cases if num_cases > 0 else 0.0,
            "avg_chunk_relevance": total_relevance / num_cases if num_cases > 0 else 0.0,
            "results": all_results
        }
        
        return summary
