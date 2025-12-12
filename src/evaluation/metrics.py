# Accuracy and factual consistency measures

import time
from typing import List, Callable, Any, Tuple


def simple_similarity(a: str, b: str) -> float:
    """Calculate simple word overlap similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    
    if len(words_b) == 0:
        return 0.0
    
    intersection = len(words_a & words_b)
    return intersection / len(words_b)


def chunk_relevance(chunks: List[str], query: str) -> float:
    """Calculate average relevance of chunks to query."""
    if len(chunks) == 0:
        return 0.0
    
    total_similarity = 0.0
    for chunk in chunks:
        similarity = simple_similarity(chunk, query)
        total_similarity += similarity
    
    return total_similarity / len(chunks)


def measure_latency(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    latency = end_time - start_time
    return result, latency
