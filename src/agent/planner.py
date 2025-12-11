# Planner component: determines required reasoning steps

from typing import List, Dict


def plan(query: str) -> List[Dict[str, str]]:
    """Transform a question into reasoning steps.
    
    Identifies document needs, defines Worker actions, and assembles results.
    """
    steps = []
    
    # Step 1: Retrieve relevant document segments
    steps.append({"action": "retrieve", "query": query})
    
    # Step 2: Generate answer from retrieved context
    steps.append({"action": "answer"})
    
    return steps
