# Main agent orchestrator

from typing import Callable, Any
from src.agent.worker import Worker
from src.llm.model_interface import ModelInterface


class Agent:
    """Agent that combines Planner + Worker + LLM for end-to-end query execution."""
    
    def __init__(self, planner: Callable, worker: Worker, model: ModelInterface):
        """Initialize agent with planner, worker, and model."""
        self.planner = planner
        self.worker = worker
        self.model = model
    
    def run(self, query: str) -> str:
        """Execute a query end-to-end and return final answer."""
        steps = self.planner(query)
        context = ""
        
        for step in steps:
            out = self.worker.run(step)
            if isinstance(out, list):
                context = "\n".join(out)
        
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        return self.model.generate(prompt)
