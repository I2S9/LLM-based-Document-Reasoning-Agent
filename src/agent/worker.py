# Worker component: executes retrieval and calls LLM

from typing import List, Dict, Any, Optional
from src.llm.model_interface import ModelInterface
from src.retrieval.retriever import Retriever


class Worker:
    """Worker that executes Planner actions."""
    
    def __init__(self, retriever: Retriever, model: ModelInterface):
        """Initialize worker with retriever and model."""
        self.retriever = retriever
        self.model = model
        self.context: Optional[List[str]] = None
        self.query: Optional[str] = None
    
    def run(self, step: Dict[str, Any]) -> Any:
        """Execute a single step from the planner."""
        action = step.get("action")
        
        if action == "retrieve":
            query = step.get("query", "")
            self.query = query
            self.context = self.retriever.retrieve(query, k=5)
            return self.context
        
        if action == "answer":
            if self.context is None:
                raise ValueError("No context available. Run 'retrieve' action first.")
            
            prompt = self._build_prompt(self.query, self.context)
            answer = self.model.generate(prompt)
            return answer
        
        raise ValueError(f"Unknown action: {action}")
    
    def _build_prompt(self, query: str, context: List[str]) -> str:
        """Build a contextual prompt from query and retrieved chunks."""
        context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
