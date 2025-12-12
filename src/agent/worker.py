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
    
    def _merge_contexts(self, contexts: List[List[str]], strategy: str = "concatenate") -> List[str]:
        """Merge multiple context lists using different strategies."""
        if strategy == "concatenate":
            merged = []
            for ctx in contexts:
                merged.extend(ctx)
            return merged
        
        elif strategy == "deduplicate":
            seen = set()
            merged = []
            for ctx in contexts:
                for chunk in ctx:
                    chunk_hash = hash(chunk)
                    if chunk_hash not in seen:
                        seen.add(chunk_hash)
                        merged.append(chunk)
            return merged
        
        elif strategy == "weighted":
            from src.retrieval.embedder import Embedder
            import numpy as np
            
            if len(contexts) == 0:
                return []
            
            embedder = Embedder()
            all_chunks = []
            all_embeddings = []
            
            for ctx in contexts:
                all_chunks.extend(ctx)
                if len(ctx) > 0:
                    embs = embedder.embed(ctx)
                    all_embeddings.extend(embs)
            
            if len(all_chunks) == 0:
                return []
            
            if len(all_chunks) == 1:
                return all_chunks
            
            embeddings_array = np.array(all_embeddings)
            centroid = np.mean(embeddings_array, axis=0)
            
            distances = []
            for emb in embeddings_array:
                dist = np.linalg.norm(emb - centroid)
                distances.append(dist)
            
            sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
            return [all_chunks[i] for i in sorted_indices[:len(all_chunks)]]
        
        else:
            return contexts[0] if contexts else []
    
    def _build_prompt(self, query: str, context: List[str], enhanced: bool = True) -> str:
        """Build a contextual prompt from query and retrieved chunks."""
        if enhanced:
            context_text = "\n\n".join([
                f"--- Context {i+1} ---\n{chunk}" 
                for i, chunk in enumerate(context)
            ])
            
            prompt = f"""You are a helpful assistant that answers questions based on provided context.

Context Information:
{context_text}

Instructions:
- Answer the question using only information from the context above
- If the context doesn't contain enough information, say so
- Be precise and concise
- Cite relevant context numbers when possible

Question: {query}

Answer:"""
        else:
            context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
            prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
