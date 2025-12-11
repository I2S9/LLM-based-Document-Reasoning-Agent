# Unified interface for all models

from abc import ABC, abstractmethod


class ModelInterface(ABC):
    """Abstract base class for LLM model interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the model given a prompt."""
        raise NotImplementedError
