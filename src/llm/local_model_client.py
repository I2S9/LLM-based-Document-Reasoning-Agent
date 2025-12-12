# Local model client (Llama, Mistral, etc.)

from typing import Optional
from src.llm.model_interface import ModelInterface


class LocalModelClient(ModelInterface):
    """Local model client for running models like Llama, Mistral, etc. locally."""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        device: str = "cpu",
        max_length: int = 1000,
        temperature: float = 0.7
    ):
        """Initialize local model client.
        
        Args:
            model_path: Path to local model directory
            model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-v0.1")
            device: Device to run model on ("cpu" or "cuda")
            max_length: Maximum generation length
            temperature: Sampling temperature
        """
        self.model_path = model_path
        self.model_name = model_name or model_path
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        if self.model_name is None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for local models. "
                "Install with: pip install transformers torch"
            )
        
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")
            print("Falling back to placeholder mode")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str) -> str:
        """Generate a response using local model."""
        if self.model is None or self.tokenizer is None:
            raise NotImplementedError(
                "Local model not loaded. "
                "Provide model_name or model_path to load a model, or install transformers/torch."
            )
        
        try:
            import torch
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")
