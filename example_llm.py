# Example script: Test LLM interface with different models

import os
import sys
from src.llm.openai_client import OpenAIClient
from src.llm.local_model_client import LocalModelClient


def test_openai():
    """Test OpenAI client."""
    print("Testing OpenAI client...")
    try:
        client = OpenAIClient(model_name="gpt-3.5-turbo")
        response = client.generate("What is the capital of France?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def test_local():
    """Test local model client (placeholder)."""
    print("Testing Local model client...")
    try:
        client = LocalModelClient()
        response = client.generate("What is the capital of France?")
        print(f"Response: {response}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_llm.py [openai|local]")
        print("\nNote: For OpenAI, set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    model_type = sys.argv[1].lower()
    
    if model_type == "openai":
        test_openai()
    elif model_type == "local":
        test_local()
    else:
        print(f"Unknown model type: {model_type}")
        print("Use 'openai' or 'local'")


if __name__ == "__main__":
    main()

