#!/usr/bin/env python3
"""
Main entry point for the LLM-based Document Reasoning Agent.

This script provides a command-line interface to interact with the document reasoning agent.
"""

import sys
import argparse
from pathlib import Path

from src.agent.agent import Agent
from src.agent.planner import plan
from src.agent.worker import Worker
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import chunk_pdf, chunk_text
from src.llm.openai_client import OpenAIClient
from src.llm.local_model_client import LocalModelClient


def setup_agent(pdf_path: str = None, text: str = None, model_type: str = "openai", semantic: bool = False):
    """Setup and return a configured agent."""
    # Setup retriever
    retriever = Retriever()
    
    if pdf_path:
        print(f"Loading PDF: {pdf_path}")
        chunks = chunk_pdf(pdf_path, chunk_size=500, semantic=semantic)
    elif text:
        print("Processing text...")
        if semantic:
            from src.retrieval.chunker import semantic_chunk_text
            chunks = semantic_chunk_text(text, max_chunk_size=500)
        else:
            chunks = chunk_text(text, size=500)
    else:
        # Use sample text
        sample_text = """
        Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.
        Deep learning uses neural networks with multiple layers to process complex patterns in data.
        Natural language processing handles text data and enables machines to understand human language.
        Supervised learning uses labeled data to train models that can make predictions.
        """
        chunks = chunk_text(sample_text, size=100)
    
    print(f"Indexed {len(chunks)} chunks")
    retriever.index_chunks(chunks)
    
    # Setup model
    if model_type == "openai":
        try:
            model = OpenAIClient()
        except ValueError:
            print("Warning: OpenAI API key not found. Using local model placeholder.")
            model = LocalModelClient()
    else:
        model = LocalModelClient()
    
    # Setup agent
    worker = Worker(retriever, model)
    agent = Agent(plan, worker, model)
    
    return agent, retriever


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="LLM-based Document Reasoning Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to process"
    )
    
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF document"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        help="Text to process"
    )
    
    parser.add_argument(
        "--model",
        choices=["openai", "local"],
        default="openai",
        help="Model type to use (default: openai)"
    )
    
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic chunking instead of fixed-size"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    # Setup agent
    try:
        agent, retriever = setup_agent(
            pdf_path=args.pdf,
            text=args.text,
            model_type=args.model,
            semantic=args.semantic
        )
    except Exception as e:
        print(f"Error setting up agent: {e}")
        sys.exit(1)
    
    # Interactive mode
    if args.interactive or not args.query:
        print("\n" + "=" * 60)
        print("LLM-based Document Reasoning Agent")
        print("=" * 60)
        print("Enter queries (type 'exit' to quit):\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not query:
                    continue
                
                print("\nProcessing...")
                answer = agent.run(query)
                print(f"\nAnswer: {answer}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    
    # Single query mode
    else:
        try:
            print(f"\nQuery: {args.query}")
            print("Processing...\n")
            answer = agent.run(args.query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

