# Example: Using the Worker

from src.agent.worker import Worker
from src.agent.planner import plan
from src.retrieval.retriever import Retriever
from src.retrieval.chunker import chunk_text
from src.llm.local_model_client import LocalModelClient

# Setup: Create retriever with indexed chunks
print("Setting up retriever...")
retriever = Retriever()
test_text = "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks. Natural language processing handles text data."
chunks = chunk_text(test_text, size=40)
retriever.index_chunks(chunks)
print(f"Indexed {len(chunks)} chunks\n")

# Create model (placeholder)
model = LocalModelClient()

# Create worker
worker = Worker(retriever, model)

# Create plan
query = "What is machine learning?"
steps = plan(query)
print(f"Query: '{query}'")
print(f"Plan: {len(steps)} steps\n")

# Execute steps
for i, step in enumerate(steps, 1):
    print(f"Step {i}: {step['action']}")
    try:
        result = worker.run(step)
        if step['action'] == 'retrieve':
            print(f"  Retrieved {len(result)} chunks")
            print(f"  First chunk: {result[0][:50]}...")
        elif step['action'] == 'answer':
            print(f"  Answer: {result}")
    except NotImplementedError:
        print("  (Model not implemented - placeholder)")

