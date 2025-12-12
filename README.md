# LLM-based Document Reasoning Agent

> A complete retrieval-augmented generation (RAG) system combined with an agent architecture for extracting structured information from long technical documents. This project implements a multi-step reasoning agent capable of processing documents, retrieving relevant segments, and generating accurate answers using various LLM models.

## Project Summary

This project provides a research-grade implementation of a document reasoning agent that combines:

- **Document Processing**: PDF loading, text cleaning, and intelligent chunking (fixed-size and semantic)
- **Retrieval System**: Vector embeddings with FAISS indexing for semantic search
- **Agent Architecture**: Planner-Worker pattern for multi-step reasoning
- **LLM Integration**: Unified interface supporting OpenAI API and local models (Llama, Mistral, etc.)
- **Evaluation Framework**: Comprehensive metrics for accuracy, relevance, and latency

The system is designed to be modular, extensible, and suitable for research experiments comparing different models and configurations.

## Architecture

### System Components

```
┌─────────────────┐
│   Document      │
│   (PDF/Text)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Chunker      │───► Fixed-size or Semantic chunking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedder      │───► Sentence Transformers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VectorStore    │───► FAISS Index
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retriever     │───► Semantic Search + Filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│    Planner      │─────►│    Worker    │
└─────────────────┘      └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │     LLM      │
                          │  (OpenAI/    │
                          │   Local)     │
                          └──────────────┘
```

### Directory Structure

```
project-root/
├── src/
│   ├── agent/           # Planner, Worker, and Agent orchestrator
│   ├── retrieval/       # Chunking, embedding, vector store, retriever
│   ├── llm/            # Model interfaces (OpenAI, Local)
│   ├── evaluation/     # Metrics and benchmark suite
│   └── utils/          # I/O utilities and logging
├── data/
│   ├── samples/        # Sample PDF documents
│   └── processed/      # Processed chunks
├── experiments/        # Benchmark scripts and results
├── notebooks/          # Jupyter notebooks for exploration
└── docs/              # Documentation and papers
```

### Key Features

1. **Semantic Chunking**: Intelligent text segmentation based on sentence similarity
2. **Chunk Filtering**: Relevance-based filtering to remove noise
3. **Context Merging**: Multiple strategies for combining retrieved contexts
4. **Enhanced Prompting**: Structured prompts with clear instructions
5. **Multi-Model Support**: Seamless switching between OpenAI and local models

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/I2S9/LLM-based-Document-Reasoning-Agent.git
cd LLM-based-Document-Reasoning-Agent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Optional - Install Local Model Support

For local model support (Llama, Mistral, etc.):

```bash
pip install transformers torch
```

### Step 4: Set Environment Variables (Optional)

For OpenAI API usage:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

#### 1. Load and Chunk Documents

```python
from src.retrieval.chunker import chunk_pdf, semantic_chunk_text

# Fixed-size chunking
chunks = chunk_pdf("data/samples/document.pdf", chunk_size=500)

# Semantic chunking (recommended)
chunks = chunk_pdf("data/samples/document.pdf", chunk_size=500, semantic=True)
```

#### 2. Index Documents

```python
from src.retrieval.retriever import Retriever

retriever = Retriever()
retriever.index_chunks(chunks)
```

#### 3. Create Agent and Query

```python
from src.agent.agent import Agent
from src.agent.planner import plan
from src.agent.worker import Worker
from src.llm.openai_client import OpenAIClient

# Setup
model = OpenAIClient(model_name="gpt-3.5-turbo")
worker = Worker(retriever, model)
agent = Agent(plan, worker, model)

# Query
answer = agent.run("What is the main topic of the document?")
print(answer)
```

### Advanced Usage

#### Using Local Models

```python
from src.llm.local_model_client import LocalModelClient

# Load a local model
model = LocalModelClient(
    model_name="mistralai/Mistral-7B-v0.1",
    device="cuda",  # or "cpu"
    max_length=1000,
    temperature=0.7
)

worker = Worker(retriever, model)
agent = Agent(plan, worker, model)
```

#### Chunk Filtering

```python
# Retrieve with minimum similarity threshold
results = retriever.retrieve(
    query="machine learning",
    k=10,
    min_similarity=0.3  # Filter out low-relevance chunks
)
```

#### Retrieve with Scores

```python
# Get chunks with similarity scores
scored_results = retriever.retrieve_with_scores("query", k=5)
for chunk, score in scored_results:
    print(f"Score: {score:.3f} - {chunk}")
```

### Running Experiments

```bash
python experiments/run_benchmark.py
```

This will:
1. Load sample documents
2. Run benchmark queries
3. Generate `experiments/results.json` with metrics

## Example Queries

### Example 1: Simple Question Answering

```python
query = "What is machine learning?"
answer = agent.run(query)
```

**Expected Output:**
```
Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed.
```

### Example 2: Technical Definition

```python
query = "Explain the difference between supervised and unsupervised learning"
answer = agent.run(query)
```

### Example 3: Multi-Step Reasoning

```python
query = "What are the main applications of deep learning mentioned in the document?"
answer = agent.run(query)
```

## Evaluation Results

### Metrics

The evaluation framework measures:

- **Similarity Score**: Word overlap between generated answer and ground truth (0-1)
- **Chunk Relevance**: Average similarity of retrieved chunks to query (0-1)
- **Latency**: End-to-end query processing time (seconds)

### Sample Results

From `experiments/results.json`:

```json
{
  "summary": {
    "avg_latency": 0.5,
    "avg_similarity": 0.75,
    "avg_chunk_relevance": 0.6
  }
}
```

### Benchmark Results

| Model | Avg Latency (s) | Avg Similarity | Avg Chunk Relevance |
|-------|----------------|----------------|---------------------|
| GPT-3.5-turbo | 0.5 | 0.75 | 0.60 |
| Local Model | TBD | TBD | TBD |

*Note: Results may vary based on document complexity and query type.*

### Running Custom Evaluations

```python
from src.evaluation.benchmark import Benchmark

test_cases = [
    {"query": "What is X?", "ground_truth": "X is..."},
    # ... more test cases
]

benchmark = Benchmark(agent, retriever)
summary = benchmark.run_benchmark(test_cases)
```

### Research Directions

- **Multi-hop Reasoning**: Extend planner to support multi-step document traversal
- **Active Learning**: Improve retrieval through user feedback
- **Domain Adaptation**: Fine-tuning for specific domains
- **Explainability**: Provide reasoning traces and source attribution