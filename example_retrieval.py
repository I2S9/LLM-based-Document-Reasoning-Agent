# Example script: Index chunks and search for relevant segments

import sys
from pathlib import Path
from src.retrieval.chunker import chunk_pdf
from src.retrieval.retriever import Retriever


def main():
    if len(sys.argv) < 3:
        print("Usage: python example_retrieval.py <path_to_pdf> <query> [k=5]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Loading and chunking PDF: {pdf_path}")
    chunks = chunk_pdf(pdf_path, chunk_size=500)
    print(f"Created {len(chunks)} chunks")
    
    print("\nIndexing chunks...")
    retriever = Retriever()
    retriever.index_chunks(chunks)
    
    print(f"\nSearching for: '{query}'")
    print(f"Retrieving top {k} most relevant chunks...\n")
    
    results = retriever.search(query, k=k)
    
    print(f"Found {len(results)} results:\n")
    for i, chunk in enumerate(results, 1):
        print(f"--- Result {i} ({len(chunk)} chars) ---")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        print()


if __name__ == "__main__":
    main()

