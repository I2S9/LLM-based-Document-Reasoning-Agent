# Example script: Load a PDF and chunk it into segments

import sys
from pathlib import Path
from src.retrieval.chunker import chunk_pdf


def main():
    if len(sys.argv) < 2:
        print("Usage: python example_chunk_pdf.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Loading PDF: {pdf_path}")
    chunks = chunk_pdf(pdf_path, chunk_size=500)
    
    print(f"\nNumber of chunks: {len(chunks)}")
    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)


if __name__ == "__main__":
    main()

