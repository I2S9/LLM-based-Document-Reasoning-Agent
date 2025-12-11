# Document preprocessing and chunking

from typing import List
from src.utils.io import clean_text


def chunk_text(text: str, size: int = 500) -> List[str]:
    """Split text into fixed-size chunks."""
    text = clean_text(text)
    chunks = []
    for i in range(0, len(text), size):
        chunk = text[i:i+size]
        chunks.append(chunk)
    return chunks


def chunk_pdf(file_path: str, chunk_size: int = 500) -> List[str]:
    """Load a PDF and return list of text chunks."""
    from src.utils.io import load_pdf
    
    text = load_pdf(file_path)
    return chunk_text(text, chunk_size)
