# Document preprocessing and chunking

from typing import List, Optional
from src.utils.io import clean_text


def chunk_text(text: str, size: int = 500) -> List[str]:
    """Split text into fixed-size chunks."""
    text = clean_text(text)
    chunks = []
    for i in range(0, len(text), size):
        chunk = text[i:i+size]
        chunks.append(chunk)
    return chunks


def semantic_chunk_text(
    text: str, 
    max_chunk_size: int = 500,
    similarity_threshold: float = 0.7
) -> List[str]:
    """Split text into semantic chunks based on sentence similarity."""
    try:
        from src.retrieval.embedder import Embedder
        import numpy as np
    except ImportError:
        raise ImportError("Semantic chunking requires embedder. Falling back to fixed-size chunking.")
    
    text = clean_text(text)
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    
    if len(sentences) == 0:
        return []
    
    if len(sentences) == 1:
        return [sentences[0]]
    
    embedder = Embedder()
    embeddings = embedder.embed(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_size = len(sentences[0])
    
    for i in range(1, len(sentences)):
        sentence = sentences[i]
        sentence_size = len(sentence)
        
        if current_size + sentence_size > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
            continue
        
        similarity = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        
        if similarity < similarity_threshold and len(current_chunk) > 0:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text]


def chunk_pdf(file_path: str, chunk_size: int = 500, semantic: bool = False) -> List[str]:
    """Load a PDF and return list of text chunks."""
    from src.utils.io import load_pdf
    
    text = load_pdf(file_path)
    if semantic:
        return semantic_chunk_text(text, max_chunk_size=chunk_size)
    return chunk_text(text, size=chunk_size)
