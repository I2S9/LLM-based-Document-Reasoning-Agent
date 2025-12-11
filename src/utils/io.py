# File I/O utilities

import re
from pathlib import Path
from typing import Optional


def load_pdf(file_path: str) -> str:
    """Load text content from a PDF file."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required. Install it with: pip install PyPDF2")
    
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    return text


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text
