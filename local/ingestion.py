"""
=============================================================================
Module: Data Ingestion & Sanitization (local/ingestion.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Provides multi-format document loading (PDF, DOCX, TXT, MD, CSV, Web),
    Unicode cleaning (zero-width character removal, whitespace normalization),
    and MD5 content-based document deduplication.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

import re
import hashlib
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader,
    UnstructuredMarkdownLoader, CSVLoader
)

LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
}

def load_single_file(path: str) -> list[Document]:
    """Load one file, tagging it with rich metadata. Returns [] on failure."""
    ext = Path(path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        print(f"[WARNING] Skipping unsupported file type: {path}")
        return []

    try:
        loader = loader_cls(path)
        docs = loader.load()
        for d in docs:
            d.metadata.update({
                "source_file": Path(path).name,
                "file_path": str(Path(path).resolve()),
                "file_type": ext,
                "ingested_at": datetime.utcnow().isoformat(),
            })
        return docs
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return []

def load_from_directory(dir_path: str, extensions: list[str] = None) -> list[Document]:
    """Recursively load all supported files in a folder and subfolders."""
    extensions = extensions or list(LOADER_MAP.keys())
    all_docs = []
    base = Path(dir_path)
    if not base.exists():
        print(f"[WARNING] Directory does not exist: {dir_path}")
        return []

    files = [f for f in base.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
    print(f"Found {len(files)} files to load in '{dir_path}'")

    for f in files:
        docs = load_single_file(str(f))
        all_docs.extend(docs)
        print(f"  - {f.name}: {len(docs)} document chunk(s)/page(s)")

    print(f"Loaded {len(all_docs)} total document pages/records")
    return all_docs

def load_from_urls(urls: list[str]) -> list[Document]:
    """Load web pages using WebBaseLoader."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for d in docs:
        d.metadata["file_type"] = "web"
        d.metadata["ingested_at"] = datetime.utcnow().isoformat()
    print(f"Loaded {len(docs)} web page(s)")
    return docs

def clean_document_text(doc: Document) -> Document:
    """Strips zero-width characters, unprintable unicode, and collapses whitespace."""
    text = doc.page_content
    # Remove zero-width characters
    text = text.replace('\u200b', '').replace('\ufeff', '')
    # Collapse consecutive whitespace
    text = re.sub(r'\s+', ' ', text)
    doc.page_content = text.strip()
    return doc

def process_and_clean_docs(raw_docs: list[Document]) -> list[Document]:
    """Cleans text and skips empty or scanned image documents."""
    cleaned_docs = []
    for doc in raw_docs:
        cleaned = clean_document_text(doc)
        if len(cleaned.page_content) < 10:
            filename = cleaned.metadata.get('source_file', 'Unknown File')
            print(f"Warning: '{filename}' has almost no text (<10 chars). Skipping.")
            continue
        cleaned_docs.append(cleaned)
    return cleaned_docs

def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Removes exact duplicate documents using MD5 hash of text content."""
    seen = set()
    unique = []
    for d in docs:
        h = hashlib.md5(d.page_content.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(d)
    print(f"Deduplication: {len(docs)} -> {len(unique)} documents")
    return unique
