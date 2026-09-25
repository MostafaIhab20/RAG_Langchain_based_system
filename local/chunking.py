"""
=============================================================================
Module: Document Chunking (local/chunking.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Implements syntax-based RecursiveCharacterTextSplitter and embedding-based
    SemanticChunker strategies for segmenting documents prior to vector storage.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker as LC_SemanticChunker

class RecursiveChunker:
    """Fast, syntax-based chunking using character counts and overlaps."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        print(f"\nExecuting Recursive Chunking on {len(documents)} document(s)...")
        chunked_docs = self.splitter.split_documents(documents)
        print(f"Generated {len(chunked_docs)} chunks.")
        return chunked_docs

class SemanticChunker:
    """Advanced embedding-based chunking that breaks text when semantic topic shifts."""

    def __init__(self, embedding_model, percentile: int = 80):
        if not embedding_model:
            raise ValueError("You must provide an 'embedding_model'.")

        self.splitter = LC_SemanticChunker(
            embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=percentile
        )

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        print(f"\nExecuting Semantic Chunking on {len(documents)} document(s)...")
        chunked_docs = self.splitter.split_documents(documents)
        print(f"Generated {len(chunked_docs)} semantic chunks.")
        return chunked_docs
