"""
=============================================================================
Module: Package Initialization (local/__init__.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Exposes the core components of the local RAG pipeline for clean imports
    across the project.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

from .config import setup_environment, get_embedding_model
from .ingestion import (
    load_single_file,
    load_from_directory,
    process_and_clean_docs,
    deduplicate_docs
)
from .chunking import RecursiveChunker, SemanticChunker
from .vector_store import VectorDBManager
from .retriever import HybridRetrieverManager
from .chain import build_rag_chain, ask, enable_llm_cache

__all__ = [
    "setup_environment",
    "get_embedding_model",
    "load_single_file",
    "load_from_directory",
    "process_and_clean_docs",
    "deduplicate_docs",
    "RecursiveChunker",
    "SemanticChunker",
    "VectorDBManager",
    "HybridRetrieverManager",
    "build_rag_chain",
    "ask",
    "enable_llm_cache",
]
