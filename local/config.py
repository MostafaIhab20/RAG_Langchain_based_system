"""
=============================================================================
Module: Configuration & Model Initialization (local/config.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: 2026-09-12
Version: 1.0.0
Description:
    Manages environment variables (.env), LangSmith tracing configuration,
    and Google Gemini embedding models with optional local caching.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.0.0"
__date__ = "2026-09-12"

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
dotenv_path = ROOT_DIR / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

# Silence user-agent warning
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "PQC-RAG-System/1.0"

def setup_environment():
    """Validates and sets up environment variables for Gemini and LangSmith."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Warning: GOOGLE_API_KEY is not set. Please add it to your .env file.")

    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    if langchain_key:
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "gemini-rag-local")
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        print(f"✅ LangSmith tracing enabled: {os.environ['LANGCHAIN_PROJECT']}")

def get_embedding_model(model: str = None):
    """Returns Google Generative AI Embeddings instance."""
    selected_model = model or os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    return GoogleGenerativeAIEmbeddings(model=selected_model)

def get_cached_embedding_model(cache_dir: str = "./embedding_cache", model: str = None):
    """Wraps the embedding model with a local file-based cache to avoid duplicate API calls."""
    try:
        from langchain_classic.embeddings import CacheBackedEmbeddings
        from langchain_classic.storage import LocalFileStore
    except ImportError:
        from langchain.embeddings import CacheBackedEmbeddings
        from langchain.storage import LocalFileStore

    underlying_embeddings = get_embedding_model(model=model)
    store = LocalFileStore(cache_dir)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model,
    )
