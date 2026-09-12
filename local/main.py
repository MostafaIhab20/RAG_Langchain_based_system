"""
=============================================================================
Module: CLI & Interactive Application Runner (local/main.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: 2026-09-12
Version: 1.0.0
Description:
    Command-line interface and interactive terminal chat loop supporting
    knowledge base indexing, single-query execution, and customizable
    retrieval/chunking hyperparameters.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.0.0"
__date__ = "2026-09-12"

import os
import sys
import pickle
import argparse
from pathlib import Path

# Add project root to sys.path so modules can be run directly: `python local/main.py`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def check_dependencies():
    """Checks if core dependencies are installed and gives helpful instructions if missing."""
    missing = []
    for pkg, import_name in [
        ("langchain", "langchain"),
        ("langchain-google-genai", "langchain_google_genai"),
        ("langchain-chroma", "langchain_chroma"),
        ("chromadb", "chromadb"),
        ("python-dotenv", "dotenv"),
        ("rank_bm25", "rank_bm25"),
        ("pymupdf", "pymupdf"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("\n❌ Missing required dependencies:")
        for m in missing:
            print(f"   - {m}")
        print("\n👉 Please install them by running:")
        print("   pip install -r requirements.txt\n")
        sys.exit(1)

def get_chunks_cache_path(persist_dir: str) -> Path:
    return Path(persist_dir) / "indexed_chunks.pkl"

def build_knowledge_base(
    data_dir: str = "./data",
    chunk_mode: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    semantic_percentile: int = 80,
    collection_name: str = "rag_production",
    persist_directory: str = "./chroma_db",
    top_k: int = 4,
):
    """
    End-to-end ingestion pipeline:
    load -> clean -> deduplicate -> chunk -> embed & store in Chroma -> configure Hybrid Retriever.
    """
    from local.config import get_embedding_model
    from local.ingestion import load_from_directory, process_and_clean_docs, deduplicate_docs
    from local.chunking import RecursiveChunker, SemanticChunker
    from local.vector_store import VectorDBManager
    from local.retriever import HybridRetrieverManager

    print("=" * 60)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 60)
    raw_docs = load_from_directory(data_dir)
    if not raw_docs:
        raise ValueError(
            f"❌ No documents found in '{data_dir}'. Make sure your files (PDF, DOCX, TXT, MD, CSV) are in this directory."
        )

    print("\n" + "=" * 60)
    print("STEP 2: CLEANING")
    print("=" * 60)
    cleaned_docs = process_and_clean_docs(raw_docs)

    print("\n" + "=" * 60)
    print("STEP 3: DEDUPLICATION")
    print("=" * 60)
    unique_docs = deduplicate_docs(cleaned_docs)

    print("\n" + "=" * 60)
    print("STEP 4: CHUNKING")
    print("=" * 60)
    if chunk_mode == "semantic":
        chunker = SemanticChunker(get_embedding_model(), percentile=semantic_percentile)
    else:
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    final_chunks = chunker.split(unique_docs)

    print("\n" + "=" * 60)
    print("STEP 5: VECTOR STORE & BM25 PERSISTENCE")
    print("=" * 60)
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=persist_directory)
    db_manager.upsert_chunks(final_chunks)

    # Save chunks locally so future query sessions can instantiate BM25 without re-parsing files
    cache_path = get_chunks_cache_path(persist_directory)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(final_chunks, f)
    print(f"💾 Cached chunk index for BM25 to '{cache_path}'")

    print("\n" + "=" * 60)
    print("STEP 6: HYBRID RETRIEVER")
    print("=" * 60)
    hybrid_manager = HybridRetrieverManager(
        vector_store=db_manager.vector_store,
        documents=final_chunks,
        top_k=top_k,
    )

    print("\n🎉 Knowledge base successfully built and ready!")
    return hybrid_manager.get_retriever(), db_manager, final_chunks

def load_existing_retriever(
    persist_directory: str = "./chroma_db",
    collection_name: str = "rag_production",
    top_k: int = 4
):
    """Loads existing Chroma vector database and cached BM25 chunks."""
    from local.vector_store import VectorDBManager
    from local.retriever import HybridRetrieverManager

    cache_path = get_chunks_cache_path(persist_directory)
    db_path = Path(persist_directory)

    if not db_path.exists() or not cache_path.exists():
        return None, None

    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=persist_directory)
    with open(cache_path, "rb") as f:
        chunks = pickle.load(f)

    hybrid_manager = HybridRetrieverManager(
        vector_store=db_manager.vector_store,
        documents=chunks,
        top_k=top_k
    )
    return hybrid_manager.get_retriever(), db_manager

def main():
    parser = argparse.ArgumentParser(description="Local LangChain RAG System with Google Gemini")
    parser.add_argument("--data-dir", default="./data", help="Directory containing source documents (default: ./data)")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Directory to store Chroma DB (default: ./chroma_db)")
    parser.add_argument("--chunk-mode", choices=["recursive", "semantic"], default="recursive", help="Chunking strategy (recursive/semantic)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for recursive splitter (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for recursive splitter (default: 150)")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    parser.add_argument("--build", action="store_true", help="Force building or rebuilding the knowledge base from data directory")
    parser.add_argument("--query", type=str, default=None, help="Ask a single question and exit")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM SQLite query cache")

    args = parser.parse_args()

    # Verify dependencies when executing commands
    check_dependencies()

    from local.config import setup_environment
    from local.chain import build_rag_chain, ask, enable_llm_cache

    setup_environment()

    if not args.no_cache:
        enable_llm_cache(persist=True)

    retriever = None
    if not args.build:
        retriever, _ = load_existing_retriever(persist_directory=args.persist_dir, top_k=args.top_k)

    if retriever is None:
        print("⚡ No existing index found or --build requested. Building knowledge base...")
        retriever, _, _ = build_knowledge_base(
            data_dir=args.data_dir,
            chunk_mode=args.chunk_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            persist_directory=args.persist_dir,
            top_k=args.top_k
        )

    rag_chain = build_rag_chain(retriever)

    # Single query mode
    if args.query:
        ask(rag_chain, retriever, args.query)
        return

    # Interactive chat mode
    print("\n" + "=" * 60)
    print("🤖 RAG Interactive Chat Session (Type 'exit' or 'quit' to end)")
    print("=" * 60)
    while True:
        try:
            user_input = input("\n💬 Enter question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break
            ask(rag_chain, retriever, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session ended.")
            break

if __name__ == "__main__":
    main()
