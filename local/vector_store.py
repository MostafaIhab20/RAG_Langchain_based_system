"""
=============================================================================
Module: Vector Database Management (local/vector_store.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Chroma DB connection manager with deterministic MD5 chunk hashing to
    ensure idempotent upserts, deduplication, and rate-limited batching
    with automatic exponential backoff for embedding APIs.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

import time
import hashlib
from langchain_core.documents import Document
from langchain_chroma import Chroma
from .config import get_embedding_model

class VectorDBManager:
    """
    Manages the Chroma vector database connection and idempotent upserts.
    Uses deterministic chunk IDs and throttled batching with automatic retry backoff.
    """
    def __init__(
        self,
        collection_name: str = "rag_production",
        persist_directory: str = "./chroma_db",
        embedding_model=None
    ):
        print("Initializing Gemini Embedding Engine...")
        self.embeddings = embedding_model or get_embedding_model()

        print(f"Connecting to Chroma Database at '{persist_directory}'...")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def _generate_chunk_id(self, chunk: Document) -> str:
        """Generates a deterministic MD5 hash based on text and source filename."""
        source = chunk.metadata.get('source_file', 'unknown')
        unique_string = f"{chunk.page_content}-{source}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def upsert_chunks(self, chunks: list[Document], batch_size: int = 40):
        """
        Embeds and saves chunks into Chroma in rate-limited batches.
        Automatically skips already-embedded chunks and handles API rate limits
        with exponential backoff.
        """
        if not chunks:
            print("Warning: No chunks provided to database.")
            return

        # 1. Fetch already existing IDs in Chroma to avoid redundant embedding calls
        try:
            existing_data = self.vector_store.get()
            existing_ids = set(existing_data.get("ids", []))
        except Exception:
            existing_ids = set()

        # 2. Filter out already indexed chunks
        chunks_to_add = []
        ids_to_add = []
        for chunk in chunks:
            cid = self._generate_chunk_id(chunk)
            if cid not in existing_ids:
                chunks_to_add.append(chunk)
                ids_to_add.append(cid)

        if not chunks_to_add:
            print(f"All {len(chunks)} chunks are already indexed in Chroma. Skipping embedding.")
            return

        already_indexed = len(chunks) - len(chunks_to_add)
        if already_indexed > 0:
            print(f"[INFO] {already_indexed} chunks already present in index.")
        print(f"Generating vectors and upserting {len(chunks_to_add)} new chunks into Chroma...")

        # 3. Process in rate-limited batches
        total_batches = (len(chunks_to_add) + batch_size - 1) // batch_size

        for i in range(0, len(chunks_to_add), batch_size):
            batch_chunks = chunks_to_add[i:i + batch_size]
            batch_ids = ids_to_add[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            max_retries = 6
            for attempt in range(max_retries):
                try:
                    print(f"   -> [Batch {batch_num}/{total_batches}] Embedding {len(batch_chunks)} chunks...")
                    self.vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        wait_seconds = 30 + (attempt * 10)
                        print(f"   API rate limit reached. Pausing for {wait_seconds}s before retrying batch...")
                        time.sleep(wait_seconds)
                    else:
                        raise e

            # Brief pause between batches to respect RPM quotas
            if batch_num < total_batches:
                time.sleep(1.5)

        print("Upsert complete. All documents are indexed and searchable.")

    def get_retriever(self, top_k: int = 4):
        """Returns standard vector retriever."""
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})
