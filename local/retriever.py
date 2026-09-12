"""
=============================================================================
Module: Hybrid Retrieval Engine (local/retriever.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: 2026-09-12
Version: 1.0.0
Description:
    Combines dense semantic vector search (Chroma) with sparse exact-keyword
    search (BM25) using Reciprocal Rank Fusion (EnsembleRetriever).
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.0.0"
__date__ = "2026-09-12"

from langchain_community.retrievers import BM25Retriever

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

class HybridRetrieverManager:
    """
    Combines dense semantic vector search (Chroma) with sparse exact-keyword search (BM25)
    using Reciprocal Rank Fusion (EnsembleRetriever).
    """
    def __init__(
        self,
        vector_store,
        documents: list,
        top_k: int = 4,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        print("⚙️ Initializing Hybrid Retrieval Engine (BM25 + Semantic)...")

        # 1. BM25 Sparse Keyword Index
        print(f"   -> Indexing {len(documents)} document chunk(s) into BM25...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = top_k

        # 2. Dense Vector Index (Chroma)
        print("   -> Connecting Chroma dense vector retriever...")
        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

        # 3. Weighted Ensemble (RRF)
        print(f"   -> Calibrating Ensemble (Weights: {bm25_weight} BM25 / {vector_weight} Chroma)...")
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[bm25_weight, vector_weight]
        )
        print("✅ Hybrid Search is ready.")

    def get_retriever(self):
        """Returns the fused retriever interface."""
        return self.hybrid_retriever
