"""
=============================================================================
Module: RAG Generation & Execution Chain (local/chain.py)
Project: Production RAG System with LangChain & Google Gemini
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Assembles prompt formatting with strict grounding, context budgeting,
    SQLite persistent LLM caching, and LangChain Expression Language (LCEL)
    chain execution with Google Gemini 2.5 Flash.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

import time
from langchain_core.documents import Document
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions using ONLY the context below.
If the answer isn't in the context, say you don't know; do not make anything up.
Cite the source_file for any fact you use.

Context:
{context}

Question: {question}

Answer:"""
)

def enable_llm_cache(persist: bool = True, db_path: str = "./llm_cache.db"):
    """
    persist=True  -> SQLiteCache, persists across sessions.
    persist=False -> InMemoryCache, lasts only for runtime lifetime.
    """
    if persist:
        set_llm_cache(SQLiteCache(database_path=db_path))
        print(f"[INFO] LLM persistent cache enabled (SQLite: {db_path})")
    else:
        set_llm_cache(InMemoryCache())
        print("[INFO] LLM in-memory cache enabled")

def format_docs(docs: list[Document], max_context_chars: int = 6000) -> str:
    """
    Concatenates retrieved documents into a tagged context string,
    enforcing max_context_chars budget to protect against runaway costs.
    """
    blocks = []
    total_chars = 0
    for d in docs:
        source = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
        block = f"[Source: {source}]\n{d.page_content}"
        if total_chars + len(block) > max_context_chars:
            print(f"[WARNING] Context limit ({max_context_chars} chars) reached; dropping lower-ranked chunks.")
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n---\n\n".join(blocks)

def build_rag_chain(
    retriever,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    max_context_chars: int = 6000
):
    """Wires together retriever, prompt, Gemini chat model, and string output parser."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens
    )

    def format_with_budget(docs):
        return format_docs(docs, max_context_chars=max_context_chars)

    chain = (
        {"context": retriever | format_with_budget, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

CACHE_LATENCY_THRESHOLD_SECONDS = 0.5

def ask(chain, retriever, question: str, show_sources: bool = True) -> str:
    """Invokes the RAG chain and reports response, latency, cache status, and cited sources."""
    start = time.perf_counter()
    answer = chain.invoke(question)
    elapsed = time.perf_counter() - start

    likely_cached = elapsed < CACHE_LATENCY_THRESHOLD_SECONDS
    status = "CACHED RESPONSE" if likely_cached else "LIVE GEMINI CALL"

    print(f"\nQuestion: {question}")
    print(f"Answer:\n{answer}\n")
    print(f"Latency: {elapsed:.3f}s  |  Source: {status}")

    if show_sources and retriever:
        docs = retriever.invoke(question)
        print("Sources retrieved:")
        for d in docs:
            source = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
            preview = d.page_content[:120].replace('\n', ' ')
            print(f"  - [{source}]: \"{preview}...\"")

    return answer
