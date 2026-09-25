"""
=============================================================================
Module: Streamlit Web UI Application (local/app.py)
Project: SASA - Silicon & Architecture Search Assistant
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Modern, interactive web-based UI for SASA (LangChain RAG pipeline).
    Features conversational chat, document upload & management, live
    hyperparameter tuning, and expandable source citations.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so modules can be imported
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

def check_dependencies():
    """Verify dependencies before rendering UI."""
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
    return missing

missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Missing required dependencies: {', '.join(missing_deps)}")
    st.info("Please install them in your terminal: `pip install -r requirements.txt`")
    st.stop()

from local.config import setup_environment
from local.chain import build_rag_chain, enable_llm_cache, CACHE_LATENCY_THRESHOLD_SECONDS
from local.main import load_existing_retriever, build_knowledge_base

# Page Setup
st.set_page_config(
    page_title="SASA | Post-Quantum Cryptography & Architecture RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E88E5;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .badge-cache {
        background-color: rgba(46, 125, 50, 0.18);
        color: #4CAF50;
        border: 1px solid rgba(76, 175, 80, 0.4);
    }
    .badge-live {
        background-color: rgba(25, 118, 210, 0.18);
        color: #42A5F5;
        border: 1px solid rgba(66, 165, 245, 0.4);
    }
    .source-card {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 4px solid #1E88E5;
        padding: 10px 14px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
    .source-title {
        font-weight: 600;
        color: #42A5F5;
        margin-bottom: 4px;
    }
    .source-snippet {
        font-style: italic;
        opacity: 0.9;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)

setup_environment()

DATA_DIR = PROJECT_ROOT / "data"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Scan existing documents
def get_existing_files(directory: Path):
    if not directory.exists():
        return []
    return [f for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".csv"]]

existing_docs = get_existing_files(DATA_DIR)

# Sidebar
with st.sidebar:
    st.title("SASA Control Panel")
    st.caption("Author: Mostafa Ihab · LangChain & Gemini")
    st.markdown("---")

    # 1. Knowledge Base Status
    st.subheader("Knowledge Base")
    db_exists = PERSIST_DIR.exists() and (PERSIST_DIR / "indexed_chunks.pkl").exists()

    if db_exists:
        st.success(f"Vector Index Active ({len(existing_docs)} files in `data/`)")
    else:
        st.warning(f"Index Not Built ({len(existing_docs)} files ready to index)")

    # 2. Document Upload
    with st.expander("Upload New Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Add files to data directory",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True
        )
        if uploaded_files:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_files:
                save_path = DATA_DIR / uf.name
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
            st.success(f"Saved {len(uploaded_files)} file(s) to `{DATA_DIR.name}/`")
            st.rerun()

    # 3. View Loaded Documents
    with st.expander(f"Document Inventory ({len(existing_docs)})", expanded=False):
        if existing_docs:
            for d in existing_docs:
                st.markdown(f"- `{d.name}`")
        else:
            st.info("No documents found in `data/`.")

    # 4. Hyperparameter Settings
    st.subheader("Pipeline Settings")
    top_k = st.slider("Retrieved Chunks (top_k)", min_value=1, max_value=10, value=4, step=1)
    temperature = st.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    chunk_mode = st.selectbox("Chunking Algorithm", ["recursive", "semantic"], index=0)
    max_context_chars = st.slider("Max Context Chars", min_value=2000, max_value=12000, value=6000, step=1000)
    enable_cache = st.toggle("Enable SQLite Cache", value=True)

    st.markdown("---")

    # 5. Rebuild Action
    if st.button("Rebuild Knowledge Base", use_container_width=True, type="primary"):
        with st.spinner("Indexing documents into Chroma & BM25..."):
            try:
                retriever, _, _ = build_knowledge_base(
                    data_dir=str(DATA_DIR),
                    chunk_mode=chunk_mode,
                    persist_directory=str(PERSIST_DIR),
                    top_k=top_k
                )
                st.session_state.retriever = retriever
                st.success("Index successfully rebuilt.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to build index: {e}")

    # 6. Reset Chat
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Ensure retriever is loaded
if st.session_state.retriever is None:
    loaded_retriever, _ = load_existing_retriever(persist_directory=str(PERSIST_DIR), top_k=top_k)
    st.session_state.retriever = loaded_retriever

# Main Content Area
st.markdown('<div class="main-header">SASA: Post-Quantum Cryptography Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions across your research papers, hardware-software co-design architectures, and RISC-V accelerators.</div>', unsafe_allow_html=True)

# Check API Key
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("**GOOGLE_API_KEY is not configured.** Please add it to your `.env` file in the project root to enable question answering.")

# Enable/disable cache
enable_llm_cache(persist=enable_cache)

# Welcome State with Suggested Questions
if not st.session_state.messages:
    st.info("**Welcome to SASA.** Ask any question about your documents, hardware accelerators, or PQC algorithms.")
    st.markdown("##### Suggested Questions:")
    col1, col2 = st.columns(2)
    sample_q1 = "What are the main hardware bottlenecks in CRYSTALS-Kyber acceleration?"
    sample_q2 = "How does the NTT (Number Theoretic Transform) accelerator improve ML-KEM performance?"
    sample_q3 = "What is the role of the RISC-V CV-X-IF interface in the ATHOS accelerator?"
    sample_q4 = "How are side-channel protections implemented in PQC hardware verification?"

    with col1:
        if st.button(sample_q1, use_container_width=True):
            st.session_state.prompt_to_submit = sample_q1
            st.rerun()
        if st.button(sample_q2, use_container_width=True):
            st.session_state.prompt_to_submit = sample_q2
            st.rerun()
    with col2:
        if st.button(sample_q3, use_container_width=True):
            st.session_state.prompt_to_submit = sample_q3
            st.rerun()
        if st.button(sample_q4, use_container_width=True):
            st.session_state.prompt_to_submit = sample_q4
            st.rerun()

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            badge_class = "badge-cache" if msg.get("is_cached") else "badge-live"
            status_label = "CACHED (SQLite)" if msg.get("is_cached") else "LIVE CALL (Gemini 2.5 Flash)"
            st.markdown(
                f'<span class="metric-badge {badge_class}">{status_label} · {msg.get("latency", 0):.3f}s</span>',
                unsafe_allow_html=True
            )
            with st.expander(f"{len(msg['sources'])} Cited Source(s)", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">[{idx}] {src['source_file']}</div>
                        <div class="source-snippet">"{src['snippet']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

# Check for button-clicked prompts
prompt_input = st.chat_input("Ask a question about your documents...")
prompt = prompt_input or st.session_state.pop("prompt_to_submit", None)

if prompt:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Check if Knowledge Base is Available
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.error("Knowledge base index not found. Please click **'Rebuild Knowledge Base'** in the sidebar first.")
    else:
        # 3. Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                try:
                    rag_chain = build_rag_chain(
                        st.session_state.retriever,
                        temperature=temperature,
                        max_context_chars=max_context_chars
                    )

                    start_time = time.perf_counter()
                    answer = rag_chain.invoke(prompt)
                    elapsed = time.perf_counter() - start_time

                    is_cached = elapsed < CACHE_LATENCY_THRESHOLD_SECONDS

                    # Retrieve source documents for citation display
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    sources = []
                    for d in retrieved_docs[:top_k]:
                        sf = d.metadata.get("source_file", d.metadata.get("source", "Document"))
                        sources.append({
                            "source_file": sf,
                            "snippet": d.page_content[:200].replace("\n", " ") + "..."
                        })

                    # Display response
                    st.markdown(answer)

                    badge_class = "badge-cache" if is_cached else "badge-live"
                    status_label = "CACHED (SQLite)" if is_cached else "LIVE CALL (Gemini 2.5 Flash)"
                    st.markdown(
                        f'<span class="metric-badge {badge_class}">{status_label} · {elapsed:.3f}s</span>',
                        unsafe_allow_html=True
                    )

                    if sources:
                        with st.expander(f"{len(sources)} Cited Source(s)", expanded=False):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">[{idx}] {src['source_file']}</div>
                                    <div class="source-snippet">"{src['snippet']}"</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # Save to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": elapsed,
                        "is_cached": is_cached
                    })

                except Exception as e:
                    st.error(f"Error generating response: {e}")
