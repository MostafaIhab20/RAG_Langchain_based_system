# SASA: Silicon & Architecture Search Assistant

A production-grade, EDA-style Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini** (`gemini-2.5-flash` and `models/gemini-embedding-2`), **Chroma DB**, and **BM25 Hybrid Search**.

**Author:** Mostafa Ihab  
**Date:** March 2026  
**Version:** 1.1.0  
**License:** MIT  

This repository provides two cohesive interfaces for exploring technical literature, hardware-software co-designs, and Post-Quantum Cryptography (PQC) accelerators:
1. **Interactive Command Shell (`sasa>`)**: A scriptable terminal environment with batch automation, live parameter tuning, and status telemetry.
2. **Streamlit Web UI**: A modern browser interface featuring conversational chat, live sliders, in-browser document upload, and expandable source citations.
3. **Google Colab Notebook (`colab/`)**: A standalone notebook for experimentation in cloud environments.

---

## Repository Structure

```
RAG_Langchain_based_system/
├── colab/
│   └── RAG_Langchain_system.ipynb    # Google Colab notebook
├── local/
│   ├── __init__.py                   # Package exports
│   ├── app.py                        # Streamlit interactive Web UI (SASA)
│   ├── config.py                     # Environment & model initialization
│   ├── ingestion.py                  # Multi-format document loading & cleaning
│   ├── chunking.py                   # Recursive and semantic chunking
│   ├── vector_store.py               # Chroma vector store with deterministic upserts
│   ├── retriever.py                  # Hybrid retriever (BM25 + Chroma ensemble)
│   ├── chain.py                      # Prompt, context budgeting, caching & LCEL chain
│   ├── main.py                       # Interactive command shell (sasa>) & CLI
│   ├── run_eval.rag                  # Sample batch evaluation script
│   └── README.md                     # In-depth technical architecture & shell guide
├── data/                             # Source documents (PDF, DOCX, TXT, MD, CSV)
│   └── Hw-sw codesign/               # Subfolders are automatically scanned recursively
├── .env.example                      # Template for API credentials
├── .env                              # Your local environment secrets (gitignored)
├── .gitignore                        # Git exclusion rules (secrets, venv, Chroma DB, caches)
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

> [!TIP]
> **Deep-Dive Technical Architecture**: For a comprehensive layer-by-layer breakdown of the data ingestion, chunking, dual-database design, Reciprocal Rank Fusion mathematics, and command shell scripting, see [**`local/README.md`**](local/README.md).

---

## Prerequisites & Setup

### 1. Obtain Your Gemini API Key
* Get a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/).
* (Optional) Get a LangSmith API key from [LangSmith](https://smith.langchain.com/) for observability and tracing.

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (or edit `.env` directly) and paste your keys:

```bash
# .env
GOOGLE_API_KEY=AIzaSy...your_actual_key_here

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt...your_key_here
LANGCHAIN_PROJECT=gemini-rag-local
```

---

## Option 1: Running in Google Colab (`colab/`)

1. Go to [Google Colab](https://colab.research.google.com/) and upload `colab/RAG_Langchain_system.ipynb`.
2. Add your `GOOGLE_API_KEY` to Colab's Secrets manager (on the left sidebar) with notebook access enabled.
3. Run the notebook cells sequentially:
   - Cell 1 installs all required dependencies.
   - Ingestion cells provide an interactive file upload widget or automatically scan `./data` if mounted.
   - Cells run retrieval, caching tests, and interactive question answering.

---

## Option 2: Running Locally (`local/`)

### 1. Create a Virtual Environment (Recommended)

```bash
# In your project root:
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Or Windows (Command Prompt):
.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Documents
Place your PDFs, Word documents, Markdown, Text, or CSV files into the `data/` folder. Subdirectories (such as `data/Hw-sw codesign/`) are automatically scanned recursively.

### 4. Running the Interactive Shell (`sasa>`)
 
Launch the interactive command shell for command-driven RAG analysis:

```bash
python local/main.py
```

Once inside `sasa>`, you have full command-driven control:
* `status`: View loaded documents, Chroma database size, active chunk count, and parameters.
* `read_docs`: Inspect all files present in `data/` without modifying the database.
* `build_index`: Compile documents into Chroma and the BM25 index.
* `query <text>`: Ask a question and view answer, latency, cache status, and cited source excerpts.
* `chat`: Enter uninterrupted conversation mode (type questions directly; type `exit` to return to shell).
* `set <param> <value>`: Dynamically adjust `top_k`, `temperature`, `chunk_mode`, `max_context_chars`.
* `gui`: **Launch the Streamlit Web UI** in your browser directly from inside the shell!
* `source <file.rag>`: Execute a batch script of commands line-by-line.
* `exit`: Leave the shell.

---

### 5. Running the Web UI (Streamlit)

You can launch the visual browser interface in two convenient ways:

1. **Direct CLI command**:
   ```bash
   python local/main.py start_gui
   # or: python local/main.py --gui
   ```
2. **From inside the shell**:
   ```bash
   sasa> gui
   # or: sasa> start_gui
   ```

* **Interactive Chat**: Conversational interface with message history, Markdown, and LaTeX math rendering.
* **In-Browser Document Upload**: Drag and drop new PDFs into `data/` directly from the sidebar.
* **One-Click Reindex**: Click *"Rebuild Knowledge Base"* with a live progress spinner.
* **Hyperparameter Controls**: Real-time sliders for `top_k`, temperature, chunking method, and context limits.
* **Source Inspector**: Collapsible cards under each response displaying exact quoted excerpts and filenames.
* **Dark & Light Mode Support**: Fully responsive, high-contrast theme styling.

---

### 6. Batch Automation & Headless Execution (`-f`)

Execute automated analysis scripts non-interactively (ideal for benchmarking or evaluation):

```bash
python local/main.py -f local/run_eval.rag
```

---

## CLI Launch Modes

The SASA interface provides clean, intuitive CLI commands:

| Command | Description |
|:---|:---|
| `python local/main.py` | Runs main: launches the interactive `sasa>` command shell with all its internal commands (default). |
| `python local/main.py start_gui` | Directly opens the SASA Streamlit Web UI on `http://localhost:8501` (alias: `--gui`). |
| `python local/main.py -f <script.rag>` | Executes an automated batch script line-by-line and exits. |

---

## Key Architectural Features

Each component of the SASA pipeline is designed for high accuracy on specialized technical and hardware research literature:

1. [**Universal Multi-Format Ingestion**](local/README.md#1-ingestion--sanitization-localingestionpy):
   - High-fidelity PDF extraction via `PyMuPDFLoader` preserving complex multi-column layouts, tables, and equations.
   - Broad format support: DOCX, Markdown, Text, CSV, and Web URLs.
2. [**Data Sanitization & Deduplication**](local/README.md#1-ingestion--sanitization-localingestionpy):
   - Strips hidden zero-width and unprintable Unicode characters (`\u200b`, `\ufeff`).
   - Normalizes whitespace and drops empty or non-text scanned pages (<10 characters).
   - Content-hash (MD5) deduplication drops duplicate pages before chunking.
3. [**Syntax & Semantic Chunking Strategies**](local/README.md#2-chunking-layer-localchunkingpy):
   - Fast hierarchical recursive chunking (1,000 chars, 150 overlap) keeping sentences cohesive.
   - Embedding-based semantic chunking breaking at topic shifts (80th percentile threshold).
4. [**Dual Storage & Deterministic Upserts**](local/README.md#3-dual-storage-architecture-localvector_storepy):
   - Chroma vector DB indexes Gemini embeddings via HNSW graphs for fast vector search.
   - Deterministic MD5 chunk IDs (`text + source_file`) prevent duplicate vector embedding costs on incremental re-indexing.
   - Throttled batching (40 chunks) with exponential backoff respects API rate limits.
   - `indexed_chunks.pkl` snapshot enables loading BM25 into memory in **<20ms** without API calls.
5. [**Hybrid Search & Reciprocal Rank Fusion (RRF)**](local/README.md#4-hybrid-search-engine--reciprocal-rank-fusion-localretrieverpy):
   - Blends dense semantic vector retrieval (60% weight) with BM25 sparse keyword retrieval (40% weight).
   - Accurately captures both high-level concepts and exact hardware acronyms (`CV-X-IF`, `ML-KEM`, `NTT`, `OpenTitan`).
6. [**Context Budgeting & SQLite LLM Caching**](local/README.md#5-generation--caching-localchainpy):
   - Hard character budget limiter (default 6,000 characters) prevents runaway prompt costs and context degradation.
   - Local SQLite query cache (`llm_cache.db`) returns instant (<0.02s) responses for repeated queries.
   - Strict zero-hallucination grounding prompt with mandatory source citation rules.

---

## Explore In-Depth Technical Documentation

Want to learn more about the mathematics, dataflow diagrams, and underlying architecture?

**Read the comprehensive technical engineering guide in [local/README.md](local/README.md)**:
* **System Architecture**: High-level dataflow diagrams from raw documents to GUI/shell.
* **Reciprocal Rank Fusion Mathematics**: Complete RRF ranking formulas and weighting explanations.
* **Dual Storage Deep Dive**: Why Chroma and `indexed_chunks.pkl` work together for zero-cost BM25 loading.
* **SASA Interactive Shell Guide**: Command syntax, parameter tuning, and batch scripting instructions.
* **File Inventory & Architecture Mapping**: Complete table detailing every file in the package and its responsibilities.
