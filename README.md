# Production RAG System with LangChain & Google Gemini

A modular, production-oriented Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini** (`gemini-2.5-flash` and `models/gemini-embedding-2`), **Chroma DB**, and **BM25 Hybrid Search**.

**Author:** Mostafa Ihab  
**Date:** 2026-09-12  
**Version:** 1.0.0

This repository is structured into two ready-to-use workflows:
1. **`colab/`**: Standalone Jupyter notebook ready for Google Colab testing.
2. **`local/`**: Modular Python package with an interactive CLI for running locally.

---

## 📁 Repository Structure

```
RAG_Langchain_based_system/
├── colab/
│   └── RAG_Langchain_system.ipynb    # Google Colab notebook
├── local/
│   ├── __init__.py                   # Package exports
│   ├── app.py                        # Streamlit interactive Web UI
│   ├── config.py                     # Environment & model initialization
│   ├── ingestion.py                  # Multi-format document loading & cleaning
│   ├── chunking.py                   # Recursive and semantic chunking
│   ├── vector_store.py               # Chroma vector store with deterministic upserts
│   ├── retriever.py                  # Hybrid retriever (BM25 + Chroma ensemble)
│   ├── chain.py                      # Prompt, context budgeting, caching & LCEL chain
│   └── main.py                       # CLI & terminal chat application
├── data/                             # Put your raw documents here (PDF, DOCX, TXT, MD, CSV)
│   └── Hw-sw codesign/               # Subfolders are automatically scanned recursively
├── .env.example                      # Template for API credentials
├── .env                              # Your local environment secrets (gitignored)
├── .gitignore                        # Git exclusion rules (secrets, venv, Chroma DB, caches)
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

---

## ⚙️ Prerequisites & Setup

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

## 🚀 Option 1: Running in Google Colab (`colab/`)

1. Go to [Google Colab](https://colab.research.google.com/) and upload `colab/RAG_Langchain_system.ipynb`.
2. Add your `GOOGLE_API_KEY` to Colab's Secrets manager (the 🔑 icon on the left sidebar) with notebook access enabled.
3. Run the notebook cells sequentially:
   - Cell 1 installs all required dependencies.
   - Ingestion cells provide an interactive file upload widget or automatically scan `./data` if mounted.
   - Cells run retrieval, caching tests, and interactive question answering.

---

## 💻 Option 2: Running Locally (`local/`)

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

### 4. Launch the Web UI (Recommended)
Start the modern, browser-based chat application:

```bash
streamlit run local/app.py
```
* **Interactive Chat**: Conversational interface with message history, markdown, and LaTeX rendering.
* **In-Browser Document Upload**: Drag and drop new documents into `data/` directly from the sidebar.
* **One-Click Reindex**: Click *"Rebuild Knowledge Base"* to index new documents into Chroma & BM25 with a progress spinner.
* **Hyperparameter Controls**: Sliders for `top_k` (1–10), LLM temperature (0.0–1.0), chunking algorithm (`recursive` vs `semantic`), and context budget.
* **Source Inspector**: Collapsible citation accordions under each answer displaying referenced text snippets and source filenames.
* **Performance Badges**: Real-time response latency and SQLite cache indicators.

---

### 5. Alternative: CLI & Terminal Chat

If you prefer using the terminal:

* **Build the index**:
  ```bash
  python local/main.py --build
  ```
* **Interactive terminal chat**:
  ```bash
  python local/main.py
  ```
* **Single query**:
  ```bash
  python local/main.py --query "What are the main hardware bottlenecks in CRYSTALS-Kyber acceleration?"
  ```

---

## 🎛️ CLI Options & Customization

You can customize chunking and retrieval parameters via command-line arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `./data` | Directory containing source documents |
| `--persist-dir` | `./chroma_db` | Storage path for Chroma vector database |
| `--chunk-mode` | `recursive` | Chunking algorithm: `recursive` or `semantic` |
| `--chunk-size` | `1000` | Target character size for recursive chunker |
| `--chunk-overlap`| `150` | Character overlap between adjacent chunks |
| `--top-k` | `4` | Number of context chunks retrieved for prompt |
| `--build` | `False` | Forces rebuilding the Chroma & BM25 index |
| `--query` | `None` | Executes a single query and prints the answer |
| `--no-cache` | `False` | Disables SQLite LLM caching |

---

## 🔍 Key Architectural Features

1. **Universal Multi-Format Ingestion**:
   - PDF (via `PyMuPDFLoader` for high-fidelity table and layout preservation)
   - DOCX, Markdown, Text, CSV, and Web URLs.
2. **Data Sanitization & Deduplication**:
   - Strips hidden zero-width and unprintable Unicode characters (`\u200b`, `\ufeff`).
   - Collapses consecutive whitespace runs.
   - Skips empty or non-text scanned pages (<10 chars).
   - Content-hash (MD5) deduplication drops duplicate pages before chunking.
3. **Hybrid Search (Ensemble Retriever)**:
   - Combines dense semantic vector retrieval (Chroma + Gemini Embeddings) with sparse keyword retrieval (BM25) using Reciprocal Rank Fusion (RRF).
4. **Deterministic Upserts**:
   - Chunks receive deterministic MD5 IDs (`{text}-{source_file}`) so re-indexing avoids database pollution.
5. **Context Character Budgeting**:
   - `format_docs` strictly enforces a character budget (default 6,000 characters) to prevent runaway prompt costs and context degradation.
6. **SQLite Persistent LLM Caching**:
   - Identical queries are returned instantaneously from a local SQLite database (`llm_cache.db`), saving API tokens.
