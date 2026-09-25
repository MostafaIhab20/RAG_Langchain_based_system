# Technical Architecture & EDA Shell Guide - SASA (`local/`)

Welcome to the internal engineering documentation of **SASA: Silicon & Architecture Search Assistant**.

**Author:** Mostafa Ihab  
**Date:** March 2026  
**Version:** 1.1.0  
**License:** MIT  

---

## System Architecture

The system implements a production-grade, two-stage Retrieval-Augmented Generation (RAG) pipeline designed specifically for highly technical domains such as Post-Quantum Cryptography (PQC), hardware accelerators, and RISC-V SoC architectures.

```
                    ┌──────────────────────────────────────────────┐
                    │               RAW DOCUMENTS                  │
                    │      PDF, DOCX, TXT, MD, CSV, Web URLs       │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & DATA SANITIZATION (ingestion.py)                                        │
│ • Universal Loaders (PyMuPDF for layout-aware table/text preservation)                 │
│ • Cleansing (Strip zero-width characters, normalize whitespace, discard noise)         │
│ • Deduplication (MD5 hash check to eliminate exact duplicate files/pages)              │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. CHUNKING LAYER (chunking.py)                                                        │
│ • Recursive Character: Hierarchical split (1000 chars, 150 overlap) preserves context  │
│ • Semantic Chunking: Splits by sentence embedding distance at 80th percentile          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
┌─────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│ Dense Embeddings (vector_store.py)      │ │ Sparse Inverted Index (retriever.py)      │
│ • Gemini Embedding 2 (models/...)       │ │ • BM25Okapi Term-Frequency Tokenizer      │
│ • Chroma Vector Database (HNSW graph)   │ │ • Zero-vector RAM index (indexed_chunks)  │
│ • Deterministic chunk IDs (MD5 upsert)  │ │ • Exact keyword / hardware acronym match  │
└────────────────────┬────────────────────┘ └─────────────────────┬─────────────────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. HYBRID RETRIEVAL & RECIPROCAL RANK FUSION (retriever.py)                            │
│ • Ensemble Retriever: Blends Dense (0.60 weight) + BM25 Sparse (0.40 weight)          │
│ • RRF Formula: Score(d) = Σ w_i / (k + rank_i(d)) with k = 60                          │
│ • Returns Top-K relevant chunks with metadata (source file, page, score)               │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. CACHING, PROMPT AUGMENTATION & GENERATION (chain.py)                                │
│ • Local SQLite Query Cache (llm_cache.db): Returns cached answers in <0.02s            │
│ • Context Budget Limiter: Hard cap (default 6,000 chars) prevents runaway token costs  │
│ • Grounding Prompt: Strict zero-hallucination instruction + mandatory file citations   │
│ • Gemini 2.5 Flash: Generates concise, fact-grounded answer via LCEL                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
┌─────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│ Interactive Shell (main.py)             │ │ Streamlit Web Application (app.py)        │
│ • Interactive CLI (sasa>)               │ │ • Full browser chat interface             │
│ • Command-driven workflow & batch files │ │ • In-browser drag-and-drop document upload│
│ • Status telemetry, runtime set/get     │ │ • Dynamic parameter sliders               │
│ • Seamless GUI launch: 'gui' command    │ │ • Expandable source citation inspector    │
└─────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## Layer-by-Layer Technical Deep Dive

### 1. Ingestion & Sanitization [`local/ingestion.py`]
* **PyMuPDF (`PyMuPDFLoader`)**: Selected over standard `PyPDF` because PyMuPDF preserves complex multi-column scientific paper layouts, block order, and tabular structures without mangling text streams.
* **Text Cleansing**: Eliminates invisible zero-width spaces (`\u200b`, `\ufeff`) often embedded in PDF copy-pastes that throw off tokenizer models. Collapses redundant whitespace and filters empty scanned pages (<10 chars).
* **Pre-Chunking Deduplication**: Generates an MD5 hash of raw text to eliminate identical files or duplicate pages before wasting compute on chunking and embedding.

### 2. Chunking Layer [`local/chunking.py`]
* **Recursive Chunker (`RecursiveCharacterTextSplitter`)**:
  * Default chunk size: 1,000 characters; overlap: 150 characters.
  * Hierarchical split separators: `["\n\n", "\n", ". ", " ", ""]` ensuring sentences and paragraphs remain cohesive.
* **Semantic Chunker (`SemanticChunker`)**:
  * Evaluates cosine distance between consecutive sentences using Gemini embeddings.
  * Splits text when semantic distance exceeds the 80th percentile threshold, creating concept-pure passages.

### 3. Dual Storage Architecture [`local/vector_store.py`]
Why do we store both Chroma and a `.pkl` file?
1. **Chroma DB (`./chroma_db/`)**:
   * Stores high-dimensional floating-point vectors produced by `models/gemini-embedding-2`.
   * Indexes vectors using Hierarchical Navigable Small World (HNSW) graphs for sub-5ms cosine similarity nearest-neighbor lookup.
   * **Deterministic Upsert IDs**: Every chunk ID is generated as `MD5(text + source_file)`. When new files are added, existing chunks are detected and skipped, preventing duplicate embeddings and preserving quota.
    * **API Rate-Limit Handling**: Embeds in throttled batches of 40 chunks with automatic exponential backoff (30s cooldown) to respect provider rate limits.
2. **Chunk Snapshot (`chroma_db/indexed_chunks.pkl`)**:
   * Raw text chunks are serialized into a lightweight `.pkl` binary.
   * BM25 only needs text strings (no vectors). Loading `indexed_chunks.pkl` takes **20ms** on app startup, building the BM25 search index directly in RAM with **zero API calls**.

### 4. Hybrid Search Engine & Reciprocal Rank Fusion [`local/retriever.py`]
Technical papers require finding exact hardware acronyms (`CV-X-IF`, `ML-KEM-768`, `NTT`, `OpenTitan`) as well as broad conceptual ideas (`"energy-efficient polynomial multiplication"`).
* **Dense Search (Chroma)**: 60% weight (semantic meaning).
* **Sparse Search (BM25)**: 40% weight (exact keyword frequencies).
* **Reciprocal Rank Fusion (RRF)**:
  \[
  \text{RRF Score}(d) = 0.4 \times \frac{1}{60 + \text{Rank}_{\text{BM25}}(d)} + 0.6 \times \frac{1}{60 + \text{Rank}_{\text{Chroma}}(d)}
  \]
  Documents identified by both keyword matches and conceptual relevance receive high rank multipliers and are prioritized at the top of the context block.

### 5. Generation & Caching [`local/chain.py`]
* **SQLite Query Cache (`llm_cache.db`)**:
  * Hashes `(prompt + retrieved_context)`.
  * If an identical query is made, response is served in **<0.02 seconds**, entirely bypassing LLM inference.
* **Context Budgeting (`format_docs`)**:
  * Enforces a hard budget (default 6,000 characters).
  * Automatically trims lower-ranked chunks if context exceeds threshold, protecting against prompt stuffing and latency spikes.
* **Strict Grounding Prompt**:
  * Enforces strict truthfulness: *"Answer using ONLY the context below. If the answer isn't in the context, say you don't know."*
  * Requires source citations for every stated fact.

---

## Interactive EDA Shell Guide (`local/main.py`)

The CLI operates as an interactive command shell offering robust, scriptable terminal control:

```
=============================================================================
  ███████╗ █████╗  ███████╗ █████╗ 
  ██╔════╝██╔══██╗ ██╔════╝██╔══██╗
  ███████╗███████║ ███████╗███████║
  ╚════██║██╔══██║ ╚════██║██╔══██║
  ███████║██║  ██║ ███████║██║  ██║
  ╚══════╝╚═╝  ╚═╝ ╚══════╝╚═╝  ╚═╝
  
  SASA: Silicon & Architecture Search Assistant
  Production RAG System with LangChain & Google Gemini
  Author: Mostafa Ihab  |  Version: 1.1.0  |  Mode: Interactive Shell
=============================================================================
sasa> 
```

### Launch Modes

| Invocation | Behavior |
|:---|:---|
| `python local/main.py` | Runs main: launches the interactive `sasa>` shell with all its commands (default). |
| `python local/main.py start_gui` | Directly launches the SASA Streamlit Web UI on `http://localhost:8501` (alias: `--gui`). |
| `python local/main.py -f script.rag` | Executes a batch script line-by-line in headless mode and exits. |

---

### Command Reference Table

| Command | Syntax | Description |
|:---|:---|:---|
| **`help`** | `help [command]` | Displays command list or detailed syntax for a specific command. |
| **`status`** | `status` (or `report_status`) | Prints full telemetry: documents count, Chroma DB size, cached chunks, SQLite cache size, and active hyperparameters. |
| **`read_docs`** | `read_docs [dir_path]` | Scans and lists all supported files in `data/` with file sizes without modifying database. |
| **`build_index`** | `build_index [--mode recursive\|semantic]` | Compiles documents into Chroma vector database and caches BM25 chunks. |
| **`query`** | `query <question>` (or `ask`) | Executes a question through the hybrid retriever and Gemini model. Displays answer, latency, cache status, and cited source excerpts. |
| **`chat`** | `chat` | Enters continuous chat mode where questions can be typed without typing `query` each time. Type `exit` to return to `sasa>`. |
| **`set`** | `set <param> <value>` | Modifies runtime variables: `top_k`, `temperature`, `chunk_mode`, `chunk_size`, `chunk_overlap`, `max_context_chars`, `cache`. |
| **`get`** | `get [param]` | Prints current value of specified parameter or all runtime parameters. |
| **`gui`** | `gui` (or `start_gui`) | Launches the Streamlit Web UI in background and opens default browser automatically. |
| **`clear_cache`**| `clear_cache` | Wipes the local SQLite LLM cache (`llm_cache.db`). |
| **`source`** | `source <script_file>` (or `run`) | Executes commands from a text/script file line-by-line. |
| **`exit`** | `exit` (or `quit`) | Exits the shell session. |

---

### Scripting with Batch Files (`.rag`)

You can automate common research workflows using `.rag` script files.

**Example `local/run_eval.rag`**:
```bash
# ---------------------------------------------
# Automation Script: PQC Accelerator Analysis
# ---------------------------------------------

# 1. Check system telemetry
status

# 2. Configure retrieval parameters
set top_k 4
set temperature 0.0

# 3. Execute targeted questions
query "What is the role of the CV-X-IF interface in the ATHOS accelerator?"
query "How does NTT polynomial multiplication accelerate CRYSTALS-Kyber?"

# 4. View updated cache status
status
```

Execute the batch script from your terminal:
```bash
python local/main.py -f local/run_eval.rag
```
Or run it from within the shell:
```bash
sasa> source local/run_eval.rag
```

---

## File Inventory & Architecture Mapping

The table below details every file in the `local/` package, its role in the RAG pipeline, and key technical responsibilities:

| File | Type | Primary Role | Key Technical Features & Responsibilities |
|:---|:---|:---|:---|
| **`__init__.py`** | Python Package Init | Module Exports & Metadata | Exposes top-level package API, versioning dunder variables (`__version__ = "1.1.0"`, `__author__ = "Mostafa Ihab"`), and submodule imports. |
| **`config.py`** | Configuration | Environment & Models | Loads `.env` secrets, initializes Google Gemini embeddings (`models/gemini-embedding-2`), sets custom User-Agent headers, and configures LangSmith tracing. |
| **`ingestion.py`** | Data Layer | Multi-Format Ingestion & Cleaning | Layout-aware parsing with `PyMuPDFLoader` (PDF), `Docx2txtLoader`, `TextLoader`, `UnstructuredMarkdownLoader`, `CSVLoader`, and `WebBaseLoader`. Strips zero-width Unicode, drops empty/scanned pages, and performs pre-chunking MD5 content deduplication. |
| **`chunking.py`** | Data Layer | Text Segmentation | Implements syntax-based `RecursiveCharacterTextSplitter` (1,000 chars, 150 overlap) and embedding-driven `SemanticChunker` (80th percentile semantic distance threshold). |
| **`vector_store.py`** | Storage Layer | Chroma DB & Deterministic Upsert | Manages persistent Chroma vector store. Computes deterministic MD5 IDs (`MD5(text + source)`) to enable idempotent indexing and zero duplicate embedding costs. Implements throttled batching (40 chunks) with exponential backoff for API rate limits. |
| **`retriever.py`** | Retrieval Layer | Hybrid Search & Fusion | Integrates dense semantic retrieval (Chroma, 60% weight) with sparse keyword retrieval (`BM25Okapi`, 40% weight) using Reciprocal Rank Fusion (RRF with $k=60$) to balance concept matching and technical acronym precision. |
| **`chain.py`** | Generation Layer | LCEL Chain, Caching & Budgeting | Constructs the LangChain LCEL pipeline with Google Gemini 2.5 Flash. Implements persistent SQLite query caching (`llm_cache.db`), hard context character budgeting (default 6,000 chars), and strict zero-hallucination grounding prompts with citation mandates. |
| **`main.py`** | CLI & Shell Interface | SASA Interactive EDA Shell | Interactive REPL (`sasa>`) offering runtime status telemetry, document inspection (`read_docs`), index compilation (`build_index`), targeted queries, continuous chat mode (`sasa-chat>`), dynamic parameter mutations (`set`/`get`), script execution (`source`), headless batch execution (`-f`), and direct GUI launch. |
| **`app.py`** | Graphical UI | Streamlit Web Application | Responsive browser chat interface featuring dark/light support, in-browser drag-and-drop file upload to `data/`, one-click knowledge base reindexing, real-time hyperparameter sliders, cache toggling, and collapsible source citation cards. |
| **`run_eval.rag`** | Scripting | Batch Automation Script | Example scripted workflow demonstrating headless system telemetry inspection, document verification, parameter calibration, and targeted technical Q&A execution on PQC accelerator topics. |
| **`README.md`** | Documentation | Technical Architecture Guide | In-depth engineering specification detailing dataflow diagrams, RRF mathematics, dual-storage persistence rationale, command references, and scripting guides. |

