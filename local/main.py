"""
=============================================================================
Module: Interactive Command Shell & CLI (local/main.py)
Project: SASA - Silicon & Architecture Search Assistant
Author: Mostafa Ihab
Date: March 2026
Version: 1.1.0
Description:
    Interactive EDA-style command shell environment supporting command-driven
    RAG operations, batch scripting (-f), and direct GUI launch.
=============================================================================
"""

__author__ = "Mostafa Ihab"
__version__ = "1.1.0"
__date__ = "March 2026"

import sys
import time
import pickle
import shlex
import argparse
import subprocess
import webbrowser
from pathlib import Path

# Add project root to sys.path so modules can be run directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
        print("\n[ERROR] Missing required dependencies:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease install them by running:")
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

    print("=" * 65)
    print("STEP 1: LOADING DOCUMENTS")
    print("=" * 65)
    raw_docs = load_from_directory(data_dir)
    if not raw_docs:
        raise ValueError(
            f"No documents found in '{data_dir}'. Make sure your files (PDF, DOCX, TXT, MD, CSV) are in this directory."
        )

    print("\n" + "=" * 65)
    print("STEP 2: DATA CLEANING & SANITIZATION")
    print("=" * 65)
    cleaned_docs = process_and_clean_docs(raw_docs)

    print("\n" + "=" * 65)
    print("STEP 3: CONTENT-HASH DEDUPLICATION")
    print("=" * 65)
    unique_docs = deduplicate_docs(cleaned_docs)

    print("\n" + "=" * 65)
    print(f"STEP 4: CHUNKING (Algorithm: {chunk_mode.upper()})")
    print("=" * 65)
    if chunk_mode == "semantic":
        chunker = SemanticChunker(get_embedding_model(), percentile=semantic_percentile)
    else:
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    final_chunks = chunker.split(unique_docs)

    print("\n" + "=" * 65)
    print("STEP 5: VECTOR STORE & BM25 PERSISTENCE")
    print("=" * 65)
    db_manager = VectorDBManager(collection_name=collection_name, persist_directory=persist_directory)
    db_manager.upsert_chunks(final_chunks)

    # Save chunks locally so future query sessions can instantiate BM25 without re-parsing files
    cache_path = get_chunks_cache_path(persist_directory)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(final_chunks, f)
    print(f"[INFO] Cached chunk index for BM25 to '{cache_path}'")

    print("\n" + "=" * 65)
    print("STEP 6: HYBRID RETRIEVER INITIALIZATION")
    print("=" * 65)
    hybrid_manager = HybridRetrieverManager(
        vector_store=db_manager.vector_store,
        documents=final_chunks,
        top_k=top_k,
    )

    print("\nKnowledge base successfully built and ready for search.")
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

# =============================================================================
# Interactive Command Shell Implementation
# =============================================================================

class InteractiveRagShell:
    """
    Interactive command-line shell environment for structured RAG workflows.
    Supports command dispatch, runtime variable inspection/mutation, script sourcing,
    and background GUI launching.
    """

    BANNER = r"""
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
  Type 'help' or '?' for command list. Type 'exit' to quit.
=============================================================================
"""

    def __init__(self, initial_args=None):
        self.params = {
            "data_dir": "./data",
            "persist_dir": "./chroma_db",
            "chunk_mode": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": 4,
            "temperature": 0.0,
            "max_context_chars": 6000,
            "cache": "on",
        }
        if initial_args:
            for key in self.params:
                if hasattr(initial_args, key) and getattr(initial_args, key) is not None:
                    self.params[key] = getattr(initial_args, key)

        self.retriever = None
        self.chain = None
        self.running = True

    def init_environment(self):
        """Pre-initializes environment and cached state."""
        from local.config import setup_environment
        from local.chain import enable_llm_cache

        setup_environment()
        if self.params["cache"] == "on":
            enable_llm_cache(persist=True)

        # Attempt to load existing index
        self.retriever, _ = load_existing_retriever(
            persist_directory=self.params["persist_dir"],
            top_k=self.params["top_k"]
        )
        if self.retriever:
            self._update_chain()

    def _update_chain(self):
        """Reconstructs RAG chain with current parameters."""
        if self.retriever is None:
            return
        from local.chain import build_rag_chain
        self.chain = build_rag_chain(
            self.retriever,
            temperature=float(self.params["temperature"]),
            max_context_chars=int(self.params["max_context_chars"])
        )

    def execute_command(self, cmd_line: str):
        """Parses and executes a single shell command string."""
        line = cmd_line.strip()
        if not line or line.startswith("#"):
            return  # Comment or blank line

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"Syntax Error: {e}")
            return

        cmd = parts[0].lower()
        args = parts[1:]

        handler_name = f"cmd_{cmd}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(args)
        else:
            print(f"Unknown command '{cmd}'. Type 'help' to see valid commands.")

    # --- Shell Commands ---

    def cmd_help(self, args):
        """Displays available commands or detailed help for a specific command."""
        commands_doc = {
            "help": "Display command list or detailed syntax: help [cmd]",
            "status": "Report full status of documents, vector DB, cache, and settings",
            "read_docs": "Scan and report all documents present in data directory",
            "build_index": "Compile documents into Chroma vector store and BM25 index",
            "query": "Execute question against hybrid knowledge base: query <question>",
            "chat": "Enter interactive conversation loop without typing 'query'",
            "set": "Modify runtime parameter: set <param> <value>",
            "get": "View current value of parameter(s): get [param]",
            "gui": "Launch Streamlit Web UI in browser: gui",
            "clear_cache": "Clear the SQLite query cache database (llm_cache.db)",
            "source": "Execute a batch script of commands: source <file.rag>",
            "exit": "Exit the shell session (alias: quit)",
        }

        if args:
            target = args[0].lower()
            if target in commands_doc:
                print(f"\nCommand: {target}")
                print(f"   {commands_doc[target]}\n")
            else:
                print(f"No help topic found for '{target}'.")
            return

        print("\n" + "=" * 65)
        print("=== AVAILABLE SHELL COMMANDS ===")
        print("=" * 65)
        for cmd_name, desc in sorted(commands_doc.items()):
            print(f"  {cmd_name:<14} : {desc}")
        print("\nTip: Run 'set' to see customizable parameters or 'status' for system telemetry.\n")

    def cmd_status(self, args):
        """Reports system telemetry, database status, and active parameters."""
        data_path = Path(self.params["data_dir"])
        db_path = Path(self.params["persist_dir"])
        cache_file = get_chunks_cache_path(self.params["persist_dir"])
        llm_db = Path("./llm_cache.db")

        print("\n" + "=" * 65)
        print("=== SYSTEM TELEMETRY & KNOWLEDGE BASE STATUS ===")
        print("=" * 65)
        
        # Files status
        if data_path.exists():
            files = [f for f in data_path.rglob("*") if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".csv"]]
            print(f"  • Source Data Dir   : {data_path.resolve()} ({len(files)} document files)")
        else:
            print(f"  • Source Data Dir   : {data_path.resolve()} (NOT FOUND)")

        # Chroma status
        if db_path.exists():
            db_size_mb = sum(f.stat().st_size for f in db_path.glob("**/*") if f.is_file()) / (1024 * 1024)
            print(f"  • Vector Database   : Chroma DB at '{db_path}' ({db_size_mb:.2f} MB)")
        else:
            print(f"  • Vector Database   : NOT BUILT")

        # BM25 Cache status
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                chunks = pickle.load(f)
            print(f"  • BM25 Sparse Index : Active ({len(chunks)} chunks cached)")
        else:
            print(f"  • BM25 Sparse Index : NOT BUILT")

        # LLM Cache
        if llm_db.exists():
            llm_size_kb = llm_db.stat().st_size / 1024
            print(f"  • LLM SQLite Cache  : Active ({llm_size_kb:.1f} KB at '{llm_db}')")
        else:
            print(f"  • LLM SQLite Cache  : Empty / Inactive")

        # Active parameters
        print("\n--- CURRENT RUNTIME PARAMETERS ---")
        for k, v in self.params.items():
            print(f"  {k:<18}: {v}")
        print("=" * 65 + "\n")

    cmd_report_status = cmd_status

    def cmd_read_docs(self, args):
        """Inspects all supported files in the data directory without building."""
        target_dir = args[0] if args else self.params["data_dir"]
        p = Path(target_dir)
        if not p.exists():
            print(f"[ERROR] Directory '{target_dir}' does not exist.")
            return

        files = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".csv"]]
        print(f"\nScanning '{target_dir}'... Found {len(files)} supported document(s):")
        for idx, f in enumerate(files, 1):
            size_kb = f.stat().st_size / 1024
            print(f"  [{idx:02d}] {f.name} ({size_kb:.1f} KB) -> {f.relative_to(p)}")
        print()

    cmd_read_documents = cmd_read_docs

    def cmd_build_index(self, args):
        """Executes knowledge base compilation."""
        mode = self.params["chunk_mode"]
        if "--mode" in args:
            idx = args.index("--mode")
            if idx + 1 < len(args):
                mode = args[idx + 1]

        print(f"\nInitiating compilation on '{self.params['data_dir']}' [Chunker: {mode}]...")
        self.retriever, _, _ = build_knowledge_base(
            data_dir=self.params["data_dir"],
            chunk_mode=mode,
            chunk_size=int(self.params["chunk_size"]),
            chunk_overlap=int(self.params["chunk_overlap"]),
            persist_directory=self.params["persist_dir"],
            top_k=int(self.params["top_k"])
        )
        self._update_chain()

    cmd_compile_db = cmd_build_index

    def cmd_query(self, args):
        """Asks a question using the hybrid retriever."""
        if not args:
            print("Usage: query <your question here>")
            return

        if self.retriever is None or self.chain is None:
            print("Warning: Knowledge base not loaded. Attempting to load from disk...")
            self.retriever, _ = load_existing_retriever(
                persist_directory=self.params["persist_dir"],
                top_k=int(self.params["top_k"])
            )
            if self.retriever is None:
                print("No index found. Run 'build_index' first.")
                return
            self._update_chain()

        question = " ".join(args)
        from local.chain import ask
        ask(self.chain, self.retriever, question)

    cmd_ask = cmd_query

    def cmd_chat(self, args):
        """Enters an uninterrupted conversational prompt loop."""
        if self.retriever is None or self.chain is None:
            print("Knowledge base index not loaded. Run 'build_index' or check 'status' first.")
            return

        print("\n" + "=" * 65)
        print("=== ENTERING CHAT MODE (Type 'exit' or 'quit' to return to sasa) ===")
        print("=" * 65)
        from local.chain import ask
        while True:
            try:
                user_input = input("\nsasa-chat> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Exiting chat mode back to sasa.\n")
                    break
                ask(self.chain, self.retriever, user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat mode.\n")
                break

    def cmd_set(self, args):
        """Modifies a configuration parameter: set <param> <value>"""
        if len(args) < 2:
            print("Usage: set <parameter> <value>")
            print(f"Valid parameters: {', '.join(self.params.keys())}")
            return

        param = args[0].lower()
        val = args[1]

        if param not in self.params:
            print(f"Unknown parameter '{param}'. Valid options: {', '.join(self.params.keys())}")
            return

        if param in ["top_k", "chunk_size", "chunk_overlap", "max_context_chars"]:
            try:
                self.params[param] = int(val)
            except ValueError:
                print(f"Parameter '{param}' must be an integer.")
                return
        elif param == "temperature":
            try:
                self.params[param] = float(val)
            except ValueError:
                print("Parameter 'temperature' must be a float (e.g. 0.0 - 1.0).")
                return
        elif param == "chunk_mode":
            if val not in ["recursive", "semantic"]:
                print("chunk_mode must be 'recursive' or 'semantic'.")
                return
            self.params[param] = val
        elif param == "cache":
            if val.lower() in ["on", "true", "1"]:
                self.params[param] = "on"
                from local.chain import enable_llm_cache
                enable_llm_cache(persist=True)
            elif val.lower() in ["off", "false", "0"]:
                self.params[param] = "off"
            else:
                print("cache must be 'on' or 'off'.")
                return
        else:
            self.params[param] = val

        print(f"Set {param} = {self.params[param]}")

        # Update live chain if retriever exists
        if param in ["top_k", "temperature", "max_context_chars"] and self.retriever:
            self._update_chain()

    def cmd_get(self, args):
        """Displays current parameter value(s): get [param]"""
        if args:
            param = args[0].lower()
            if param in self.params:
                print(f"  {param} = {self.params[param]}")
            else:
                print(f"Unknown parameter '{param}'.")
        else:
            print("\n--- RUNTIME CONFIGURATION ---")
            for k, v in self.params.items():
                print(f"  {k:<18} = {v}")
            print()

    def cmd_gui(self, args):
        """Launches the Streamlit Web UI in the browser."""
        app_path = CURRENT_DIR / "app.py"
        if not app_path.exists():
            print(f"Cannot find '{app_path}'.")
            return

        print("Launching Streamlit Web UI on http://localhost:8501 ...")
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
        try:
            # Spawn in background so shell remains active
            subprocess.Popen(cmd)
            time.sleep(1.5)
            webbrowser.open("http://localhost:8501")
            print("Streamlit server launched. Press Ctrl+C in its console to terminate when finished.\n")
        except Exception as e:
            print(f"Failed to launch GUI: {e}")

    cmd_start_gui = cmd_gui

    def cmd_clear_cache(self, args):
        """Deletes the SQLite query cache."""
        llm_db = Path("./llm_cache.db")
        if llm_db.exists():
            try:
                llm_db.unlink()
                print("Cleared SQLite LLM cache (llm_cache.db removed).")
            except Exception as e:
                print(f"Error clearing cache: {e}")
        else:
            print("No cache file found.")

    def cmd_source(self, args):
        """Executes commands from a script file: source <file.rag>"""
        if not args:
            print("Usage: source <script_file>")
            return

        script_path = Path(args[0])
        if not script_path.exists():
            print(f"Script file '{script_path}' not found.")
            return

        print(f"Sourcing '{script_path}'...")
        with open(script_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip()
                if clean_line and not clean_line.startswith("#"):
                    print(f"sasa [{line_idx}]> {clean_line}")
                    self.execute_command(clean_line)
        print(f"Finished execution of '{script_path}'.\n")

    cmd_run = cmd_source

    def cmd_exit(self, args):
        """Exits the interactive shell."""
        print("Exiting SASA Shell. Goodbye!")
        self.running = False

    cmd_quit = cmd_exit

    def run_repl(self):
        """Runs the interactive Read-Eval-Print Loop."""
        print(self.BANNER)
        self.init_environment()

        while self.running:
            try:
                prompt_line = input("sasa> ").strip()
                if prompt_line:
                    self.execute_command(prompt_line)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session.")
                break

# =============================================================================
# CLI Entrypoint & Argument Dispatcher
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SASA: Silicon & Architecture Search Assistant (Interactive EDA Shell & Web UI)"
    )
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        help="Optional action: 'start_gui' to launch Web UI (default: launches interactive shell)"
    )
    parser.add_argument("--gui", "-gui", action="store_true", help="Launch the SASA Streamlit Web UI")
    parser.add_argument("-f", "--file", type=str, default=None, help="Execute batch script (.rag) and exit")
    parser.add_argument("--data-dir", default="./data", help="Source documents folder (default: ./data)")
    parser.add_argument("--persist-dir", default="./chroma_db", help="Chroma database folder (default: ./chroma_db)")

    args = parser.parse_args()

    # 1. Direct GUI Launch Mode via 'start_gui' positional command or '--gui' flag
    if args.gui or args.action in ["start_gui", "gui"]:
        check_dependencies()
        app_path = CURRENT_DIR / "app.py"
        print("Launching SASA Web UI (Streamlit)...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
        return

    # 2. Check dependencies for Shell
    check_dependencies()

    shell = InteractiveRagShell(initial_args=args)

    # 3. Batch Script File Execution Mode (-f)
    if args.file:
        shell.init_environment()
        shell.execute_command(f"source {args.file}")
        return

    # 4. Default: Run main -> Interactive EDA Shell with its commands
    shell.run_repl()

if __name__ == "__main__":
    main()
