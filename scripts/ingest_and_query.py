"""Layer 1 entry point: load a PDF, embed it, ask one question.

Run from the repo root (with .venv activated):
    python scripts/ingest_and_query.py

This script is intentionally simple — no CLI arguments, no logging framework,
no error handling beyond what you choose to add. Those come in Layer 2.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these two values before running.
PDF_PATH = Path("data/RNN Notes.pdf")
QUESTION = "What is the main topic of this document?"

# ChromaDB will write its files here. Listed in .gitignore.
PERSIST_DIR = Path("chroma_db")
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    # Load OPENAI_API_KEY (and any other keys) from .env into os.environ.
    # Must happen before importing anything that reads environment variables.
    load_dotenv()

    # Add src/ to the Python path so `import docuchat` works when running
    # the script directly (before or without `pip install -e .`).
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from docuchat.ingestion.embedder import build_vectorstore
    from docuchat.ingestion.loader import load_pdf
    from docuchat.ingestion.splitter import split_documents
    from docuchat.retrieval.chain import build_qa_chain


    # Step 1 — Load the PDF.
    # Call load_pdf(PDF_PATH) → list[Document]
    # Print how many pages were loaded.

    document = load_pdf(PDF_PATH)

    print(f"Loaded {len(document)} pages")

    # Step 2 — Split into chunks.
    # Call split_documents(documents) → list[Document]
    # Print how many chunks were produced.

    chunks = split_documents(document)

    print(f"Split into {len(chunks)} chunks")

    # Step 3 — Embed and store.
    # Call build_vectorstore(chunks, PERSIST_DIR) → Chroma
    # This makes OpenAI API calls — it will take a few seconds.

    vector_store = build_vectorstore(chunks, PERSIST_DIR)

    # Step 4 — Build the QA chain.
    # Call build_qa_chain(vectorstore) → RetrievalQA

    chain = build_qa_chain(vector_store)

    # Step 5 — Ask the question.
    # Call chain.invoke({"query": QUESTION}) → dict
    # The answer is in result["result"].
    # The retrieved chunks are in result["source_documents"].

    answer = chain.invoke(QUESTION)

    # Step 6 — Print the answer.
    # Also consider printing the source document metadata so you can see
    # which pages were used to construct the answer.

    print(answer)


if __name__ == "__main__":
    main()
