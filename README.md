# docuchat

A production-quality RAG PDF chatbot built with LangChain, OpenAI, ChromaDB, and FastAPI.
Built incrementally across 6 layers — from a single-script CLI to a full evaluated API.

## Stack

| Layer | Technology |
|-------|-----------|
| PDF parsing | LangChain + pypdf |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB |
| LLM | OpenAI GPT-4o-mini |
| API (Layer 2+) | FastAPI + uvicorn |

## Setup

### 1. Prerequisites

- Python ≥ 3.11 (`python3 --version`)
- An OpenAI API key

### 2. Clone and create virtual environment

```bash
git clone https://github.com/Serpionwhite/rag-pdf-assistant.git
cd rag-pdf-assistant

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Verify installation

```bash
python -c "import docuchat; print(docuchat.__version__)"
# Expected: 0.1.0
```

## Layer 1 Usage (CLI)

Drop a PDF into the `data/` directory, then edit the constants at the top of `scripts/ingest_and_query.py`:

```python
PDF_PATH = Path("data/your_document.pdf")
QUESTION = "What is the main topic of this document?"
```

Run the pipeline:

```bash
python scripts/ingest_and_query.py
```

## Project Structure

```
src/docuchat/
├── ingestion/     # PDF loading, chunking, embedding
├── retrieval/     # Vector search and QA chain
├── memory/        # Conversation history (Layer 4)
├── api/           # FastAPI app (Layer 2)
└── evaluation/    # RAG evaluation harness (Layer 6)
```

## Development

```bash
# Lint + format
ruff check .
ruff format .

# Type check
mypy src/

# Tests
pytest
```

## Releases

| Version | Layer | Description |
|---------|-------|-------------|
| v0.1.0 | 1 | Single PDF CLI pipeline |
| v0.2.0 | 2 | FastAPI endpoints + error handling |
| v0.3.0 | 3 | Retrieval quality (MMR, reranker) |
| v0.4.0 | 4 | Conversation memory |
| v0.5.0 | 5 | Multi-document support |
| v0.6.0 | 6 | RAG evaluation harness |
