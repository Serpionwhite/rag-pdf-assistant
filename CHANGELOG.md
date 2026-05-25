# Changelog

All notable changes to docuchat are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

## [0.3.0] - 2026-05-25

### Added
- Retriever factory (`retrieval/retriever.py`): supports `similarity`, `mmr`, and `rerank` strategies
- Cross-encoder reranker (`retrieval/reranker.py`): two-stage retrieve-then-rerank using `sentence-transformers`
- New config fields: `retriever_type`, `retriever_k`, `fetch_k`, `mmr_lambda_mult`, `reranker_model`
- Unit tests for retriever factory and reranker
- `[retrieval]` optional dependency group: `rank-bm25`, `sentence-transformers`

### Changed
- `build_qa_chain()` now accepts a `BaseRetriever` instead of a `Chroma` vectorstore
- FastAPI dependency graph updated: `get_retriever()` sits between `get_vectorstore()` and `get_chain()`

## [0.2.0] - 2026-05-21

### Added
- FastAPI app with `/ingest` (PDF upload) and `/chat` (QA) endpoints
- Pydantic-settings config with `.env` support
- Structured logging via `logging_config.py`
- Integration tests for `/ingest` and `/chat`
- `[api]` optional dependency group: `fastapi`, `uvicorn`, `pydantic-settings`

## [0.1.0] - 2026-05-20

### Added
- Project scaffold: `src/` layout, `pyproject.toml`, virtual environment setup
- PDF ingestion pipeline: load → split → embed → store (ChromaDB)
- OpenAI embeddings (`text-embedding-3-small`) + GPT-4o-mini for QA
- Layer 1 CLI entry point: `scripts/ingest_and_query.py`
- `.env.example` for API key configuration
