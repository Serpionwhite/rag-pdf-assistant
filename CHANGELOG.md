# Changelog

All notable changes to docuchat are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

## [0.1.0] - 2026-05-20

### Added
- Project scaffold: `src/` layout, `pyproject.toml`, virtual environment setup
- PDF ingestion pipeline: load → split → embed → store (ChromaDB)
- OpenAI embeddings (`text-embedding-3-small`) + GPT-4o-mini for QA
- Layer 1 CLI entry point: `scripts/ingest_and_query.py`
- `.env.example` for API key configuration
