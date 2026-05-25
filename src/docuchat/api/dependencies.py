"""FastAPI dependency injection functions."""

from functools import lru_cache

from fastapi import HTTPException

from docuchat.config import get_settings
from docuchat.ingestion.embedder import load_vectorstore
from docuchat.retrieval.chain import build_qa_chain
from docuchat.retrieval.retriever import build_retriever


@lru_cache
def get_vectorstore():
    """Load the ChromaDB vectorstore from disk.

    Cached so the vectorstore is loaded once and reused across requests.
    Raises HTTP 503 if no vectorstore exists yet (no PDF has been ingested).
    """
    settings = get_settings()
    try:
        return load_vectorstore(settings.chroma_persist_dir)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="No documents ingested yet. POST a PDF to /ingest first.",
        ) from e


@lru_cache
def get_retriever():
    """Build the configured retriever from the persisted vectorstore.

    Retriever strategy is controlled by ``settings.retriever_type``
    ("similarity", "mmr", or "rerank"). Cached so the retriever is
    constructed once and reused across requests.
    """
    vectorstore = get_vectorstore()
    settings = get_settings()
    return build_retriever(vectorstore, settings)


@lru_cache
def get_chain():
    """Build the QA chain from the configured retriever.

    Cached so the chain is built once and reused across requests.
    """
    retriever = get_retriever()
    return build_qa_chain(retriever)
