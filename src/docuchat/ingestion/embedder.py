"""Embedding and vector store: converts chunks to vectors and persists them."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# A stable name for the ChromaDB collection used in Layer 1.
# Changing this invalidates any existing persisted data.
COLLECTION_NAME = "docuchat_layer1"


def build_vectorstore(
    chunks: list[Document],
    persist_dir: str | Path = "chroma_db",
) -> Chroma:
    """Embed document chunks and persist them in ChromaDB.

    Calls the OpenAI Embeddings API for each chunk (batched internally).
    Persists to disk so you don't re-embed on every run.

    Args:
        chunks: Output from ``split_documents()``.
        persist_dir: Directory where ChromaDB stores its data files.
            On first run this directory is created. On subsequent runs,
            the existing collection is overwritten (Layer 1 keeps it simple).

    Returns:
        A ``Chroma`` instance ready to use as a retriever.

    Note:
        Requires ``OPENAI_API_KEY`` to be set in the environment.
    """
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
             documents=chunks,
             embedding=embeddings,
             collection_name=COLLECTION_NAME,
             persist_directory=str(persist_dir),
         )
    
    return vector_store


def load_vectorstore(persist_dir: str | Path = "chroma_db") -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Use this on subsequent runs so you don't re-embed the same PDF.

    Args:
        persist_dir: Same directory passed to ``build_vectorstore()``.

    Returns:
        A ``Chroma`` instance ready to use as a retriever.
    """
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
             collection_name=COLLECTION_NAME,
             persist_directory=str(persist_dir),
             embedding_function=embeddings,
         )
    
    return vector_store
