"""Text splitting: breaks Documents into smaller chunks for embedding."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split a list of Documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter, which tries to split on paragraph
    boundaries first, then sentences, then words — preserving semantic units
    as much as possible.

    Args:
        documents: Output from ``load_pdf()``.
        chunk_size: Maximum number of characters per chunk.
            Larger chunks → more context per retrieval, but higher token cost.
            Smaller chunks → more precise retrieval, but may lose context.
            Start with 1000 and experiment in Layer 3.
        chunk_overlap: Characters shared between consecutive chunks.
            Prevents a relevant sentence from being split across chunk boundaries.
            Typical range: 10–20% of chunk_size.

    Returns:
        List of smaller Document chunks. Each chunk inherits the
        ``metadata`` from its source document (source filename, page number).
    """
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
