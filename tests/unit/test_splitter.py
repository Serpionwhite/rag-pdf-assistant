from langchain_core.documents import Document

from docuchat.ingestion.splitter import split_documents


def test_split_produces_multiple_chunks():
    # Build a Document inline — no PDF needed
    doc = Document(page_content="word " * 500, metadata={"source": "test.pdf"})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1

def test_chunks_respect_chunk_size():
    doc = Document(page_content="word " * 500, metadata={"source": "test.pdf"})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        assert len(chunk.page_content) <= 100

def test_chunks_inherit_metadata():
    doc = Document(page_content="word " * 500, metadata={"source": "test.pdf"})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"


