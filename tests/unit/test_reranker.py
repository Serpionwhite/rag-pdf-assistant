"""Unit tests for CrossEncoderReranker."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from docuchat.retrieval.reranker import CrossEncoderReranker


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(page_content="RNNs handle sequential data."),
        Document(page_content="Transformers use self-attention."),
        Document(page_content="CNNs are designed for images."),
    ]


@pytest.fixture
def reranker() -> CrossEncoderReranker:
    """CrossEncoderReranker with a mocked CrossEncoder model.

    Patches sentence_transformers.CrossEncoder so no model is downloaded.
    After instantiation the mock is accessible via reranker._model.
    """
    mock_model = MagicMock()
    with patch("docuchat.retrieval.reranker.CrossEncoder", return_value=mock_model):
        r = CrossEncoderReranker(model_name="test-model", top_n=2)
    return r


def test_compress_documents_returns_top_n(reranker, sample_docs):
    """compress_documents should return exactly top_n docs."""
    reranker._model.predict.return_value = [0.9, 0.3, 0.7]
    result = reranker.compress_documents(sample_docs, query="What are RNNs?")
    assert len(result) == 2


def test_compress_documents_sorted_by_score_descending(reranker, sample_docs):
    """compress_documents should return docs sorted highest score first."""
    # scores: doc0=0.9, doc1=0.3, doc2=0.7  →  top-2 are doc0, doc2
    reranker._model.predict.return_value = [0.9, 0.3, 0.7]
    result = reranker.compress_documents(sample_docs, query="What are RNNs?")
    assert result[0].page_content == "RNNs handle sequential data."
    assert result[1].page_content == "CNNs are designed for images."


def test_compress_documents_fewer_docs_than_top_n(reranker):
    """compress_documents should return all docs when fewer than top_n."""
    docs = [Document(page_content="Only one document.")]
    reranker._model.predict.return_value = [0.8]
    result = reranker.compress_documents(docs, query="question")
    assert len(result) == 1


def test_compress_documents_calls_predict_with_pairs(reranker, sample_docs):
    """compress_documents should call predict with (query, page_content) pairs."""
    reranker._model.predict.return_value = [0.5, 0.6, 0.4]
    query = "What are transformers?"
    reranker.compress_documents(sample_docs, query=query)

    call_args = reranker._model.predict.call_args
    pairs = call_args.args[0]
    assert len(pairs) == len(sample_docs)
    assert all(pair[0] == query for pair in pairs)
    assert pairs[0][1] == sample_docs[0].page_content
