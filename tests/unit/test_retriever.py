"""Unit tests for retriever factory (build_retriever)."""

from unittest.mock import MagicMock, patch

import pytest

from docuchat.config import Settings
from docuchat.retrieval.retriever import build_retriever


@pytest.fixture
def mock_vectorstore():
    """A mock Chroma vectorstore whose as_retriever() returns a fake retriever."""
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    return vs


def test_similarity_retriever_calls_as_retriever_with_k(mock_vectorstore):
    """similarity type should call as_retriever with retriever_k."""
    settings = Settings(
        openai_api_key="sk-test", retriever_type="similarity", retriever_k=6
    )
    build_retriever(mock_vectorstore, settings)

    mock_vectorstore.as_retriever.assert_called_once()
    kwargs = mock_vectorstore.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"]["k"] == 6


def test_mmr_retriever_uses_mmr_search_type(mock_vectorstore):
    """mmr type should call as_retriever with search_type='mmr'."""
    settings = Settings(
        openai_api_key="sk-test",
        retriever_type="mmr",
        retriever_k=4,
        fetch_k=20,
        mmr_lambda_mult=0.7,
    )
    build_retriever(mock_vectorstore, settings)

    kwargs = mock_vectorstore.as_retriever.call_args.kwargs
    assert kwargs["search_type"] == "mmr"
    assert kwargs["search_kwargs"]["k"] == 4
    assert kwargs["search_kwargs"]["fetch_k"] == 20
    assert kwargs["search_kwargs"]["lambda_mult"] == 0.7


def test_rerank_retriever_builds_compression_retriever(mock_vectorstore):
    """rerank type should construct a ContextualCompressionRetriever."""
    settings = Settings(openai_api_key="sk-test", retriever_type="rerank")

    # Patch both so Pydantic doesn't reject plain MagicMock objects during validation
    with (
        patch("docuchat.retrieval.retriever.CrossEncoderReranker"),
        patch("docuchat.retrieval.retriever.ContextualCompressionRetriever") as MockCCR,
    ):
        build_retriever(mock_vectorstore, settings)

    MockCCR.assert_called_once()


def test_rerank_base_retriever_uses_fetch_k(mock_vectorstore):
    """rerank base retriever should fetch fetch_k candidates, not retriever_k."""
    settings = Settings(
        openai_api_key="sk-test",
        retriever_type="rerank",
        retriever_k=4,
        fetch_k=20,
    )

    with (
        patch("docuchat.retrieval.retriever.CrossEncoderReranker"),
        patch("docuchat.retrieval.retriever.ContextualCompressionRetriever"),
    ):
        build_retriever(mock_vectorstore, settings)

    kwargs = mock_vectorstore.as_retriever.call_args.kwargs
    assert kwargs["search_kwargs"]["k"] == 20


def test_unknown_retriever_type_raises_value_error(mock_vectorstore):
    """An unrecognised retriever_type should raise ValueError."""
    settings = Settings(openai_api_key="sk-test")
    # Bypass Literal validation by mutating after construction
    object.__setattr__(settings, "retriever_type", "unknown")

    with pytest.raises(ValueError, match="unknown"):
        build_retriever(mock_vectorstore, settings)
