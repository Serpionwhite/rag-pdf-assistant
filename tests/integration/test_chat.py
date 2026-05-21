"""Integration tests for POST /chat."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from docuchat.api.dependencies import get_chain


@pytest.fixture
def mock_chain(app):
    """Override get_chain dependency with a mock that returns a fake answer.

    app.dependency_overrides replaces a real dependency with a fake one
    for the duration of the test. Cleaned up after each test.
    """
    chain = MagicMock()
    chain.invoke.return_value = "RNNs are used for sequential data."

    app.dependency_overrides[get_chain] = lambda: chain
    yield chain
    app.dependency_overrides.clear()


def test_chat_returns_answer(client: TestClient, mock_chain):
    # TODO: POST {"question": "What are RNNs?"} to /chat
    # assert response status is 200
    # assert "answer" key is in response JSON
    # assert response JSON["answer"] is a non-empty string
    ...


def test_chat_empty_question_rejected(client: TestClient, mock_chain):
    # TODO: POST {"question": ""} to /chat
    # FastAPI validates min_length=1 on the question field
    # assert response status is 422 (Unprocessable Entity)
    ...


def test_chat_no_documents_ingested(client: TestClient, app):
    # TODO: override get_chain to raise HTTPException(503)
    # This simulates the case where no PDF has been ingested yet
    # assert response status is 503
    #
    # Hint:
    #   from fastapi import HTTPException
    #   from docuchat.api.dependencies import get_chain
    #   def raise_503(): raise HTTPException(status_code=503, detail="...")
    #   app.dependency_overrides[get_chain] = raise_503
    ...
