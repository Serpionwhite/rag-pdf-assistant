"""Integration tests for POST /chat."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
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
    
    response = client.post("/chat", json={"question": "What are RNNs?"})
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert len(data['answer']) > 0

def test_chat_empty_question_rejected(client: TestClient, mock_chain):
    
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422



def test_chat_no_documents_ingested(client: TestClient, app):
    def raise_503():
        raise HTTPException(status_code=503, detail="No documents ingested yet.")

    app.dependency_overrides[get_chain] = raise_503

    response = client.post("/chat", json={"question": "What are RNNs?"})
    assert response.status_code == 503

    app.dependency_overrides.clear()
