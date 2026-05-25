"""Integration tests for POST /ingest."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_build_vectorstore():
    """Patch build_vectorstore so no real OpenAI calls are made."""
    with patch("docuchat.api.routers.ingest.build_vectorstore") as mock:
        mock.return_value = MagicMock()
        yield mock


def test_ingest_valid_pdf(client: TestClient, tmp_pdf_path, mock_build_vectorstore):
    
    with open(tmp_pdf_path, "rb") as f:
        response = client.post(
            "/ingest", files={"file": ("test.pdf", f, "application/pdf")}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["pages"] > 0
    assert data["chunks"] > 0

    mock_build_vectorstore.assert_called_once()



def test_ingest_non_pdf_rejected(client: TestClient):
    
    response = client.post(
        "/ingest", files={"file": ("test.txt", "Hello Boi !", "application/txt")}
    )
    assert response.status_code == 400


def test_ingest_empty_pdf_rejected(client: TestClient, mock_build_vectorstore):
    
    with patch("docuchat.api.routers.ingest.load_pdf") as mock_load:
          mock_load.return_value = []
          response = client.post(
              "/ingest",
              files={
                  "file": ("test.pdf", BytesIO(b"%PDF-1.4 empty"), "application/pdf")
              },
          )
    assert response.status_code == 400
