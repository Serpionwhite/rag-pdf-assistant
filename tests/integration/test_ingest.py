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
    # TODO: POST tmp_pdf_path to /ingest as a file upload
    # Hint: open the file in binary mode and pass it as:
    #   files={"file": ("test.pdf", f, "application/pdf")}
    # assert response status is 200
    # assert response JSON has "filename", "pages", "chunks" keys
    # assert mock_build_vectorstore was called once
    ...


def test_ingest_non_pdf_rejected(client: TestClient):
    # TODO: POST a .txt file to /ingest
    # files={"file": ("test.txt", b"some text", "text/plain")}
    # assert response status is 400
    ...


def test_ingest_empty_pdf_rejected(client: TestClient, mock_build_vectorstore):
    # TODO: POST a valid but empty PDF (no extractable text)
    # Use BytesIO(b"%PDF-1.4 empty") as the file content
    # assert response status is 400
    #
    # Hint: you'll also need to patch load_pdf to return []
    # with patch("docuchat.api.routers.ingest.load_pdf") as mock_load:
    #     mock_load.return_value = []
    ...
