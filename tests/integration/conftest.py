"""Shared fixtures for integration tests."""

import pytest
from fastapi.testclient import TestClient

from docuchat.api.app import create_app


@pytest.fixture
def app():
    """Create a fresh app instance for each test."""
    return create_app()


@pytest.fixture
def client(app):
    """Return a TestClient wrapping the app.

    TestClient spins up the full FastAPI app in-process — no real
    network calls, but all routing, validation, and middleware run.
    """
    with TestClient(app) as c:
        yield c
