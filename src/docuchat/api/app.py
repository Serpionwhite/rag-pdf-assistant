"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from docuchat.config import get_settings
from docuchat.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup and shutdown logic around the app's lifetime.

    Startup: configure logging, validate settings.
    Shutdown: any cleanup (nothing needed in Layer 2).
    """
    load_dotenv()  # populate os.environ before OpenAI clients are created
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("docuchat API starting up")
    logger.info("Using embedding model: %s", settings.openai_embedding_model)
    logger.info("Using chat model: %s", settings.openai_chat_model)
    yield
    logger.info("docuchat API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="docuchat",
        description="RAG-based PDF chatbot API",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Register routers
    from docuchat.api.routers.ingest import router as ingest_router
    from docuchat.api.routers.chat import router as chat_router

    app.include_router(ingest_router)
    app.include_router(chat_router)

    return app


# Module-level app instance for uvicorn
app = create_app()
