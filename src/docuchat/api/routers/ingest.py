"""POST /ingest — accept a PDF upload and run the ingestion pipeline."""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from docuchat.api.dependencies import get_chain, get_vectorstore
from docuchat.api.models import IngestResponse
from docuchat.config import get_settings
from docuchat.ingestion.embedder import build_vectorstore
from docuchat.ingestion.loader import load_pdf
from docuchat.ingestion.splitter import split_documents

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile) -> IngestResponse:
    """Accept a PDF upload, embed it, and store in ChromaDB.

    Args:
        file: The uploaded PDF file (multipart/form-data).

    Returns:
        IngestResponse with filename, page count, and chunk count.

    Raises:
        400: If the file is not a PDF or has no extractable text.
        500: If embedding or storage fails.
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    settings = get_settings()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        document = load_pdf(tmp_path)
    except Exception as e:
        logger.error("Failed to load PDF: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {e}") from e

    if not len(document) > 0:
        raise HTTPException(status_code=400, detail="PDF has no extractable text.")

    chunks = split_documents(
        document, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    try:
        build_vectorstore(chunks, settings.chroma_persist_dir)
    except Exception as e:
        logger.error("Failed to build vectorstore: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to store embeddings: {e}"
        ) from e
    finally:
        tmp_path.unlink(missing_ok=True)

    get_vectorstore.cache_clear()
    get_chain.cache_clear()

    logger.info(
        "Ingested %s: %d pages, %d chunks", file.filename, len(document), len(chunks)
    )

    return IngestResponse(
        filename=file.filename, pages=len(document), chunks=len(chunks)
    )
