"""Pydantic request and response schemas for the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(..., min_length=1, description="The question to ask.")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str = Field(..., description="The generated answer.")
    source_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers of the chunks used to generate the answer.",
    )


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    filename: str = Field(..., description="Name of the uploaded file.")
    pages: int = Field(..., description="Number of pages loaded from the PDF.")
    chunks: int = Field(..., description="Number of chunks stored in the vector store.")
