"""POST /chat — accept a question and return an answer."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from docuchat.api.dependencies import get_chain
from docuchat.api.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, chain=Depends(get_chain)) -> ChatResponse:
    """Answer a question using the ingested documents.

    Args:
        request: ChatRequest containing the question string.
        chain: QA chain injected by FastAPI's dependency system.

    Returns:
        ChatResponse with the answer and source page numbers.

    Raises:
        503: If no documents have been ingested yet (raised by get_chain dependency).
        502: If the OpenAI API call fails.
        404: If no relevant chunks were found.
    """
    
    try:
      answer = chain.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail="Failed to get answer from LLM.")
    
    if not answer:
            raise HTTPException(status_code=404, detail="No relevant information found for your question.")

    logger.info("Q: %s | A: %s", request.question, answer[:100])
    return ChatResponse(answer=answer, source_pages=[])
