"""Retriever factory: builds different retrieval strategies from a vectorstore."""

from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever

from docuchat.config import Settings
from docuchat.retrieval.reranker import CrossEncoderReranker


def build_retriever(vectorstore: Chroma, settings: Settings) -> BaseRetriever:
    """Build the appropriate retriever based on ``settings.retriever_type``.

    Three strategies are supported:

    ``"similarity"``
        Standard cosine similarity search. Returns the top ``retriever_k``
        chunks by embedding distance. Fast and straightforward.

    ``"mmr"``
        Maximal Marginal Relevance. Fetches ``fetch_k`` candidates, then
        iteratively selects chunks that are relevant to the query *and*
        diverse relative to already-selected chunks.
        ``mmr_lambda_mult`` controls the trade-off:
        0 = maximise diversity, 1 = maximise relevance.

    ``"rerank"``
        Two-stage retrieve-then-rerank. First fetches ``fetch_k`` candidates
        via similarity search, then scores each (query, chunk) pair with a
        neural cross-encoder (``CrossEncoderReranker``) and returns the top
        ``retriever_k`` by cross-encoder score.

    Args:
        vectorstore: A ``Chroma`` instance from ``build_vectorstore()`` or
            ``load_vectorstore()``.
        settings: Application settings carrying retriever type and parameters.

    Returns:
        A ``BaseRetriever`` ready to be passed to ``build_qa_chain()``.

    Raises:
        ValueError: If ``settings.retriever_type`` is not a known strategy.

    """

    if settings.retriever_type == "similarity":
        return vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})

    elif settings.retriever_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.retriever_k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda_mult,
            },
        )

    elif settings.retriever_type == "rerank":
        base = vectorstore.as_retriever(search_kwargs={"k": settings.fetch_k})

        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            top_n=settings.retriever_k,
        )

        return ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base,
        )

    else:
        raise ValueError(f"Unknown retriever_type: {settings.retriever_type!r}")
