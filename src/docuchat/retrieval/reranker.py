"""Cross-encoder reranker: re-scores retrieved docs by query-doc relevance."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder


class CrossEncoderReranker(BaseDocumentCompressor):
    """Re-ranks documents using a sentence-transformers CrossEncoder model.

    A cross-encoder jointly encodes (query, document) pairs and outputs a
    relevance score. Unlike the bi-encoder used for retrieval — which embeds
    query and document separately — the cross-encoder sees both at once and
    produces more accurate relevance judgements at the cost of speed.

    Used as the second stage in "retrieve then rerank":
      1. A base retriever fetches ~fetch_k candidate chunks cheaply.
      2. This reranker scores each chunk against the query and returns top_n.

    Attributes:
        model_name: HuggingFace model ID for a sentence-transformers CrossEncoder.
        top_n: Number of documents to return after reranking.
    """

    model_name: str
    top_n: int

    _model: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Load the CrossEncoder model after Pydantic field initialization.

        Called automatically by Pydantic after __init__. Use this to load
        heavy resources (like ML models) that can't be declared as fields.

        """

        self._model = CrossEncoder(self.model_name)

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
        callbacks: Any = None,
    ) -> list[Document]:
        """Score documents against the query and return the top_n.

        Args:
            documents: Candidate docs from the upstream base retriever.
            query: The user's question string.
            callbacks: LangChain callback handlers — pass through, don't use.

        Returns:
            Top ``self.top_n`` documents sorted by cross-encoder score
            (highest first). Returns all docs if fewer than top_n are passed in.

        """

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        scored = sorted(
            zip(documents, scores, strict=False), key=lambda x: x[1], reverse=True
        )
        return [doc for doc, _ in scored[: self.top_n]]
