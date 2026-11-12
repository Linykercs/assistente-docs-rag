"""Retrieval: embed the question and fetch the closest chunks."""

from collections.abc import Sequence
from typing import Protocol

from assistente_docs.embeddings import Embedder
from assistente_docs.models import RetrievedChunk


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedChunk]: ...


class VectorIndex(Protocol):
    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]: ...


class VectorRetriever:
    """Top-k cosine similarity search over the vector index."""

    def __init__(self, embedder: Embedder, index: VectorIndex, *, top_k: int) -> None:
        self._embedder = embedder
        self._index = index
        self._top_k = top_k

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self._index.search(self._embedder.embed_query(question), top_k=self._top_k)
