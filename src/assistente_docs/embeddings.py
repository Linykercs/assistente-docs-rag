"""Text embeddings, computed locally with fastembed (ONNX, no API key needed)."""

from collections.abc import Sequence
from typing import Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Wraps a fastembed model, adding the query/passage prefixes asymmetric models expect."""

    def __init__(self, model_name: str, *, query_prefix: str, passage_prefix: str) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([f"{self._passage_prefix}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self._query_prefix}{text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]
