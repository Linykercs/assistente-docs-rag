"""Wires the concrete components together from settings."""

from collections.abc import Iterator
from contextlib import contextmanager

import anthropic

from assistente_docs.config import Settings
from assistente_docs.embeddings import FastEmbedEmbedder
from assistente_docs.observability import build_tracer
from assistente_docs.rag.llm import ClaudeAnswerer
from assistente_docs.rag.retrieval import VectorRetriever
from assistente_docs.rag.service import RagService
from assistente_docs.store import PgVectorStore


def build_store(settings: Settings) -> PgVectorStore:
    return PgVectorStore(settings.database_url, embedding_dim=settings.embedding_dim)


def build_embedder(settings: Settings) -> FastEmbedEmbedder:
    return FastEmbedEmbedder(
        settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
    )


def build_anthropic_client(settings: Settings) -> anthropic.Anthropic:
    # Without an explicit key the SDK resolves credentials from the environment on its own.
    key = settings.anthropic_api_key
    return anthropic.Anthropic(api_key=key.get_secret_value() if key else None)


@contextmanager
def open_rag_service(settings: Settings) -> Iterator[RagService]:
    """Build the full RAG service, releasing the DB pool and flushing traces on exit."""
    tracer = build_tracer(settings)
    with build_store(settings) as store:
        retriever = VectorRetriever(
            build_embedder(settings), store, top_k=settings.retrieval_top_k
        )
        answerer = ClaudeAnswerer(
            build_anthropic_client(settings),
            model=settings.llm_model,
            effort=settings.llm_effort,
        )
        try:
            yield RagService(retriever, answerer, tracer)
        finally:
            tracer.flush()
