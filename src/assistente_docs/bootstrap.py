"""Wires the concrete components together from settings."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import anthropic
import httpx

from assistente_docs.config import Settings
from assistente_docs.embeddings import FastEmbedEmbedder
from assistente_docs.observability import build_tracer
from assistente_docs.rag.expansion import ExpandingRetriever
from assistente_docs.rag.llm import AnthropicLlm, LlmAnswerer, StructuredLlm
from assistente_docs.rag.openai_compat import OpenAICompatLlm
from assistente_docs.rag.retrieval import HybridRetriever, Retriever, VectorRetriever
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


# Free tiers can be slow under load; a full answer still fits well within this.
OPENAI_COMPAT_TIMEOUT_SECONDS = 180.0


@contextmanager
def open_llm(settings: Settings, *, model: str) -> Iterator[StructuredLlm]:
    """Build the configured LLM provider, closing its HTTP client on exit."""
    if settings.llm_provider == "anthropic":
        yield AnthropicLlm(
            build_anthropic_client(settings), model=model, effort=settings.llm_effort
        )
        return

    if settings.llm_api_key is None:
        raise RuntimeError(
            "LLM_PROVIDER=openai_compat precisa de LLM_API_KEY (a chave do Groq ou do Gemini)."
        )
    with httpx.Client(timeout=OPENAI_COMPAT_TIMEOUT_SECONDS) as client:
        yield OpenAICompatLlm(
            client,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            model=model,
            reasoning_effort=settings.llm_reasoning_effort,
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
        )


def build_retriever(settings: Settings, store: PgVectorStore) -> Retriever:
    embedder = build_embedder(settings)
    retriever: Retriever
    if settings.retrieval_mode == "hybrid":
        retriever = HybridRetriever(
            embedder,
            store,
            top_k=settings.retrieval_top_k,
            candidates=settings.retrieval_candidates,
        )
    else:
        retriever = VectorRetriever(embedder, store, top_k=settings.retrieval_top_k)
    if settings.context_expand_top:
        retriever = ExpandingRetriever(
            retriever,
            store,
            expand_top=settings.context_expand_top,
            max_chars=settings.context_group_max_chars,
        )
    return retriever


@contextmanager
def open_rag_service(
    settings: Settings, *, wrap_retriever: Callable[[Retriever], Retriever] | None = None
) -> Iterator[RagService]:
    """Build the full RAG service, releasing the DB pool and flushing traces on exit.

    `wrap_retriever` decorates the real retriever; the evals use it for red teaming.
    """
    tracer = build_tracer(settings)
    with build_store(settings) as store, open_llm(settings, model=settings.llm_model) as llm:
        retriever = build_retriever(settings, store)
        if wrap_retriever:
            retriever = wrap_retriever(retriever)
        try:
            yield RagService(retriever, LlmAnswerer(llm), tracer)
        finally:
            tracer.flush()
