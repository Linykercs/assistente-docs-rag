"""Ingestion pipeline: fetch → clean → chunk → embed → store."""

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import unquote, urlsplit

import httpx

from assistente_docs.embeddings import Embedder
from assistente_docs.ingestion.chunking import chunk_document
from assistente_docs.ingestion.cleaning import clean_markdown, extract_title, html_to_markdown
from assistente_docs.ingestion.fetcher import PageFetcher
from assistente_docs.models import Chunk, Document

logger = logging.getLogger(__name__)


class ChunkStore(Protocol):
    def existing_hashes(self, source_url: str) -> set[str]: ...

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def delete_stale(self, source_url: str, keep_hashes: Collection[str]) -> int: ...

    def delete_other_sources(self, keep_urls: Collection[str]) -> int: ...


@dataclass(frozen=True, slots=True)
class IngestReport:
    documents: int
    chunks: int
    embedded: int
    deleted: int


def load_document(fetcher: PageFetcher, url: str) -> Document:
    """Fetch a docs page, preferring its Markdown version and falling back to HTML."""
    try:
        page = fetcher.fetch(f"{url}.md")
    except httpx.HTTPStatusError:
        page = fetcher.fetch(url)
    markdown = html_to_markdown(page.body) if page.is_html else clean_markdown(page.body)
    slug = unquote(urlsplit(url).path).rsplit("/", maxsplit=1)[-1]
    return Document(url=url, title=extract_title(markdown, fallback=slug), markdown=markdown)


def ingest(
    urls: Sequence[str],
    *,
    fetcher: PageFetcher,
    embedder: Embedder,
    store: ChunkStore,
    max_chars: int,
    overlap_chars: int,
) -> IngestReport:
    """Bring the store in sync with `urls`.

    Idempotent: only chunks whose content hash is new get embedded, and chunks that no
    longer exist in the docs (or belong to pages removed from `urls`) are deleted.
    """
    total_chunks = embedded = deleted = 0

    for url in urls:
        document = load_document(fetcher, url)
        chunks = chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars)
        known_hashes = store.existing_hashes(url)
        new_chunks = [chunk for chunk in chunks if chunk.content_hash not in known_hashes]
        if new_chunks:
            vectors = embedder.embed_passages([chunk.embedding_text for chunk in new_chunks])
            store.upsert(new_chunks, vectors)
        removed = store.delete_stale(url, [chunk.content_hash for chunk in chunks])

        total_chunks += len(chunks)
        embedded += len(new_chunks)
        deleted += removed
        logger.info(
            "%s: %d chunks (%d new, %d removed)", url, len(chunks), len(new_chunks), removed
        )

    deleted += store.delete_other_sources(urls)
    return IngestReport(
        documents=len(urls), chunks=total_chunks, embedded=embedded, deleted=deleted
    )
