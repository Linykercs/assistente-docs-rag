"""Small-to-big retrieval: search with small chunks, answer with their surroundings.

Small chunks keep the embeddings precise, but some answers don't fit in one: a list of
events split across sections, or a step-by-step where the question matches step 2 and the
answer needs step 1. The best-ranked chunks are expanded to their page neighbors and to the
rest of their top-level section, within a size budget.
"""

from typing import Protocol

from assistente_docs.models import Chunk, RetrievedChunk
from assistente_docs.rag.retrieval import Retriever

_TRAIL_SEPARATOR = " > "


class PageSource(Protocol):
    def page_chunks(self, source_url: str) -> list[Chunk]: ...


def top_section(chunk: Chunk) -> str:
    """First heading of the trail: the section a chunk belongs to at the top level."""
    return chunk.heading.split(_TRAIL_SEPARATOR, maxsplit=1)[0]


def expansion_group(seed: Chunk, page: list[Chunk], *, max_chars: int) -> list[Chunk]:
    """The seed plus its neighbors and section siblings, closest first, within `max_chars`."""
    section = top_section(seed)
    candidates = [
        chunk
        for chunk in page
        if abs(chunk.position - seed.position) <= 1
        or (section and top_section(chunk) == section)
    ]
    # Closest to the seed first, so the budget keeps what is most likely related.
    candidates.sort(key=lambda chunk: (abs(chunk.position - seed.position), chunk.position))
    group: list[Chunk] = []
    size = 0
    for chunk in candidates:
        is_seed = chunk.content_hash == seed.content_hash
        if not is_seed and size + len(chunk.content) > max_chars:
            continue
        group.append(chunk)
        size += len(chunk.content)
    if not any(chunk.content_hash == seed.content_hash for chunk in group):
        group.append(seed)
    return sorted(group, key=lambda chunk: chunk.position)


class ExpandingRetriever:
    def __init__(
        self,
        inner: Retriever,
        pages: PageSource,
        *,
        expand_top: int,
        max_chars: int,
    ) -> None:
        self._inner = inner
        self._pages = pages
        self._expand_top = expand_top
        self._max_chars = max_chars

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        retrieved = self._inner.retrieve(question)
        pages: dict[str, list[Chunk]] = {}

        result: list[RetrievedChunk] = []
        seen: set[str] = set()
        for rank, item in enumerate(retrieved):
            if rank < self._expand_top:
                url = item.chunk.source_url
                if url not in pages:
                    pages[url] = self._pages.page_chunks(url)
                group = expansion_group(item.chunk, pages[url], max_chars=self._max_chars)
            else:
                group = [item.chunk]
            for chunk in group:
                if chunk.content_hash in seen:
                    continue
                seen.add(chunk.content_hash)
                # Expanded chunks borrow the seed's score: they were pulled in by it.
                result.append(RetrievedChunk(chunk=chunk, score=item.score))
        return result
