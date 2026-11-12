"""Domain types shared across ingestion, retrieval, answering and evals."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

AnswerStatus = Literal["answered", "no_answer", "needs_clarification"]


@dataclass(frozen=True, slots=True)
class Document:
    """A documentation page, already converted to clean Markdown."""

    url: str
    title: str
    markdown: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document plus the metadata needed to cite it."""

    source_url: str
    title: str
    heading: str
    content: str

    @property
    def content_hash(self) -> str:
        """Stable identity of the chunk, used for idempotent upserts."""
        payload = "\n".join((self.source_url, self.heading, self.content))
        return sha256(payload.encode()).hexdigest()

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedding model: the heading trail gives short chunks context."""
        trail = " > ".join(part for part in (self.title, self.heading) if part)
        return f"{trail}\n\n{self.content}"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Source:
    """A retrieved chunk the answer actually cites; `id` matches the [n] markers in the text."""

    id: int
    url: str
    title: str
    section: str
    score: float


@dataclass(frozen=True, slots=True)
class RagAnswer:
    status: AnswerStatus
    answer: str
    sources: list[Source]
    retrieved: list[RetrievedChunk]
    model: str | None
