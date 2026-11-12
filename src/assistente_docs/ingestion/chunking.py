"""Heading-aware chunking.

A document is split into sections at Markdown headings, so every chunk knows the heading
trail it lives under. Sections longer than `max_chars` are split at paragraph boundaries,
repeating the trailing paragraphs of one chunk at the start of the next (overlap).
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from assistente_docs.models import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_FENCE = "```"
_TRAIL_SEPARATOR = " > "


@dataclass(frozen=True, slots=True)
class _Section:
    heading: str
    body: str


def chunk_document(document: Document, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Split a document into chunks of roughly `max_chars`, preserving source metadata."""
    return [
        Chunk(
            source_url=document.url,
            title=document.title,
            heading=section.heading,
            content=content,
        )
        for section in _split_sections(document)
        for content in _split_body(section.body, max_chars, overlap_chars)
    ]


def _split_sections(document: Document) -> Iterator[_Section]:
    trail: list[tuple[int, str]] = []
    body: list[str] = []
    in_code = False

    def current_section() -> _Section:
        # The page title is stored separately, so it is left out of the heading trail.
        titles = [title for _, title in trail if title != document.title]
        return _Section(heading=_TRAIL_SEPARATOR.join(titles), body="\n".join(body).strip())

    for line in document.markdown.split("\n"):
        if line.strip().startswith(_FENCE):
            in_code = not in_code
        heading = None if in_code else _HEADING.match(line)
        if heading is None:
            body.append(line)
            continue

        section = current_section()
        if section.body:
            yield section
        level = len(heading.group(1))
        while trail and trail[-1][0] >= level:
            trail.pop()
        trail.append((level, heading.group(2)))
        body = []

    section = current_section()
    if section.body:
        yield section


def _split_body(body: str, max_chars: int, overlap_chars: int) -> Iterator[str]:
    current: list[str] = []
    for block in _blocks(body, max_chars):
        if current and len(_join([*current, block])) > max_chars:
            yield _join(current)
            current = _overlap_tail(current, overlap_chars)
        current.append(block)
    if current:
        yield _join(current)


def _blocks(body: str, max_chars: int) -> Iterator[str]:
    """Yield paragraphs, keeping fenced code blocks whole even when they contain blank lines."""
    block: list[str] = []
    in_code = False
    for line in body.split("\n"):
        if line.strip().startswith(_FENCE):
            in_code = not in_code
        if line.strip() or in_code:
            block.append(line)
        elif block:
            yield from _fit("\n".join(block), max_chars)
            block = []
    if block:
        yield from _fit("\n".join(block), max_chars)


def _fit(block: str, max_chars: int) -> Iterator[str]:
    """Split a block that alone exceeds `max_chars` (long table or code sample) at line breaks."""
    if len(block) <= max_chars:
        yield block
        return
    piece = ""
    for line in block.split("\n"):
        candidate = f"{piece}\n{line}" if piece else line
        if piece and len(candidate) > max_chars:
            yield piece
            piece = line
        else:
            piece = candidate
    if piece:
        yield piece


def _overlap_tail(blocks: list[str], overlap_chars: int) -> list[str]:
    """Return the trailing blocks that fit in `overlap_chars`, to open the next chunk."""
    tail: list[str] = []
    size = 0
    for block in reversed(blocks):
        size += len(block)
        if size > overlap_chars:
            break
        tail.insert(0, block)
    return tail


def _join(blocks: list[str]) -> str:
    return "\n\n".join(blocks)
