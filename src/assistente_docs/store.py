"""Chunk storage and similarity search on PostgreSQL + pgvector."""

from collections.abc import Collection, Sequence
from importlib import resources
from types import TracebackType
from typing import Self

from psycopg import sql
from psycopg_pool import ConnectionPool

from assistente_docs.models import Chunk, RetrievedChunk

_UPSERT = """
    INSERT INTO chunks (content_hash, source_url, title, heading, content, embedding)
    VALUES (%s, %s, %s, %s, %s, %s::vector)
    ON CONFLICT (content_hash) DO NOTHING
"""

# `<=>` is pgvector's cosine distance; similarity = 1 - distance.
_SEARCH = """
    SELECT source_url, title, heading, content, 1 - (embedding <=> %(query)s::vector) AS score
    FROM chunks
    ORDER BY embedding <=> %(query)s::vector
    LIMIT %(top_k)s
"""


class PgVectorStore:
    def __init__(self, database_url: str, *, embedding_dim: int) -> None:
        self._pool = ConnectionPool(database_url, min_size=1, max_size=4, open=True)
        self._embedding_dim = embedding_dim

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._pool.close()

    def apply_schema(self) -> None:
        """Create the extension, table and indexes if they don't exist yet."""
        schema_file = resources.files("assistente_docs").joinpath("db", "schema.sql")
        schema = schema_file.read_text(encoding="utf-8")
        statement = sql.SQL(schema).format(dim=sql.Literal(self._embedding_dim))
        with self._pool.connection() as conn:
            conn.execute(statement)

    def existing_hashes(self, source_url: str) -> set[str]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT content_hash FROM chunks WHERE source_url = %s", (source_url,)
            ).fetchall()
        return {row[0] for row in rows}

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        rows = [
            (
                chunk.content_hash,
                chunk.source_url,
                chunk.title,
                chunk.heading,
                chunk.content,
                _vector_literal(embedding),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        with self._pool.connection() as conn, conn.cursor() as cursor:
            cursor.executemany(_UPSERT, rows)

    def delete_stale(self, source_url: str, keep_hashes: Collection[str]) -> int:
        """Delete chunks of a page that are no longer part of its current content."""
        with self._pool.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE source_url = %s AND NOT (content_hash = ANY(%s))",
                (source_url, list(keep_hashes)),
            )
            return cursor.rowcount

    def delete_other_sources(self, keep_urls: Collection[str]) -> int:
        """Delete chunks of pages that are no longer in the list of sources."""
        with self._pool.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE NOT (source_url = ANY(%s))", (list(keep_urls),)
            )
            return cursor.rowcount

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]:
        """Return the `top_k` chunks closest to `embedding` by cosine similarity."""
        params = {"query": _vector_literal(embedding), "top_k": top_k}
        with self._pool.connection() as conn:
            rows = conn.execute(_SEARCH, params).fetchall()
        return [
            RetrievedChunk(
                chunk=Chunk(source_url=url, title=title, heading=heading, content=content),
                score=float(score),
            )
            for url, title, heading, content, score in rows
        ]


def _vector_literal(embedding: Sequence[float]) -> str:
    """Format an embedding in pgvector's text input format, e.g. `[0.1,0.2]`."""
    return "[" + ",".join(map(str, embedding)) + "]"
