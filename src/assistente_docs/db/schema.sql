-- Vector store schema. Idempotent: safe to apply on every deploy.
-- {dim} is filled in by PgVectorStore.apply_schema() with the embedding model's dimension.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content_hash TEXT        NOT NULL UNIQUE,
    source_url   TEXT        NOT NULL,
    title        TEXT        NOT NULL,
    heading      TEXT        NOT NULL,
    content      TEXT        NOT NULL,
    embedding    vector({dim}) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_source_url_idx ON chunks (source_url);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
