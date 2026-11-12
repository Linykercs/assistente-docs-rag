FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv pip install --system --no-cache ".[observability]"

# Bake the embedding model into the image so cold starts don't download it.
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-large
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='${EMBEDDING_MODEL}')"

COPY sources.yaml ./
COPY evals/perguntas.yaml ./evals/perguntas.yaml

RUN useradd --create-home app && chown -R app /app /opt/fastembed
USER app

# Cloud Run injects PORT; 8080 is the default for local runs.
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn assistente_docs.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
