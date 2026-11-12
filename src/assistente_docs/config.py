"""Application settings, loaded from environment variables (and `.env` in local dev)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
LlmProvider = Literal["anthropic", "openai_compat"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/assistente_docs"

    # "anthropic" uses Claude. "openai_compat" uses any OpenAI-compatible API, such as
    # Google Gemini, which has a free tier.
    llm_provider: LlmProvider = "anthropic"
    anthropic_api_key: SecretStr | None = None
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    # Only used by the Anthropic provider.
    llm_effort: Effort = "low"
    judge_model: str = "claude-sonnet-5-5"

    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = Field(default=1024, gt=0)
    # E5 models are trained with these prefixes; set both to "" for models that don't use them.
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    retrieval_top_k: int = Field(default=6, gt=0)
    chunk_max_chars: int = Field(default=1500, gt=0)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    sources_file: Path = Path("sources.yaml")
    raw_cache_dir: Path = Path("data/raw")
    crawl_delay_seconds: float = Field(default=1.0, ge=0)
    crawl_user_agent: str = "assistente-docs-rag/0.1"

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()
