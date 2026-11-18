"""Application settings, loaded from environment variables (and `.env` in local dev)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
LlmProvider = Literal["anthropic", "openai_compat"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/assistente_docs"

    # "anthropic" uses Claude. "openai_compat" uses any OpenAI-compatible API, such as
    # Google Gemini or Groq, which have free tiers.
    llm_provider: LlmProvider = "anthropic"
    anthropic_api_key: SecretStr | None = None
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    llm_api_key: SecretStr | None = None
    llm_model: str = "claude-sonnet-5-5"
    # Only used by the OpenAI-compatible provider, when the model supports it ("low", ...).
    llm_reasoning_effort: str | None = None
    # Only used by the OpenAI-compatible provider. 0 makes answers repeatable between eval runs.
    llm_temperature: float | None = None
    # Only used by the OpenAI-compatible provider. Ollama and Groq accept it; Gemini does not.
    llm_seed: int | None = None
    # Only used by the Anthropic provider.
    llm_effort: Effort = "low"
    judge_model: str = "claude-sonnet-5-5"
    # The judge can run on a different provider than the answers, so every model under test is
    # graded by the same judge. Without JUDGE_PROVIDER the judge reuses the answer provider and
    # any judge_* field left unset falls back to the matching llm_* field. With JUDGE_PROVIDER
    # the judge block stands alone: provider-specific llm_* fields are not carried over.
    judge_provider: LlmProvider | None = None
    judge_base_url: str | None = None
    judge_api_key: SecretStr | None = None
    judge_reasoning_effort: str | None = None
    judge_temperature: float | None = None
    judge_seed: int | None = None

    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = Field(default=1024, gt=0)
    # E5 models are trained with these prefixes; set both to "" for models that don't use them.
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # 10, not 6: with 6 the chunk holding the answer was missing in half the eval cases.
    retrieval_top_k: int = Field(default=10, gt=0)
    # "hybrid" adds keyword search (Postgres full-text) to the vector search.
    retrieval_mode: Literal["vector", "hybrid"] = "vector"
    # Candidates each side contributes before the hybrid merge.
    retrieval_candidates: int = Field(default=30, gt=0)
    chunk_max_chars: int = Field(default=1500, gt=0)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    sources_file: Path = Path("sources.yaml")
    raw_cache_dir: Path = Path("data/raw")
    crawl_delay_seconds: float = Field(default=1.0, ge=0)
    crawl_user_agent: str = "assistente-docs-rag/0.1"

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @field_validator(
        "llm_reasoning_effort",
        "llm_temperature",
        "llm_seed",
        "judge_provider",
        "judge_base_url",
        "judge_api_key",
        "judge_reasoning_effort",
        "judge_temperature",
        "judge_seed",
        mode="before",
    )
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        # CI passes these as empty strings when they don't apply to the chosen provider.
        return None if value == "" else value

    def for_judge(self) -> "Settings":
        """Settings whose llm_* fields point at the judge, for building the judge's LLM."""
        judge_fields = {
            "llm_base_url": self.judge_base_url,
            "llm_api_key": self.judge_api_key,
            "llm_reasoning_effort": self.judge_reasoning_effort,
            "llm_temperature": self.judge_temperature,
            "llm_seed": self.judge_seed,
        }
        overrides: dict[str, object] = {"llm_model": self.judge_model}
        if self.judge_provider is None:
            # Same provider as the answers: only what the judge sets explicitly changes.
            overrides |= {key: value for key, value in judge_fields.items() if value is not None}
        else:
            # Another provider: nothing provider-specific leaks over from the answer side.
            overrides |= {"llm_provider": self.judge_provider, **judge_fields}
            if self.judge_base_url is None:
                overrides.pop("llm_base_url")
        return self.model_copy(update=overrides)


@lru_cache
def get_settings() -> Settings:
    return Settings()
