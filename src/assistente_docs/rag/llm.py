"""LLM integration: structured completions and the grounded answerer.

`StructuredLlm` is the seam between the RAG/eval code and a concrete provider. Two
implementations exist: `AnthropicLlm` (Claude, the default) and `OpenAICompatLlm`
(any OpenAI-compatible chat completions API, such as GitHub Models), in `openai_compat.py`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import anthropic
from pydantic import BaseModel

from assistente_docs.config import Effort
from assistente_docs.models import RetrievedChunk
from assistente_docs.rag.prompt import SYSTEM_PROMPT, AnswerDraft, build_user_message

# Thinking tokens count toward this limit, so it is well above the size of a typical answer.
MAX_TOKENS = 16_000


class LlmError(RuntimeError):
    """The LLM call failed or returned something unusable."""


@dataclass(frozen=True)
class Completion[T: BaseModel]:
    output: T
    input_tokens: int
    output_tokens: int


class StructuredLlm(Protocol):
    """A single-turn LLM call whose response is validated against a Pydantic model."""

    @property
    def model(self) -> str: ...

    def complete[T: BaseModel](
        self, *, system: str, user: str, output_format: type[T]
    ) -> Completion[T]: ...


class AnthropicLlm:
    """Claude via the official Anthropic SDK, with structured outputs."""

    def __init__(self, client: anthropic.Anthropic, *, model: str, effort: Effort) -> None:
        self._client = client
        self._model = model
        self._effort = effort

    @property
    def model(self) -> str:
        return self._model

    def complete[T: BaseModel](
        self, *, system: str, user: str, output_format: type[T]
    ) -> Completion[T]:
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self._effort},
                output_format=output_format,
            )
        except anthropic.APIError as exc:
            raise LlmError(f"Claude API request failed: {exc}") from exc

        # A refusal or a truncated response (max_tokens) carries no parsed output.
        if response.parsed_output is None:
            raise LlmError(
                f"Claude returned no structured output (stop_reason={response.stop_reason})"
            )
        return Completion(
            output=response.parsed_output,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class Answerer(Protocol):
    @property
    def model(self) -> str: ...

    def answer(
        self, question: str, chunks: Sequence[RetrievedChunk]
    ) -> Completion[AnswerDraft]: ...


class LlmAnswerer:
    """Answers a question using only the retrieved chunks as context."""

    def __init__(self, llm: StructuredLlm) -> None:
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> Completion[AnswerDraft]:
        return self._llm.complete(
            system=SYSTEM_PROMPT,
            user=build_user_message(question, chunks),
            output_format=AnswerDraft,
        )
