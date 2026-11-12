"""Structured completions over any OpenAI-compatible chat completions API.

Used with Google Gemini's OpenAI-compatible endpoint, which has a free, rate-limited tier.
Other compatible providers (Groq, OpenRouter, a local Ollama) work the same way.
"""

import copy
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from assistente_docs.rag.llm import Completion, LlmError

# Free tiers limit requests per minute; these statuses mean "try again shortly".
RETRYABLE_STATUS = frozenset({429, 500, 502, 503})
MAX_RETRIES = 4
DEFAULT_RETRY_SECONDS = 15.0
MAX_RETRY_SECONDS = 60.0


def strict_json_schema(output_format: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for strict structured outputs: every object closed to extra properties."""
    schema = copy.deepcopy(output_format.model_json_schema())

    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for item in node:
                close(item)

    close(schema)
    return schema


def _strip_code_fence(text: str) -> str:
    """Some models wrap JSON in a Markdown code fence even when asked for raw JSON."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```")
    return stripped.strip()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Honor Retry-After when given in seconds; otherwise back off linearly."""
    try:
        delay = float(response.headers.get("Retry-After", ""))
    except ValueError:
        delay = DEFAULT_RETRY_SECONDS * (attempt + 1)
    return min(max(delay, 0.0), MAX_RETRY_SECONDS)


class OpenAICompatLlm:
    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str,
        api_key: str,
        model: str,
        reasoning_effort: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    def complete[T: BaseModel](
        self, *, system: str, user: str, output_format: type[T]
    ) -> Completion[T]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_format.__name__,
                    "schema": strict_json_schema(output_format),
                    "strict": True,
                },
            },
        }
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        response = self._post(payload)
        if response.status_code != httpx.codes.OK:
            raise LlmError(f"LLM API returned HTTP {response.status_code}: {response.text[:500]}")

        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LlmError(f"Unexpected LLM API response: {response.text[:500]}") from exc

        content = message.get("content")
        if not content:
            reason = message.get("refusal") or choice.get("finish_reason")
            raise LlmError(f"LLM returned no content ({reason})")
        try:
            output = output_format.model_validate_json(_strip_code_fence(content))
        except ValidationError as exc:
            raise LlmError(f"LLM output does not match {output_format.__name__}: {exc}") from exc

        usage = body.get("usage") or {}
        return Completion(
            output=output,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.post(self._url, json=payload, headers=self._headers)
            except httpx.TimeoutException as exc:
                # Overloaded free tiers sometimes hang instead of answering 503.
                if attempt == MAX_RETRIES:
                    raise LlmError(f"LLM request timed out: {exc}") from exc
                self._sleep(DEFAULT_RETRY_SECONDS * (attempt + 1))
                continue
            except httpx.HTTPError as exc:
                raise LlmError(f"LLM request failed: {exc}") from exc
            if response.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                return response
            self._sleep(_retry_delay(response, attempt))
        raise AssertionError("unreachable")
