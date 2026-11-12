"""Optional tracing with Langfuse.

Tracing turns on only when the Langfuse keys are configured; otherwise a no-op tracer is
used, so the rest of the code never has to check whether observability is enabled.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from assistente_docs.config import Settings


class Observation(Protocol):
    def update(self, **fields: Any) -> Any: ...


class Tracer(Protocol):
    def span(self, name: str, **fields: Any) -> AbstractContextManager[Observation]: ...

    def generation(self, name: str, **fields: Any) -> AbstractContextManager[Observation]: ...

    def flush(self) -> None: ...


class _NoopObservation:
    def update(self, **fields: Any) -> None:
        pass


class NoopTracer:
    @contextmanager
    def span(self, name: str, **fields: Any) -> Iterator[Observation]:
        yield _NoopObservation()

    generation = span

    def flush(self) -> None:
        pass


class LangfuseTracer:
    def __init__(self, *, public_key: str, secret_key: str, host: str) -> None:
        # Imported here because langfuse is an optional dependency (extra "observability").
        from langfuse import Langfuse

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    def span(self, name: str, **fields: Any) -> AbstractContextManager[Observation]:
        return self._client.start_as_current_span(name=name, **fields)

    def generation(self, name: str, **fields: Any) -> AbstractContextManager[Observation]:
        return self._client.start_as_current_generation(name=name, **fields)

    def flush(self) -> None:
        self._client.flush()


def build_tracer(settings: Settings) -> Tracer:
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        return LangfuseTracer(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
    return NoopTracer()
