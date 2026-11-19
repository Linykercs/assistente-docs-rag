"""The RAG flow: retrieve → answer from context → attach the cited sources."""

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from assistente_docs.models import RagAnswer, RetrievedChunk, Source, Usage
from assistente_docs.observability import Tracer
from assistente_docs.rag.llm import Answerer
from assistente_docs.rag.retrieval import Retriever

NO_CONTEXT_ANSWER = "Não encontrei nada sobre isso na documentação que eu consulto."

logger = logging.getLogger(__name__)


class AskService(Protocol):
    def ask(self, question: str) -> RagAnswer: ...


class RagService:
    def __init__(
        self,
        retriever: Retriever,
        answerer: Answerer,
        tracer: Tracer,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._retriever = retriever
        self._answerer = answerer
        self._tracer = tracer
        self._clock = clock

    def ask(self, question: str) -> RagAnswer:
        with self._tracer.span("ask", input=question) as trace:
            started = self._clock()
            retrieved = self._retrieve(question)
            retrieve_ms = (self._clock() - started) * 1000
            answer = self._answer(question, retrieved, retrieve_ms)
            trace.update(output=answer.answer, metadata={"status": answer.status})
            # One line per question, so cost and latency are visible without Langfuse.
            usage = answer.usage
            logger.info(
                "ask status=%s chunks=%d input_tokens=%d output_tokens=%d "
                "retrieve_ms=%.0f answer_ms=%.0f",
                answer.status,
                len(retrieved),
                usage.input_tokens,
                usage.output_tokens,
                usage.retrieve_ms,
                usage.answer_ms,
            )
            return answer

    def _retrieve(self, question: str) -> list[RetrievedChunk]:
        with self._tracer.span("retrieve", input=question) as span:
            retrieved = self._retriever.retrieve(question)
            span.update(
                output=[
                    {
                        "url": item.chunk.source_url,
                        "section": item.chunk.heading,
                        "score": item.score,
                    }
                    for item in retrieved
                ]
            )
            return retrieved

    def _answer(
        self, question: str, retrieved: list[RetrievedChunk], retrieve_ms: float
    ) -> RagAnswer:
        # Without context there is nothing to ground an answer on, so the LLM is not called.
        if not retrieved:
            return RagAnswer(
                status="no_answer",
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                retrieved=[],
                model=None,
                usage=Usage(retrieve_ms=retrieve_ms),
            )

        model = self._answerer.model
        with self._tracer.generation("answer", model=model, input=question) as generation:
            started = self._clock()
            completion = self._answerer.answer(question, retrieved)
            answer_ms = (self._clock() - started) * 1000
            draft = completion.output
            generation.update(
                output=draft.answer,
                usage_details={
                    "input": completion.input_tokens,
                    "output": completion.output_tokens,
                },
            )
        return RagAnswer(
            status=draft.status,
            answer=draft.answer,
            sources=cited_sources(draft.cited_ids, retrieved),
            retrieved=retrieved,
            model=model,
            usage=Usage(
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                retrieve_ms=retrieve_ms,
                answer_ms=answer_ms,
            ),
        )


def cited_sources(cited_ids: Sequence[int], retrieved: Sequence[RetrievedChunk]) -> list[Source]:
    """Map the ids cited by the model back to retrieved chunks.

    Ids that don't match a retrieved chunk are dropped, so a source in the response always
    points to a real page that was part of the context.
    """
    return [
        Source(
            id=cited_id,
            url=retrieved[cited_id - 1].chunk.source_url,
            title=retrieved[cited_id - 1].chunk.title,
            section=retrieved[cited_id - 1].chunk.heading,
            score=retrieved[cited_id - 1].score,
        )
        for cited_id in dict.fromkeys(cited_ids)
        if 1 <= cited_id <= len(retrieved)
    ]
