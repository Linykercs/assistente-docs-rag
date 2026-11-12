"""The RAG flow: retrieve → answer from context → attach the cited sources."""

from collections.abc import Sequence
from typing import Protocol

from assistente_docs.models import RagAnswer, RetrievedChunk, Source
from assistente_docs.observability import Tracer
from assistente_docs.rag.llm import Answerer
from assistente_docs.rag.retrieval import Retriever

NO_CONTEXT_ANSWER = "Não encontrei nada sobre isso na documentação que eu consulto."


class AskService(Protocol):
    def ask(self, question: str) -> RagAnswer: ...


class RagService:
    def __init__(self, retriever: Retriever, answerer: Answerer, tracer: Tracer) -> None:
        self._retriever = retriever
        self._answerer = answerer
        self._tracer = tracer

    def ask(self, question: str) -> RagAnswer:
        with self._tracer.span("ask", input=question) as trace:
            retrieved = self._retrieve(question)
            answer = self._answer(question, retrieved)
            trace.update(output=answer.answer, metadata={"status": answer.status})
            return answer

    def _retrieve(self, question: str) -> list[RetrievedChunk]:
        with self._tracer.span("retrieve", input=question) as span:
            retrieved = self._retriever.retrieve(question)
            span.update(
                output=[
                    {"url": item.chunk.source_url, "section": item.chunk.heading, "score": item.score}
                    for item in retrieved
                ]
            )
            return retrieved

    def _answer(self, question: str, retrieved: list[RetrievedChunk]) -> RagAnswer:
        # Without context there is nothing to ground an answer on, so the LLM is not called.
        if not retrieved:
            return RagAnswer(
                status="no_answer", answer=NO_CONTEXT_ANSWER, sources=[], retrieved=[], model=None
            )

        model = self._answerer.model
        with self._tracer.generation("answer", model=model, input=question) as generation:
            completion = self._answerer.answer(question, retrieved)
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
