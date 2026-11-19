"""Indirect prompt injection for the evals: a poisoned "documentation" chunk in the context.

In a RAG system the dangerous text rarely comes from the user. It comes from the documents:
a page edited by someone else, a scraped comment, a support article. The retrieved chunks go
straight into the prompt, so an instruction hidden in them reaches the model with the same
weight as real documentation.

`InjectingRetriever` wraps the real retriever and, for cases with `documento_injetado`, puts
that text first in the context as if it were a page of the docs. The rest of the pipeline is
the production one: same prompt, same model, same answer parsing.
"""

from assistente_docs.evals.dataset import EvalCase
from assistente_docs.models import Chunk, RetrievedChunk
from assistente_docs.rag.retrieval import Retriever

INJECTED_URL = "https://docs.asaas.com/docs/atualizacoes-importantes"


class InjectingRetriever:
    def __init__(self, base: Retriever) -> None:
        self._base = base
        self._payload = ""

    def prepare(self, case: EvalCase) -> None:
        """Called before each case: arm the attack for this case, or disarm it."""
        self._payload = case.documento_injetado.strip()

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        retrieved = self._base.retrieve(question)
        if not self._payload:
            return retrieved
        poisoned = Chunk(
            source_url=INJECTED_URL,
            title="Atualizações importantes",
            heading="Aviso",
            content=self._payload,
        )
        # Same score as the best real chunk: as relevant-looking as anything retrieved.
        score = retrieved[0].score if retrieved else 1.0
        return [RetrievedChunk(chunk=poisoned, score=score), *retrieved]
