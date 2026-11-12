"""In-memory stand-ins for the components that need network, a database or an API key."""

from collections.abc import Collection, Sequence

from assistente_docs.models import Chunk, RagAnswer, RetrievedChunk
from assistente_docs.rag.llm import Completion
from assistente_docs.rag.prompt import AnswerDraft


def make_retrieved(number: int, score: float = 0.9) -> RetrievedChunk:
    chunk = Chunk(
        source_url=f"https://docs.asaas.com/docs/pagina-{number}",
        title=f"Página {number}",
        heading=f"Seção {number}",
        content=f"Conteúdo do trecho {number}.",
    )
    return RetrievedChunk(chunk=chunk, score=score)


class FakeRetriever:
    def __init__(self, retrieved: list[RetrievedChunk]) -> None:
        self._retrieved = retrieved

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return self._retrieved


class FakeAnswerer:
    model = "fake-model"

    def __init__(self, draft: AnswerDraft) -> None:
        self._draft = draft
        self.calls = 0

    def answer(self, question: str, chunks: Sequence[RetrievedChunk]) -> Completion[AnswerDraft]:
        self.calls += 1
        return Completion(output=self._draft, input_tokens=100, output_tokens=20)


class FakeAskService:
    def __init__(self, answer: RagAnswer | Exception) -> None:
        self._answer = answer
        self.questions: list[str] = []

    def ask(self, question: str) -> RagAnswer:
        self.questions.append(question)
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


class FakeEmbedder:
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class InMemoryChunkStore:
    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}

    def existing_hashes(self, source_url: str) -> set[str]:
        return {
            content_hash
            for content_hash, chunk in self.chunks.items()
            if chunk.source_url == source_url
        }

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, _ in zip(chunks, embeddings, strict=True):
            self.chunks.setdefault(chunk.content_hash, chunk)

    def delete_stale(self, source_url: str, keep_hashes: Collection[str]) -> int:
        stale = self.existing_hashes(source_url) - set(keep_hashes)
        return self._delete(stale)

    def delete_other_sources(self, keep_urls: Collection[str]) -> int:
        other = {
            content_hash
            for content_hash, chunk in self.chunks.items()
            if chunk.source_url not in keep_urls
        }
        return self._delete(other)

    def _delete(self, hashes: set[str]) -> int:
        for content_hash in hashes:
            del self.chunks[content_hash]
        return len(hashes)
