from assistente_docs.observability import NoopTracer
from assistente_docs.rag.prompt import SYSTEM_PROMPT, AnswerDraft, build_user_message
from assistente_docs.rag.retrieval import HybridRetriever, VectorRetriever
from assistente_docs.rag.service import NO_CONTEXT_ANSWER, RagService, cited_sources
from fakes import FakeAnswerer, FakeEmbedder, FakeRetriever, make_retrieved


def make_service(retrieved: list, draft: AnswerDraft) -> tuple[RagService, FakeAnswerer]:
    answerer = FakeAnswerer(draft)
    return RagService(FakeRetriever(retrieved), answerer, NoopTracer()), answerer


def test_user_message_numbers_documents_and_ends_with_the_question() -> None:
    message = build_user_message("Como crio um webhook?", [make_retrieved(1), make_retrieved(2)])

    assert message.startswith("<documentos>\n")
    assert (
        '<documento id="1" titulo="Página 1" secao="Seção 1" '
        'url="https://docs.asaas.com/docs/pagina-1">\nConteúdo do trecho 1.\n</documento>'
    ) in message
    assert '<documento id="2"' in message
    assert message.endswith("<pergunta>\nComo crio um webhook?\n</pergunta>")


def test_system_prompt_states_the_grounding_rules() -> None:
    for expected in ("única fonte", "português", "no_answer", "needs_clarification", "[2]"):
        assert expected in SYSTEM_PROMPT


def test_ask_returns_the_answer_with_its_cited_sources() -> None:
    retrieved = [make_retrieved(1, score=0.91), make_retrieved(2, score=0.84)]
    draft = AnswerDraft(status="answered", answer="Use o header access_token [2].", cited_ids=[2])
    service, _ = make_service(retrieved, draft)

    answer = service.ask("Como me autentico?")

    assert answer.status == "answered"
    assert answer.answer == "Use o header access_token [2]."
    assert answer.model == "fake-model"
    assert answer.retrieved == retrieved
    [source] = answer.sources
    assert (source.id, source.url, source.title, source.section, source.score) == (
        2,
        "https://docs.asaas.com/docs/pagina-2",
        "Página 2",
        "Seção 2",
        0.84,
    )


def test_ask_does_not_call_the_llm_without_context() -> None:
    draft = AnswerDraft(status="answered", answer="Resposta inventada.", cited_ids=[])
    service, answerer = make_service([], draft)

    answer = service.ask("Como emito nota fiscal?")

    assert answer.status == "no_answer"
    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.sources == []
    assert answerer.calls == 0


def test_cited_sources_drops_unknown_ids_and_duplicates() -> None:
    retrieved = [make_retrieved(1), make_retrieved(2)]

    sources = cited_sources([2, 7, 0, 2, 1], retrieved)

    assert [source.id for source in sources] == [2, 1]


class RecordingIndex:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, embedding, *, top_k):
        self.calls.append({"mode": "vector", "embedding": embedding, "top_k": top_k})
        return [make_retrieved(1)]

    def hybrid_search(self, embedding, text, *, top_k, candidates):
        self.calls.append(
            {
                "mode": "hybrid",
                "embedding": embedding,
                "text": text,
                "top_k": top_k,
                "candidates": candidates,
            }
        )
        return [make_retrieved(2)]


def test_hybrid_retriever_sends_the_question_text_for_keyword_search() -> None:
    index = RecordingIndex()
    retriever = HybridRetriever(FakeEmbedder(), index, top_k=10, candidates=30)

    retrieved = retriever.retrieve("Como uso o pixQrCode?")

    assert retrieved == [make_retrieved(2)]
    assert index.calls == [
        {
            "mode": "hybrid",
            "embedding": [21.0],
            "text": "Como uso o pixQrCode?",
            "top_k": 10,
            "candidates": 30,
        }
    ]


def test_hybrid_retriever_never_asks_for_fewer_candidates_than_it_returns() -> None:
    index = RecordingIndex()
    HybridRetriever(FakeEmbedder(), index, top_k=15, candidates=5).retrieve("pergunta")

    assert index.calls[0]["candidates"] == 15


def test_vector_retriever_ignores_the_keyword_side() -> None:
    index = RecordingIndex()
    VectorRetriever(FakeEmbedder(), index, top_k=6).retrieve("pergunta")

    assert index.calls == [{"mode": "vector", "embedding": [8.0], "top_k": 6}]
