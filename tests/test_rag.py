from assistente_docs.observability import NoopTracer
from assistente_docs.rag.prompt import SYSTEM_PROMPT, AnswerDraft, build_user_message
from assistente_docs.rag.service import NO_CONTEXT_ANSWER, RagService, cited_sources
from fakes import FakeAnswerer, FakeRetriever, make_retrieved


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
