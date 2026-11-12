from fastapi.testclient import TestClient

from assistente_docs.api.app import create_app
from assistente_docs.models import RagAnswer, Source
from assistente_docs.rag.llm import LlmError
from fakes import FakeAskService, make_retrieved

ANSWER = RagAnswer(
    status="answered",
    answer="Envie a chave no header access_token [1].",
    sources=[
        Source(
            id=1,
            url="https://docs.asaas.com/docs/autenticação-1",
            title="Autenticação",
            section="Configure os headers",
            score=0.87,
        )
    ],
    retrieved=[make_retrieved(1)],
    model="fake-model",
)


def test_health() -> None:
    with TestClient(create_app(FakeAskService(ANSWER))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_answer_and_sources() -> None:
    service = FakeAskService(ANSWER)
    with TestClient(create_app(service)) as client:
        response = client.post("/ask", json={"question": "Como me autentico na API?"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "answered",
        "answer": "Envie a chave no header access_token [1].",
        "sources": [
            {
                "id": 1,
                "url": "https://docs.asaas.com/docs/autenticação-1",
                "title": "Autenticação",
                "section": "Configure os headers",
                "score": 0.87,
            }
        ],
        "model": "fake-model",
    }
    assert service.questions == ["Como me autentico na API?"]


def test_ask_rejects_an_empty_question() -> None:
    service = FakeAskService(ANSWER)
    with TestClient(create_app(service)) as client:
        response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
    assert service.questions == []


def test_ask_maps_llm_failures_to_bad_gateway() -> None:
    service = FakeAskService(LlmError("Claude API request failed"))
    with TestClient(create_app(service)) as client:
        response = client.post("/ask", json={"question": "Como crio uma cobrança?"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Claude API request failed"}
