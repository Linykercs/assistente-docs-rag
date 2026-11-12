import json

import httpx
import pytest

from assistente_docs.bootstrap import open_llm
from assistente_docs.config import Settings
from assistente_docs.evals.judge import JudgeVerdict, LlmJudge
from assistente_docs.rag.llm import AnthropicLlm, LlmAnswerer, LlmError
from assistente_docs.rag.openai_compat import OpenAICompatLlm, strict_json_schema
from assistente_docs.rag.prompt import AnswerDraft
from fakes import make_retrieved

BASE_URL = "https://models.example.com/inference"
SLEEPS: list[float] = []


@pytest.fixture(autouse=True)
def _clear_sleeps() -> None:
    SLEEPS.clear()


def chat_response(content: str | None, **message: object) -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, **message},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }


def make_llm(
    handler, *, reasoning_effort: str | None = None
) -> tuple[OpenAICompatLlm, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    llm = OpenAICompatLlm(
        client,
        base_url=BASE_URL + "/",
        api_key="token-123",
        model="openai/x",
        reasoning_effort=reasoning_effort,
        sleep=SLEEPS.append,
    )
    return llm, requests


def test_answerer_sends_a_strict_schema_request_and_parses_the_answer() -> None:
    draft = {"status": "answered", "answer": "Use o header access_token [1].", "cited_ids": [1]}
    llm, requests = make_llm(lambda _: httpx.Response(200, json=chat_response(json.dumps(draft))))

    completion = LlmAnswerer(llm).answer("Como me autentico?", [make_retrieved(1)])

    assert completion.output == AnswerDraft(**draft)
    assert (completion.input_tokens, completion.output_tokens) == (120, 30)

    [request] = requests
    assert str(request.url) == BASE_URL + "/chat/completions"
    assert request.headers["Authorization"] == "Bearer token-123"
    payload = json.loads(request.content)
    assert payload["model"] == "openai/x"
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["messages"][1]["content"].endswith("<pergunta>\nComo me autentico?\n</pergunta>")
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "AnswerDraft"
    assert response_format["json_schema"]["strict"] is True


def test_strict_schema_closes_every_object() -> None:
    schema = strict_json_schema(AnswerDraft)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "answer", "cited_ids"}


def test_json_inside_a_code_fence_is_accepted() -> None:
    verdict = {"reasoning": "Mesmos fatos.", "correct": True}
    fenced = f"```json\n{json.dumps(verdict)}\n```"
    llm, _ = make_llm(lambda _: httpx.Response(200, json=chat_response(fenced)))

    result = LlmJudge(llm).grade(question="P?", expected="R.", actual="R.")

    assert result == JudgeVerdict(**verdict)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(429, text="Rate limit exceeded"), "HTTP 429"),
        (httpx.Response(200, json=chat_response("não é JSON")), "does not match AnswerDraft"),
        (httpx.Response(200, json=chat_response(None, refusal="Não posso ajudar.")), "Não posso"),
        (httpx.Response(200, json={"error": "inesperado"}), "Unexpected LLM API response"),
    ],
)
def test_failures_become_llm_errors(response: httpx.Response, message: str) -> None:
    llm, _ = make_llm(lambda _: response)

    with pytest.raises(LlmError, match=message):
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])


def test_rate_limits_are_retried_before_failing() -> None:
    draft = {"status": "answered", "answer": "Ok [1].", "cited_ids": [1]}
    responses = [
        httpx.Response(429, headers={"Retry-After": "7"}, text="quota"),
        httpx.Response(503, text="busy"),
        httpx.Response(200, json=chat_response(json.dumps(draft))),
    ]
    llm, requests = make_llm(lambda _: responses.pop(0))

    completion = LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])

    assert completion.output.answer == "Ok [1]."
    assert len(requests) == 3
    assert SLEEPS == [7.0, 30.0]


def test_rate_limits_give_up_after_the_last_retry() -> None:
    llm, requests = make_llm(lambda _: httpx.Response(429, text="quota"))

    with pytest.raises(LlmError, match="HTTP 429"):
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
    assert len(requests) == 5


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    draft = json.dumps({"status": "answered", "answer": "Ok [1].", "cited_ids": [1]})

    def ok(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(draft))

    for effort in (None, "low"):
        llm, requests = make_llm(ok, reasoning_effort=effort)
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
        assert json.loads(requests[0].content).get("reasoning_effort") == effort


def test_timeouts_are_retried_before_failing() -> None:
    draft = json.dumps({"status": "answered", "answer": "Ok [1].", "cited_ids": [1]})
    attempts = []

    def slow_then_ok(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("demorou", request=request)
        return httpx.Response(200, json=chat_response(draft))

    llm, _ = make_llm(slow_then_ok)

    assert LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)]).output.answer == "Ok [1]."
    assert len(attempts) == 2


def test_timeouts_give_up_after_the_last_retry() -> None:
    def always_slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("demorou", request=request)

    llm, _ = make_llm(always_slow)

    with pytest.raises(LlmError, match="timed out"):
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])


def test_network_errors_become_llm_errors() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sem rede", request=request)

    llm, _ = make_llm(fail)

    with pytest.raises(LlmError, match="sem rede"):
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])


def test_open_llm_picks_the_configured_provider() -> None:
    anthropic_settings = Settings(_env_file=None, anthropic_api_key="sk-test")
    with open_llm(anthropic_settings, model="claude-x") as llm:
        assert isinstance(llm, AnthropicLlm)
        assert llm.model == "claude-x"

    compat_settings = Settings(_env_file=None, llm_provider="openai_compat", llm_api_key="t")
    with open_llm(compat_settings, model="openai/x") as llm:
        assert isinstance(llm, OpenAICompatLlm)
        assert llm.model == "openai/x"


def test_open_llm_requires_a_key_for_the_openai_compatible_provider() -> None:
    settings = Settings(_env_file=None, llm_provider="openai_compat")

    with pytest.raises(RuntimeError, match="LLM_API_KEY"), open_llm(settings, model="openai/x"):
        pass
