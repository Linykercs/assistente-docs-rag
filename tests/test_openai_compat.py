import json

import httpx
import pytest

from assistente_docs.bootstrap import open_llm
from assistente_docs.config import Settings
from assistente_docs.evals.judge import JudgeVerdict, LlmJudge
from assistente_docs.rag.llm import AnthropicLlm, LlmAnswerer, LlmError, QuotaExhaustedError
from assistente_docs.rag.openai_compat import MAX_RETRIES, OpenAICompatLlm, strict_json_schema
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
    handler,
    *,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
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
        temperature=temperature,
        seed=seed,
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
    assert len(requests) == MAX_RETRIES + 1


@pytest.mark.parametrize(
    "message",
    [
        "Quota exceeded: Generate content requests per day limit exceeded on Free Tier.",
        "Rate limit reached for model on tokens per day (TPD): Limit 200000.",
        "Rate limit reached on requests per day (RPD): Limit 1000, Used 1000.",
    ],
)
def test_daily_quota_is_not_retried(message: str) -> None:
    llm, requests = make_llm(lambda _: httpx.Response(429, text=message))

    with pytest.raises(QuotaExhaustedError, match="Cota diária"):
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
    assert len(requests) == 1
    assert SLEEPS == []


def test_reasoning_effort_is_sent_only_when_configured() -> None:
    draft = json.dumps({"status": "answered", "answer": "Ok [1].", "cited_ids": [1]})

    def ok(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(draft))

    for effort in (None, "low"):
        llm, requests = make_llm(ok, reasoning_effort=effort)
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
        assert json.loads(requests[0].content).get("reasoning_effort") == effort


def test_temperature_and_seed_are_sent_only_when_configured() -> None:
    draft = json.dumps({"status": "answered", "answer": "Ok [1].", "cited_ids": [1]})

    def ok(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(draft))

    llm, requests = make_llm(ok)
    LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
    assert "temperature" not in json.loads(requests[0].content)

    llm, requests = make_llm(ok, temperature=0)
    LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
    payload = json.loads(requests[0].content)
    assert payload["temperature"] == 0
    # Gemini rejects "seed", so temperature alone must not add it.
    assert "seed" not in payload

    llm, requests = make_llm(ok, temperature=0, seed=0)
    LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
    payload = json.loads(requests[0].content)
    assert (payload["temperature"], payload["seed"]) == (0, 0)


def test_empty_settings_from_ci_mean_unset() -> None:
    settings = Settings(_env_file=None, llm_temperature="", llm_reasoning_effort="")

    assert settings.llm_temperature is None
    assert settings.llm_reasoning_effort is None


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


def test_judge_inherits_the_answer_provider_when_not_configured() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compat",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        llm_temperature=0,
        judge_model="qwen-judge",
    )

    judge = settings.for_judge()

    assert judge.llm_provider == "openai_compat"
    assert judge.llm_base_url == "http://localhost:11434/v1"
    assert judge.llm_api_key is not None
    assert judge.llm_api_key.get_secret_value() == "ollama"
    assert judge.llm_temperature == 0
    assert judge.llm_model == "qwen-judge"


def test_judge_can_run_on_another_provider() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key="sk-test",
        judge_provider="openai_compat",
        judge_base_url="https://api.groq.com/openai/v1",
        judge_api_key="gsk-test",
        judge_reasoning_effort="low",
        judge_temperature=0,
        judge_model="openai/gpt-oss-120b",
    )

    judge = settings.for_judge()

    assert judge.llm_provider == "openai_compat"
    assert judge.llm_base_url == "https://api.groq.com/openai/v1"
    assert judge.llm_api_key is not None
    assert judge.llm_api_key.get_secret_value() == "gsk-test"
    assert (judge.llm_reasoning_effort, judge.llm_temperature) == ("low", 0)
    # The answers keep their own provider.
    assert settings.llm_provider == "anthropic"
    with open_llm(judge, model=settings.judge_model) as llm:
        assert isinstance(llm, OpenAICompatLlm)
        assert llm.model == "openai/gpt-oss-120b"


def test_empty_judge_settings_from_ci_mean_unset() -> None:
    settings = Settings(
        _env_file=None,
        judge_provider="",
        judge_base_url="",
        judge_api_key="",
        judge_reasoning_effort="",
        judge_temperature="",
    )

    judge = settings.for_judge()

    assert judge.llm_provider == settings.llm_provider
    assert judge.llm_base_url == settings.llm_base_url


def test_a_judge_on_another_provider_does_not_inherit_the_answer_seed() -> None:
    # Ollama answers with a fixed seed; a Gemini judge would reject it with HTTP 400.
    settings = Settings(
        _env_file=None,
        llm_provider="openai_compat",
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="ollama",
        llm_temperature=0,
        llm_seed=0,
        llm_reasoning_effort="high",
        judge_provider="openai_compat",
        judge_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        judge_api_key="gemini-key",
        judge_temperature=0,
        judge_model="gemini-flash-latest",
    )

    judge = settings.for_judge()

    assert judge.llm_seed is None
    assert judge.llm_reasoning_effort is None
    assert judge.llm_temperature == 0
    assert judge.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_json_generation_failures_are_retried() -> None:
    draft = json.dumps({"status": "answered", "answer": "Ok [1].", "cited_ids": [1]})
    attempts = []

    def broken_then_ok(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {"code": "json_validate_failed", "message": "Failed to generate JSON"}
                },
            )
        return httpx.Response(200, json=chat_response(draft))

    llm, _ = make_llm(broken_then_ok)

    assert LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)]).output.answer == "Ok [1]."
    assert len(attempts) == 2


def test_other_bad_requests_are_not_retried() -> None:
    attempts = []

    def bad(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(400, json={"error": {"code": "invalid_request_error"}})

    llm, _ = make_llm(bad)

    with pytest.raises(LlmError, match="HTTP 400"):
        LlmAnswerer(llm).answer("Pergunta?", [make_retrieved(1)])
    assert len(attempts) == 1
