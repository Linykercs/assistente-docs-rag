from datetime import UTC, datetime
from pathlib import Path

import pytest

from assistente_docs.evals.dataset import EvalCase, load_cases
from assistente_docs.evals.judge import JudgeVerdict
from assistente_docs.evals.report import write_report
from assistente_docs.evals.runner import run_evals
from assistente_docs.evals.scoring import normalize_url, score_case, summarize
from assistente_docs.models import AnswerStatus, RagAnswer, Source
from assistente_docs.rag.llm import LlmError
from fakes import FakeAskService, make_retrieved

DATASET = Path(__file__).parents[1] / "evals" / "perguntas.yaml"


def make_case(
    *,
    categoria: str = "cobrancas",
    resposta_esperada: str = "",
    fonte: str = "",
    tem_resposta: bool = True,
) -> EvalCase:
    return EvalCase(
        id=1,
        categoria=categoria,
        pergunta="Como crio uma cobrança via Pix?",
        resposta_esperada=resposta_esperada,
        fonte=fonte,
        tem_resposta=tem_resposta,
    )


def make_answer(status: AnswerStatus = "answered", *, cited: bool = True) -> RagAnswer:
    retrieved = [make_retrieved(1), make_retrieved(2)]
    sources = [
        Source(id=1, url=retrieved[0].chunk.source_url, title="Página 1", section="", score=0.9)
    ]
    return RagAnswer(
        status=status,
        answer="Resposta do assistente [1].",
        sources=sources if cited else [],
        retrieved=retrieved,
        model="fake-model",
    )


class FakeJudge:
    def __init__(self) -> None:
        self.graded: list[str] = []

    def grade(self, *, question: str, expected: str, actual: str) -> JudgeVerdict:
        self.graded.append(question)
        return JudgeVerdict(reasoning="Mesmos fatos da referência.", correct=True)


def test_dataset_loads_and_derives_the_expected_behavior() -> None:
    cases = {case.id: case for case in load_cases(DATASET)}

    assert len(cases) == 12
    assert cases[1].expected_status == "answered"
    assert cases[11].expected_status == "no_answer"
    assert cases[12].expected_status == "needs_clarification"
    # Behavior-only cases are checked through the status, never by the correctness judge.
    assert not cases[11].has_reference_answer
    assert not cases[12].has_reference_answer


def test_case_without_reference_skips_correctness_and_retrieval_metrics() -> None:
    result = score_case(make_case(), make_answer(), verdict=None)

    assert result.behavior_ok is True
    assert result.citation_ok is True
    assert result.retrieval_hit is None
    assert result.correct is None


def test_answer_without_citation_fails_the_citation_metric() -> None:
    result = score_case(make_case(), make_answer(cited=False), verdict=None)

    assert result.citation_ok is False


@pytest.mark.parametrize(
    ("fonte", "expected_hit"),
    [
        ("https://docs.asaas.com/docs/pagina-2", True),
        ("https://docs.asaas.com/docs/pagina-2.md#secao", True),
        ("https://docs.asaas.com/docs/pagina-9", False),
    ],
)
def test_retrieval_hit_compares_the_expected_source_with_retrieved_pages(
    fonte: str, expected_hit: bool
) -> None:
    result = score_case(make_case(fonte=fonte), make_answer(), verdict=None)

    assert result.retrieval_hit is expected_hit


def test_refusal_is_the_expected_behavior_for_unanswerable_questions() -> None:
    case = make_case(categoria="fora_do_recorte", tem_resposta=False)

    refused = score_case(case, make_answer("no_answer", cited=False), verdict=None)
    answered = score_case(case, make_answer("answered"), verdict=None)

    assert refused.behavior_ok is True
    assert refused.citation_ok is None
    assert answered.behavior_ok is False


def test_ambiguous_questions_expect_a_clarification_request() -> None:
    case = make_case(categoria="ambigua")

    asked = score_case(case, make_answer("needs_clarification", cited=False), verdict=None)
    guessed = score_case(case, make_answer("answered"), verdict=None)

    assert asked.behavior_ok is True
    assert guessed.behavior_ok is False


def test_judge_verdict_becomes_the_correctness_metric() -> None:
    verdict = JudgeVerdict(reasoning="Faltou o header.", correct=False)

    result = score_case(make_case(resposta_esperada="Use access_token."), make_answer(), verdict)

    assert result.correct is False
    assert result.judge_reasoning == "Faltou o header."


def test_normalize_url_ignores_encoding_suffix_and_trailing_slash() -> None:
    encoded = "https://docs.asaas.com/docs/autentica%C3%A7%C3%A3o-1.md"

    assert normalize_url(encoded) == normalize_url("http://docs.asaas.com/docs/autenticação-1/")


def test_summarize_only_counts_cases_where_the_metric_applies() -> None:
    results = [
        score_case(make_case(), make_answer(), verdict=None),
        score_case(
            make_case(resposta_esperada="Use access_token.", fonte="https://x.test/outra"),
            make_answer(),
            JudgeVerdict(reasoning="Ok.", correct=True),
        ),
    ]

    summary = summarize(results)

    assert (summary["behavior_ok"].passed, summary["behavior_ok"].total) == (2, 2)
    assert (summary["retrieval_hit"].passed, summary["retrieval_hit"].total) == (0, 1)
    assert (summary["correct"].passed, summary["correct"].total) == (1, 1)
    assert summary["correct"].rate == 1.0
    assert summarize([])["correct"].rate is None


def test_runner_only_calls_the_judge_for_cases_with_a_reference_answer() -> None:
    cases = [
        make_case(),
        make_case(resposta_esperada="Use access_token."),
        make_case(
            categoria="fora_do_recorte", resposta_esperada="Deve recusar.", tem_resposta=False
        ),
    ]
    service = FakeAskService(make_answer())
    judge = FakeJudge()

    results = run_evals(cases, service, judge)

    assert len(service.questions) == 3
    assert len(judge.graded) == 1
    assert [result.correct for result in results] == [None, True, None]


def test_run_evals_skips_cases_whose_llm_call_fails() -> None:
    service = FakeAskService(LlmError("timeout"))

    results = run_evals([make_case(), make_case()], service, FakeJudge())

    assert results == []
    assert len(service.questions) == 2


def test_report_is_written_as_markdown_and_json(tmp_path: Path) -> None:
    results = [score_case(make_case(), make_answer(), verdict=None)]

    markdown_path, json_path = write_report(
        results,
        tmp_path,
        generated_at=datetime(2026, 1, 15, 12, 30, tzinfo=UTC),
        model="fake-model",
        judge_model="fake-judge",
    )

    assert markdown_path.name == "eval-20260115-123000.md"
    assert json_path.name == "eval-20260115-123000.json"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "| Resposta com fonte citada | 1/1 | 100% |" in markdown
    assert "| Resposta correta (LLM-as-judge) | 0/0 | n/a |" in markdown
    assert "| 1 | cobrancas | answered | answered | ok | ok | n/a | n/a |" in markdown
    assert '"judge_model": "fake-judge"' in json_path.read_text(encoding="utf-8")
