import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from assistente_docs.evals.aggregate import write_runs_report
from assistente_docs.evals.cost import render_usage_table, usage_stats
from assistente_docs.evals.report import write_report
from assistente_docs.evals.scoring import CaseResult


def result(case_id: int, input_tokens: int, answer_ms: float) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        categoria="cobrancas",
        pergunta=f"Pergunta {case_id}?",
        expected_status="answered",
        status="answered",
        answer="Resposta [1].",
        cited_urls=[],
        retrieved_urls=[],
        behavior_ok=True,
        citation_ok=True,
        retrieval_hit=None,
        context_hit=None,
        missing_terms=[],
        correct=None,
        judge_reasoning=None,
        input_tokens=input_tokens,
        output_tokens=input_tokens // 10,
        retrieve_ms=40.0,
        answer_ms=answer_ms,
    )


RESULTS = [result(1, 1000, 800.0), result(2, 3000, 1200.0), result(3, 2000, 5000.0)]


def test_stats_cover_mean_median_p95_and_total() -> None:
    stats = usage_stats(RESULTS)["input_tokens"]

    assert stats is not None
    assert (stats.count, stats.total, stats.mean, stats.median, stats.p95) == (
        3,
        6000.0,
        2000.0,
        2000.0,
        3000.0,
    )


def test_p95_is_an_observed_value_so_one_slow_call_shows_up() -> None:
    stats = usage_stats(RESULTS)["answer_ms"]

    assert stats is not None
    assert stats.p95 == 5000.0
    assert stats.median == 1200.0


def test_questions_that_never_reached_the_llm_are_left_out() -> None:
    no_context = result(4, 0, 0.0)

    stats = usage_stats([*RESULTS, no_context])["input_tokens"]

    assert stats is not None
    assert stats.count == 3


def test_reports_without_usage_say_so_instead_of_showing_zeros() -> None:
    old = [replace(item, input_tokens=0, output_tokens=0, answer_ms=0.0) for item in RESULTS]

    lines = render_usage_table(usage_stats(old))

    assert lines == ["Sem dados de custo nesta execução (relatório anterior ao registro de uso)."]


def test_single_run_report_has_a_cost_section(tmp_path: Path) -> None:
    when = datetime(2026, 11, 19, tzinfo=UTC)
    markdown, json_path = write_report(
        RESULTS, tmp_path, generated_at=when, model="m", judge_model="j"
    )

    text = markdown.read_text(encoding="utf-8")
    assert "## Custo e latência" in text
    assert "| Tokens de entrada (prompt + contexto) | 2.000 | 2.000 | 3.000 | 6.000 |" in text
    usage = json.loads(json_path.read_text(encoding="utf-8"))["usage"]
    assert usage["answer_ms"]["p95"] == 5000.0


def test_runs_report_adds_up_every_run(tmp_path: Path) -> None:
    when = datetime(2026, 11, 19, tzinfo=UTC)
    markdown, _ = write_runs_report(
        [RESULTS, RESULTS], tmp_path, generated_at=when, model="m", judge_model="j"
    )

    text = markdown.read_text(encoding="utf-8")
    assert "Calculado sobre as 6 perguntas que chegaram ao LLM." in text
    assert "| 12.000 |" in text
