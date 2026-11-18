from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assistente_docs.evals.aggregate import case_stability, summarize_runs, write_runs_report
from assistente_docs.evals.scoring import CaseResult


def result(case_id: int, *, correct: bool | None, behavior_ok: bool = True) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        categoria="cobrancas",
        pergunta=f"Pergunta {case_id}?",
        expected_status="answered",
        status="answered",
        answer="Resposta [1].",
        cited_urls=[],
        retrieved_urls=[],
        behavior_ok=behavior_ok,
        citation_ok=True,
        retrieval_hit=None,
        context_hit=None,
        missing_terms=[],
        correct=correct,
        judge_reasoning=None,
    )


# Mesma configuração, três execuções: o caso 1 sempre acerta, o 2 oscila, o 3 sempre erra.
RUNS = [
    [result(1, correct=True), result(2, correct=True), result(3, correct=False)],
    [result(1, correct=True), result(2, correct=False), result(3, correct=False)],
    [result(1, correct=True), result(2, correct=True), result(3, correct=False)],
]


def test_summary_keeps_each_run_and_the_spread() -> None:
    correct = summarize_runs(RUNS)["correct"]

    assert correct.passed == [2, 1, 2]
    assert correct.total == [3, 3, 3]
    assert correct.mean == pytest.approx(5 / 9)
    assert (correct.low, correct.high) == (pytest.approx(1 / 3), pytest.approx(2 / 3))


def test_metrics_that_never_apply_have_no_rate() -> None:
    context = summarize_runs(RUNS)["context_hit"]

    assert context.total == [0, 0, 0]
    assert context.mean is None


def test_stability_separates_flaky_cases_from_always_failing_ones() -> None:
    by_case = {c.case_id: c for c in case_stability(RUNS)}

    assert by_case[1].passed["correct"] == 3
    assert by_case[2].passed["correct"] == 2
    assert by_case[3].passed["correct"] == 0
    assert by_case[3].applicable["correct"] == 3


def test_a_case_skipped_in_one_run_counts_only_the_runs_where_it_ran() -> None:
    runs = [RUNS[0], [r for r in RUNS[1] if r.case_id != 2]]

    by_case = {c.case_id: c for c in case_stability(runs)}

    assert by_case[2].applicable["correct"] == 1


def test_runs_report_is_written_as_markdown_and_json(tmp_path: Path) -> None:
    runs = [RUNS[0], [replace(r, behavior_ok=False) if r.case_id == 3 else r for r in RUNS[1]]]

    markdown_path, json_path = write_runs_report(
        runs,
        tmp_path,
        generated_at=datetime(2026, 1, 15, 12, 30, tzinfo=UTC),
        model="m",
        judge_model="j",
    )

    assert markdown_path.name == "eval-20260115-123000-resumo.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Relatório de evals: 2 execuções" in markdown
    assert "| Resposta correta (LLM-as-judge) | 2/3, 1/3 | 50% | 33% | 67% |" in markdown
    assert "| 3 | Pergunta 3? | 1/2 | n/a | 0/2 |" in markdown
    assert '"runs": 2' in json_path.read_text(encoding="utf-8")


def test_single_run_reports_can_be_read_back_and_combined(tmp_path: Path) -> None:
    from assistente_docs.evals.aggregate import load_report
    from assistente_docs.evals.report import write_report

    paths = []
    for day, results in enumerate(RUNS, start=1):
        _, json_path = write_report(
            results,
            tmp_path,
            generated_at=datetime(2026, 1, day, 12, 0, tzinfo=UTC),
            model="m",
            judge_model="j",
        )
        paths.append(json_path)

    loaded = [load_report(path) for path in paths]

    assert {(model, judge) for model, judge, _ in loaded} == {("m", "j")}
    assert summarize_runs([results for _, _, results in loaded])["correct"].passed == [2, 1, 2]
