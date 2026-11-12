"""Renders eval results as a Markdown report (for people) and JSON (for tooling)."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from assistente_docs.evals.scoring import METRICS, CaseResult, MetricSummary, summarize

_METRIC_LABELS = {
    "behavior_ok": "Comportamento esperado (respondeu / recusou / pediu contexto)",
    "citation_ok": "Resposta com fonte citada",
    "retrieval_hit": "Fonte esperada recuperada (retrieval hit)",
    "correct": "Resposta correta (LLM-as-judge)",
}


def write_report(
    results: Sequence[CaseResult],
    output_dir: Path,
    *,
    generated_at: datetime,
    model: str,
    judge_model: str,
) -> tuple[Path, Path]:
    """Write the Markdown and JSON reports, returning their paths."""
    summary = summarize(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{generated_at:%Y%m%d-%H%M%S}"

    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(
        render_markdown(
            results, summary, generated_at=generated_at, model=model, judge_model=judge_model
        ),
        encoding="utf-8",
    )

    json_path = output_dir / f"{stem}.json"
    payload = {
        "generated_at": generated_at.isoformat(),
        "model": model,
        "judge_model": judge_model,
        "summary": {
            metric: {"passed": item.passed, "total": item.total, "rate": item.rate}
            for metric, item in summary.items()
        },
        "cases": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path


def render_markdown(
    results: Sequence[CaseResult],
    summary: dict[str, MetricSummary],
    *,
    generated_at: datetime,
    model: str,
    judge_model: str,
) -> str:
    lines = [
        "# Relatório de evals",
        "",
        f"- Gerado em: {generated_at:%Y-%m-%d %H:%M:%S %Z}".rstrip(),
        f"- Modelo: `{model}`",
        f"- Juiz: `{judge_model}`",
        f"- Casos: {len(results)}",
        "",
        "## Resumo",
        "",
        "| Métrica | Passou | Taxa |",
        "| --- | --- | --- |",
        *(
            f"| {_METRIC_LABELS[metric]} | {summary[metric].passed}/{summary[metric].total} "
            f"| {_format_rate(summary[metric].rate)} |"
            for metric in METRICS
        ),
        "",
        "Cada métrica considera só os casos em que ela se aplica: sem `fonte` preenchida não "
        "há retrieval hit, e sem `resposta_esperada` não há avaliação de corretude.",
        "",
        "## Casos",
        "",
        "| # | Categoria | Esperado | Obtido | Comportamento | Citação | Retrieval | Corretude |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(
            f"| {result.case_id} | {result.categoria} | {result.expected_status} "
            f"| {result.status} | {_format_outcome(result.behavior_ok)} "
            f"| {_format_outcome(result.citation_ok)} | {_format_outcome(result.retrieval_hit)} "
            f"| {_format_outcome(result.correct)} |"
            for result in results
        ),
        "",
        "## Detalhes",
    ]
    for result in results:
        lines += ["", f"### {result.case_id}. {result.pergunta}", "", result.answer.strip(), ""]
        lines += [f"- Fonte citada: {url}" for url in result.cited_urls]
        if result.judge_reasoning:
            lines.append(f"- Juiz: {result.judge_reasoning}")
    return "\n".join(lines).rstrip() + "\n"


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def _format_outcome(outcome: bool | None) -> str:
    if outcome is None:
        return "n/a"
    return "ok" if outcome else "falhou"
