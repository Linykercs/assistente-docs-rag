"""Aggregates repeated eval runs, so a change is compared against the noise of the provider.

Even at temperature 0 with a fixed seed, hosted models don't always return the same answer:
the same configuration has scored 8/11 and 6/11 on correctness on different days. One run
can't tell a real improvement from that noise; several runs of the same configuration can.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from assistente_docs.evals.report import METRIC_LABELS
from assistente_docs.evals.scoring import METRICS, CaseResult, summarize


@dataclass(frozen=True, slots=True)
class RunsSummary:
    """One metric across runs: the passed/total of each run and the spread of the rates."""

    passed: list[int]
    total: list[int]

    @property
    def rates(self) -> list[float]:
        return [p / t for p, t in zip(self.passed, self.total, strict=True) if t]

    @property
    def mean(self) -> float | None:
        return sum(self.rates) / len(self.rates) if self.rates else None

    @property
    def low(self) -> float | None:
        return min(self.rates) if self.rates else None

    @property
    def high(self) -> float | None:
        return max(self.rates) if self.rates else None


@dataclass(frozen=True, slots=True)
class CaseStability:
    """How often one case passed each metric, out of the runs where the metric applied."""

    case_id: int
    pergunta: str
    passed: dict[str, int]
    applicable: dict[str, int]


def summarize_runs(runs: Sequence[Sequence[CaseResult]]) -> dict[str, RunsSummary]:
    per_run = [summarize(results) for results in runs]
    return {
        metric: RunsSummary(
            passed=[summary[metric].passed for summary in per_run],
            total=[summary[metric].total for summary in per_run],
        )
        for metric in METRICS
    }


def case_stability(runs: Sequence[Sequence[CaseResult]]) -> list[CaseStability]:
    by_case: dict[int, list[CaseResult]] = {}
    for results in runs:
        for result in results:
            by_case.setdefault(result.case_id, []).append(result)

    stability = []
    for case_id in sorted(by_case):
        results = by_case[case_id]
        passed: dict[str, int] = {}
        applicable: dict[str, int] = {}
        for metric in METRICS:
            values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
            passed[metric] = sum(values)
            applicable[metric] = len(values)
        stability.append(CaseStability(case_id, results[0].pergunta, passed, applicable))
    return stability


def write_runs_report(
    runs: Sequence[Sequence[CaseResult]],
    output_dir: Path,
    *,
    generated_at: datetime,
    model: str,
    judge_model: str,
) -> tuple[Path, Path]:
    """Write the aggregate Markdown and JSON reports for repeated runs."""
    summary = summarize_runs(runs)
    stability = case_stability(runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{generated_at:%Y%m%d-%H%M%S}-resumo"

    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text(
        render_runs_markdown(
            summary,
            stability,
            runs=len(runs),
            generated_at=generated_at,
            model=model,
            judge_model=judge_model,
        ),
        encoding="utf-8",
    )

    json_path = output_dir / f"{stem}.json"
    payload = {
        "generated_at": generated_at.isoformat(),
        "model": model,
        "judge_model": judge_model,
        "runs": len(runs),
        "summary": {
            metric: {
                "passed": item.passed,
                "total": item.total,
                "mean": item.mean,
                "min": item.low,
                "max": item.high,
            }
            for metric, item in summary.items()
        },
        "cases": [
            {"case_id": c.case_id, "passed": c.passed, "applicable": c.applicable}
            for c in stability
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path


def render_runs_markdown(
    summary: dict[str, RunsSummary],
    stability: Sequence[CaseStability],
    *,
    runs: int,
    generated_at: datetime,
    model: str,
    judge_model: str,
) -> str:
    lines = [
        f"# Relatório de evals: {runs} execuções",
        "",
        f"- Gerado em: {generated_at:%Y-%m-%d %H:%M:%S %Z}".rstrip(),
        f"- Modelo: `{model}`",
        f"- Juiz: `{judge_model}`",
        "",
        "## Resumo",
        "",
        "| Métrica | Por execução | Média | Mín. | Máx. |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        item = summary[metric]
        each = ", ".join(f"{p}/{t}" for p, t in zip(item.passed, item.total, strict=True))
        lines.append(
            f"| {METRIC_LABELS[metric]} | {each} | {_pct(item.mean)} "
            f"| {_pct(item.low)} | {_pct(item.high)} |"
        )
    lines += [
        "",
        "## Estabilidade por caso",
        "",
        "Quantas execuções passaram em cada métrica. Um caso que oscila entre execuções é "
        "ruído do provedor; um caso que sempre falha é um problema de verdade.",
        "",
        "| # | Pergunta | Comportamento | Context hit | Corretude |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in stability:
        lines.append(
            f"| {case.case_id} | {case.pergunta} | {_ratio(case, 'behavior_ok')} "
            f"| {_ratio(case, 'context_hit')} | {_ratio(case, 'correct')} |"
        )
    return "\n".join(lines) + "\n"


def _ratio(case: CaseStability, metric: str) -> str:
    total = case.applicable[metric]
    return f"{case.passed[metric]}/{total}" if total else "n/a"


def _pct(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"
