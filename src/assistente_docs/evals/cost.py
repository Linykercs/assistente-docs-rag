"""Cost and latency of an eval run: tokens and milliseconds per question.

Quality metrics alone hide the price of a change. Raising top_k from 6 to 10 doubled
correctness, but it also sends more context on every call; these numbers show how much.
Pure functions over `CaseResult`, so old reports (with zeros) simply show "n/a".
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from assistente_docs.evals.scoring import CaseResult

USAGE_FIELDS = ("input_tokens", "output_tokens", "retrieve_ms", "answer_ms")

USAGE_LABELS = {
    "input_tokens": "Tokens de entrada (prompt + contexto)",
    "output_tokens": "Tokens de saída",
    "retrieve_ms": "Busca (ms)",
    "answer_ms": "LLM (ms)",
}


@dataclass(frozen=True, slots=True)
class UsageStats:
    """Distribution of one usage field over the cases where the LLM answered."""

    count: int
    total: float
    mean: float
    median: float
    p95: float


def llm_cases(results: Sequence[CaseResult]) -> list[CaseResult]:
    """Cases with a recorded LLM call. Questions without retrieved context never reach the
    LLM, and reports from before usage was recorded have only zeros: both are left out."""
    return [result for result in results if result.input_tokens > 0]


def usage_stats(results: Sequence[CaseResult]) -> dict[str, UsageStats | None]:
    cases = llm_cases(results)
    return {name: _stats([float(getattr(case, name)) for case in cases]) for name in USAGE_FIELDS}


def _stats(values: list[float]) -> UsageStats | None:
    if not values:
        return None
    ordered = sorted(values)
    return UsageStats(
        count=len(ordered),
        total=sum(ordered),
        mean=sum(ordered) / len(ordered),
        median=_percentile(ordered, 0.5),
        p95=_percentile(ordered, 0.95),
    )


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile: always a value that was actually observed."""
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def render_usage_table(stats: dict[str, UsageStats | None]) -> list[str]:
    """Markdown lines for the cost section, or an explanation when nothing was recorded."""
    if all(item is None for item in stats.values()):
        return ["Sem dados de custo nesta execução (relatório anterior ao registro de uso)."]
    count = next(item.count for item in stats.values() if item is not None)
    lines = [
        f"Calculado sobre as {count} perguntas que chegaram ao LLM. O juiz não entra na conta.",
        "",
        "| Medida | Média | Mediana | p95 | Total |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in USAGE_FIELDS:
        item = stats[name]
        if item is None:
            continue
        lines.append(
            f"| {USAGE_LABELS[name]} | {_fmt(item.mean)} | {_fmt(item.median)} "
            f"| {_fmt(item.p95)} | {_fmt(item.total)} |"
        )
    return lines


def _fmt(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")
