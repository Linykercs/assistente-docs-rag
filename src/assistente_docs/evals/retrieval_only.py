"""Retrieval-only evals: does the context carry what the answer needs? No LLM involved.

Retrieval is deterministic and free, so changes that only touch it (embedding model, top_k,
chunking, expansion) can be compared here as often as needed without spending LLM quota.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from assistente_docs.evals.dataset import EvalCase
from assistente_docs.evals.scoring import normalize_url
from assistente_docs.models import RetrievedChunk
from assistente_docs.rag.retrieval import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    case_id: int
    pergunta: str
    retrieval_hit: bool | None
    context_hit: bool | None
    missing_terms: list[str]
    context_chars: int


def score_retrieval(case: EvalCase, retrieved: Sequence[RetrievedChunk]) -> RetrievalResult:
    urls = {normalize_url(item.chunk.source_url) for item in retrieved}
    context = "\n".join(item.chunk.content for item in retrieved)
    folded = context.casefold()
    missing = [term for term in case.trechos if term.casefold() not in folded]
    return RetrievalResult(
        case_id=case.id,
        pergunta=case.pergunta,
        retrieval_hit=normalize_url(case.fonte) in urls if case.fonte.strip() else None,
        context_hit=not missing if case.trechos else None,
        missing_terms=missing,
        context_chars=len(context),
    )


def run_retrieval_evals(cases: Sequence[EvalCase], retriever: Retriever) -> list[RetrievalResult]:
    return [score_retrieval(case, retriever.retrieve(case.pergunta)) for case in cases]


def write_retrieval_report(
    results: Sequence[RetrievalResult],
    output_dir: Path,
    *,
    generated_at: datetime,
    settings_label: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"retrieval-{generated_at:%Y%m%d-%H%M%S}"
    page = [r for r in results if r.retrieval_hit is not None]
    ctx = [r for r in results if r.context_hit is not None]
    page_hits = sum(r.retrieval_hit for r in page)
    ctx_hits = sum(r.context_hit for r in ctx)
    mean_chars = sum(r.context_chars for r in results) / len(results) if results else 0

    lines = [
        "# Relatório de retrieval (sem LLM)",
        "",
        f"- Gerado em: {generated_at:%Y-%m-%d %H:%M:%S %Z}".rstrip(),
        f"- Configuração: {settings_label}",
        f"- Retrieval hit: {page_hits}/{len(page)}",
        f"- Context hit: {ctx_hits}/{len(ctx)}",
        f"- Contexto médio: {mean_chars:,.0f} caracteres".replace(",", "."),
        "",
        "| # | Pergunta | Página | Contexto | Termos fora do contexto |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        missing = ", ".join(f"`{t}`" for t in r.missing_terms)
        lines.append(
            f"| {r.case_id} | {r.pergunta} | {_mark(r.retrieval_hit)} | {_mark(r.context_hit)} "
            f"| {missing} |"
        )
    markdown_path = output_dir / f"{stem}.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = output_dir / f"{stem}.json"
    payload = {
        "generated_at": generated_at.isoformat(),
        "settings": settings_label,
        "retrieval_hit": {"passed": page_hits, "total": len(page)},
        "context_hit": {"passed": ctx_hits, "total": len(ctx)},
        "mean_context_chars": mean_chars,
        "cases": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path, json_path


def _mark(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "ok" if value else "falhou"
