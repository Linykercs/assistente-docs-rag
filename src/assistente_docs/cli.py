"""Command-line entry points: `assistente-docs {init-db,ingest,ask,eval}`."""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

from assistente_docs.bootstrap import (
    build_embedder,
    build_store,
    open_llm,
    open_rag_service,
)
from assistente_docs.config import Settings, get_settings
from assistente_docs.evals.dataset import load_cases
from assistente_docs.evals.judge import LlmJudge
from assistente_docs.evals.report import write_report
from assistente_docs.evals.runner import run_evals
from assistente_docs.evals.scoring import summarize
from assistente_docs.ingestion.fetcher import PageFetcher
from assistente_docs.ingestion.pipeline import ingest
from assistente_docs.ingestion.sources import load_source_urls


def main() -> None:
    parser = argparse.ArgumentParser(prog="assistente-docs", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db", help="cria a extensão pgvector, a tabela e os índices")

    ingest_parser = commands.add_parser("ingest", help="baixa, processa e indexa a documentação")
    ingest_parser.add_argument(
        "--refresh", action="store_true", help="ignora o cache local e baixa as páginas de novo"
    )

    ask_parser = commands.add_parser("ask", help="faz uma pergunta ao assistente")
    ask_parser.add_argument("question")

    eval_parser = commands.add_parser("eval", help="roda a suíte de evals e gera o relatório")
    eval_parser.add_argument("--dataset", type=Path, default=Path("evals/perguntas.yaml"))
    eval_parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()

    match args.command:
        case "init-db":
            _init_db(settings)
        case "ingest":
            _ingest(settings, refresh=args.refresh)
        case "ask":
            _ask(settings, args.question)
        case "eval":
            _eval(settings, dataset=args.dataset, output_dir=args.output_dir)


def _init_db(settings: Settings) -> None:
    with build_store(settings) as store:
        store.apply_schema()
    print("Schema aplicado.")


def _ingest(settings: Settings, *, refresh: bool) -> None:
    urls = load_source_urls(settings.sources_file)
    headers = {"User-Agent": settings.crawl_user_agent}
    with (
        httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client,
        build_store(settings) as store,
    ):
        fetcher = PageFetcher(
            client,
            settings.raw_cache_dir,
            delay_seconds=settings.crawl_delay_seconds,
            refresh=refresh,
        )
        report = ingest(
            urls,
            fetcher=fetcher,
            embedder=build_embedder(settings),
            store=store,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
    print(
        f"{report.documents} páginas, {report.chunks} chunks "
        f"({report.embedded} novos, {report.deleted} removidos)."
    )


def _ask(settings: Settings, question: str) -> None:
    with open_rag_service(settings) as service:
        answer = service.ask(question)
    print(answer.answer)
    for source in answer.sources:
        print(f"[{source.id}] {source.title} - {source.url}")


def _eval(settings: Settings, *, dataset: Path, output_dir: Path) -> None:
    cases = load_cases(dataset)
    with (
        open_llm(settings, model=settings.judge_model) as judge_llm,
        open_rag_service(settings) as service,
    ):
        results = run_evals(cases, service, LlmJudge(judge_llm))

    markdown_path, json_path = write_report(
        results,
        output_dir,
        generated_at=datetime.now(UTC),
        model=settings.llm_model,
        judge_model=settings.judge_model,
    )
    for metric, item in summarize(results).items():
        print(f"{metric}: {item.passed}/{item.total}")
    print(f"Relatório: {markdown_path} e {json_path}")

    skipped = len(cases) - len(results)
    if skipped:
        print(f"{skipped} de {len(cases)} casos pulados por falha do LLM (veja o log).")
    if not results:
        raise SystemExit("Nenhum caso foi avaliado.")


if __name__ == "__main__":
    main()
