"""Command-line entry points: `assistente-docs {init-db,ingest,ask,eval}`."""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

from assistente_docs.bootstrap import (
    build_embedder,
    build_retriever,
    build_store,
    open_llm,
    open_rag_service,
)
from assistente_docs.config import Settings, get_settings
from assistente_docs.evals.aggregate import load_report, summarize_runs, write_runs_report
from assistente_docs.evals.dataset import EvalCase, load_cases
from assistente_docs.evals.judge import LlmJudge
from assistente_docs.evals.report import write_report
from assistente_docs.evals.retrieval_only import run_retrieval_evals, write_retrieval_report
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

    summary_parser = commands.add_parser(
        "resumo", help="junta relatórios JSON de execuções separadas (ex.: um por dia)"
    )
    summary_parser.add_argument(
        "reports", nargs="+", type=Path, help="eval-*.json de cada execução"
    )
    summary_parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))

    eval_parser = commands.add_parser("eval", help="roda a suíte de evals e gera o relatório")
    eval_parser.add_argument("--dataset", type=Path, default=Path("evals/perguntas.yaml"))
    eval_parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    eval_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="roda a suíte N vezes e gera também um resumo com média, mínimo e máximo",
    )
    eval_parser.add_argument(
        "--only-retrieval",
        action="store_true",
        help="mede só o retrieval (retrieval hit e context hit), sem chamar LLM nem juiz",
    )
    eval_parser.add_argument(
        "--cases",
        type=_parse_case_ids,
        default=None,
        help="roda só os casos indicados, por exemplo --cases 7,23,24",
    )

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
        case "resumo":
            _summarize_reports(args.reports, output_dir=args.output_dir)
        case "eval":
            _eval(
                settings,
                dataset=args.dataset,
                output_dir=args.output_dir,
                repeat=args.repeat,
                only_retrieval=args.only_retrieval,
                case_ids=args.cases,
            )


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


def _parse_case_ids(value: str) -> set[int]:
    try:
        return {int(part) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use números separados por vírgula, ex.: 7,23") from exc


def _eval(
    settings: Settings,
    *,
    dataset: Path,
    output_dir: Path,
    repeat: int = 1,
    only_retrieval: bool = False,
    case_ids: set[int] | None = None,
) -> None:
    if repeat < 1:
        raise SystemExit("--repeat precisa ser 1 ou mais.")
    cases = load_cases(dataset)
    if case_ids:
        unknown = case_ids - {case.id for case in cases}
        if unknown:
            raise SystemExit(f"Casos inexistentes no dataset: {sorted(unknown)}")
        cases = [case for case in cases if case.id in case_ids]
    if only_retrieval:
        _eval_retrieval(settings, cases, output_dir)
        return
    runs = []
    with (
        open_llm(settings.for_judge(), model=settings.judge_model) as judge_llm,
        open_rag_service(settings) as service,
    ):
        for run in range(1, repeat + 1):
            if repeat > 1:
                print(f"Execução {run} de {repeat}")
            results = run_evals(cases, service, LlmJudge(judge_llm))
            runs.append(results)
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

    if repeat > 1:
        markdown_path, json_path = write_runs_report(
            runs,
            output_dir,
            generated_at=datetime.now(UTC),
            model=settings.llm_model,
            judge_model=settings.judge_model,
        )
        print(f"Resumo das {repeat} execuções:")
        for metric, item in summarize_runs(runs).items():
            each = ", ".join(f"{p}/{t}" for p, t in zip(item.passed, item.total, strict=True))
            print(f"{metric}: {each}")
        print(f"Resumo: {markdown_path} e {json_path}")
    if not any(runs):
        raise SystemExit("Nenhum caso foi avaliado.")


def _summarize_reports(paths: list[Path], *, output_dir: Path) -> None:
    runs, models, judges = [], set(), set()
    for path in paths:
        model, judge_model, results = load_report(path)
        runs.append(results)
        models.add(model)
        judges.add(judge_model)
    if len(models) > 1 or len(judges) > 1:
        raise SystemExit(
            f"Relatórios de configurações diferentes: modelos {models}, juízes {judges}"
        )
    markdown_path, json_path = write_runs_report(
        runs,
        output_dir,
        generated_at=datetime.now(UTC),
        model=models.pop(),
        judge_model=judges.pop(),
    )
    for metric, item in summarize_runs(runs).items():
        each = ", ".join(f"{p}/{t}" for p, t in zip(item.passed, item.total, strict=True))
        print(f"{metric}: {each}")
    print(f"Resumo: {markdown_path} e {json_path}")


def _eval_retrieval(settings: Settings, cases: list[EvalCase], output_dir: Path) -> None:
    label = (
        f"{settings.embedding_model}, {settings.retrieval_mode}, top_k {settings.retrieval_top_k}, "
        f"expansão {settings.context_expand_top}"
    )
    with build_store(settings) as store:
        results = run_retrieval_evals(cases, build_retriever(settings, store))
    markdown_path, json_path = write_retrieval_report(
        results, output_dir, generated_at=datetime.now(UTC), settings_label=label
    )
    pages = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    contexts = [r.context_hit for r in results if r.context_hit is not None]
    print(f"Configuração: {label}")
    print(f"retrieval_hit: {sum(pages)}/{len(pages)}")
    print(f"context_hit: {sum(contexts)}/{len(contexts)}")
    print(f"Relatório: {markdown_path} e {json_path}")


if __name__ == "__main__":
    main()
