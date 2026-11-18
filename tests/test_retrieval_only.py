from datetime import UTC, datetime
from pathlib import Path

from assistente_docs.evals.dataset import EvalCase
from assistente_docs.evals.retrieval_only import (
    run_retrieval_evals,
    score_retrieval,
    write_retrieval_report,
)
from fakes import FakeRetriever, make_retrieved


def case(trechos: list[str], fonte: str = "https://docs.asaas.com/docs/pagina-1") -> EvalCase:
    return EvalCase(
        id=7,
        categoria="webhooks",
        pergunta="Quais eventos existem?",
        fonte=fonte,
        trechos=trechos,
        tem_resposta=True,
    )


def test_retrieval_is_scored_without_any_llm() -> None:
    # make_retrieved(1) and (2) come from pagina-1 and pagina-2, with "Conteúdo do trecho N."
    result = score_retrieval(
        case(["trecho 2", "PAYMENT_CREATED"]), [make_retrieved(1), make_retrieved(2)]
    )

    assert result.retrieval_hit is True
    assert result.context_hit is False
    assert result.missing_terms == ["PAYMENT_CREATED"]
    assert result.context_chars == len("Conteúdo do trecho 1.\nConteúdo do trecho 2.")


def test_cases_without_source_or_terms_are_not_applicable() -> None:
    result = score_retrieval(case([], fonte=""), [make_retrieved(1)])

    assert (result.retrieval_hit, result.context_hit) == (None, None)


def test_run_and_report(tmp_path: Path) -> None:
    results = run_retrieval_evals([case(["trecho 1"])], FakeRetriever([make_retrieved(1)]))

    markdown_path, json_path = write_retrieval_report(
        results,
        tmp_path,
        generated_at=datetime(2026, 1, 15, 12, 30, tzinfo=UTC),
        settings_label="x",
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "- Context hit: 1/1" in markdown
    assert "| 7 | Quais eventos existem? | ok | ok |  |" in markdown
    assert '"context_hit": {' in json_path.read_text(encoding="utf-8")
