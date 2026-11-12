"""Scoring of eval cases. Pure functions: no I/O, no LLM calls.

Each metric is `True`/`False`, or `None` when it doesn't apply to the case (for example,
correctness can't be graded while the reference answer is still empty). Cases where a
metric is `None` are left out of that metric's rate.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from assistente_docs.evals.dataset import EvalCase
from assistente_docs.evals.judge import JudgeVerdict
from assistente_docs.models import AnswerStatus, RagAnswer

METRICS = ("behavior_ok", "citation_ok", "retrieval_hit", "correct")


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: int
    categoria: str
    pergunta: str
    expected_status: AnswerStatus
    status: AnswerStatus
    answer: str
    cited_urls: list[str]
    retrieved_urls: list[str]
    # The assistant behaved as expected: answered, refused or asked for clarification.
    behavior_ok: bool
    # An answer was given and it cites at least one retrieved source.
    citation_ok: bool | None
    # The expected source page is among the retrieved chunks.
    retrieval_hit: bool | None
    # The LLM judge considers the answer equivalent to the reference answer.
    correct: bool | None
    judge_reasoning: str | None


@dataclass(frozen=True, slots=True)
class MetricSummary:
    passed: int
    total: int

    @property
    def rate(self) -> float | None:
        return self.passed / self.total if self.total else None


def score_case(case: EvalCase, answer: RagAnswer, verdict: JudgeVerdict | None) -> CaseResult:
    cited_urls = [source.url for source in answer.sources]
    retrieved_urls = list(dict.fromkeys(item.chunk.source_url for item in answer.retrieved))

    citation_ok = bool(cited_urls) if answer.status == "answered" else None
    retrieval_hit = None
    if case.fonte.strip():
        expected_url = normalize_url(case.fonte)
        retrieval_hit = any(normalize_url(url) == expected_url for url in retrieved_urls)

    return CaseResult(
        case_id=case.id,
        categoria=case.categoria,
        pergunta=case.pergunta,
        expected_status=case.expected_status,
        status=answer.status,
        answer=answer.answer,
        cited_urls=cited_urls,
        retrieved_urls=retrieved_urls,
        behavior_ok=answer.status == case.expected_status,
        citation_ok=citation_ok,
        retrieval_hit=retrieval_hit,
        correct=verdict.correct if verdict else None,
        judge_reasoning=verdict.reasoning if verdict else None,
    )


def summarize(results: Sequence[CaseResult]) -> dict[str, MetricSummary]:
    """Aggregate each metric over the cases where it applies."""
    summary: dict[str, MetricSummary] = {}
    for metric in METRICS:
        values = [getattr(result, metric) for result in results]
        applicable = [value for value in values if value is not None]
        summary[metric] = MetricSummary(passed=sum(applicable), total=len(applicable))
    return summary


def normalize_url(url: str) -> str:
    """Make docs URLs comparable: ignore scheme, query, fragment, encoding and `.md` suffix."""
    parts = urlsplit(unquote(url.strip()))
    path = parts.path.rstrip("/").removesuffix(".md")
    return f"{parts.netloc}{path}".lower()
