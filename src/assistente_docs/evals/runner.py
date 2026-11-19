"""Runs the eval dataset through the RAG service and scores every case."""

import logging
from collections.abc import Callable, Sequence

from assistente_docs.evals.dataset import EvalCase
from assistente_docs.evals.judge import Judge
from assistente_docs.evals.scoring import CaseResult, score_case
from assistente_docs.rag.llm import LlmError, QuotaExhaustedError
from assistente_docs.rag.service import AskService

logger = logging.getLogger(__name__)


def run_evals(
    cases: Sequence[EvalCase],
    service: AskService,
    judge: Judge,
    *,
    before_case: Callable[[EvalCase], None] | None = None,
) -> list[CaseResult]:
    """Score every case. A case whose LLM call fails is logged and skipped, not fatal.

    `before_case` runs before each question; the red-team runs use it to arm the injected
    document of the case.
    """
    results = []
    for case in cases:
        if before_case:
            before_case(case)
        try:
            answer = service.ask(case.pergunta)
            # Correctness is graded only when a human-written reference answer exists.
            verdict = (
                judge.grade(
                    question=case.pergunta, expected=case.resposta_esperada, actual=answer.answer
                )
                if case.has_reference_answer
                else None
            )
        except QuotaExhaustedError as exc:
            # Every remaining case would fail the same way, so stop here.
            logger.error("case %d: %s Parando a suíte.", case.id, exc)
            break
        except LlmError as exc:
            logger.warning("case %d skipped: %s", case.id, exc)
            continue
        result = score_case(case, answer, verdict)
        logger.info("case %d: expected=%s got=%s", case.id, result.expected_status, result.status)
        results.append(result)
    return results
