"""Runs the eval dataset through the RAG service and scores every case."""

import logging
from collections.abc import Sequence

from assistente_docs.evals.dataset import EvalCase
from assistente_docs.evals.judge import Judge
from assistente_docs.evals.scoring import CaseResult, score_case
from assistente_docs.rag.service import AskService

logger = logging.getLogger(__name__)


def run_evals(cases: Sequence[EvalCase], service: AskService, judge: Judge) -> list[CaseResult]:
    results = []
    for case in cases:
        answer = service.ask(case.pergunta)
        # Correctness is graded only when a human-written reference answer exists.
        verdict = (
            judge.grade(
                question=case.pergunta, expected=case.resposta_esperada, actual=answer.answer
            )
            if case.has_reference_answer
            else None
        )
        result = score_case(case, answer, verdict)
        logger.info(
            "case %d: expected=%s got=%s", case.id, result.expected_status, result.status
        )
        results.append(result)
    return results
