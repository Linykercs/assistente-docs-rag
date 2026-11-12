"""The eval dataset: questions with the expected answer, source and behavior."""

from pathlib import Path

import yaml
from pydantic import BaseModel, TypeAdapter

from assistente_docs.models import AnswerStatus

AMBIGUOUS_CATEGORY = "ambigua"


class EvalCase(BaseModel):
    id: int
    categoria: str
    pergunta: str
    # Empty while the dataset owner hasn't written the reference answer / source yet.
    resposta_esperada: str = ""
    fonte: str = ""
    tem_resposta: bool

    @property
    def expected_status(self) -> AnswerStatus:
        """How the assistant should behave: answer, admit it doesn't know, or ask back."""
        if not self.tem_resposta:
            return "no_answer"
        if self.categoria == AMBIGUOUS_CATEGORY:
            return "needs_clarification"
        return "answered"

    @property
    def has_reference_answer(self) -> bool:
        """Whether there is a factual reference answer to grade correctness against."""
        return self.expected_status == "answered" and bool(self.resposta_esperada.strip())


def load_cases(path: Path) -> list[EvalCase]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[EvalCase]).validate_python(data)
