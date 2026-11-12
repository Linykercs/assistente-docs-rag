"""LLM-as-judge: grades an answer against the human-written reference answer."""

from typing import Protocol

import anthropic
from pydantic import BaseModel, Field

from assistente_docs.config import Effort
from assistente_docs.rag.llm import complete_structured

JUDGE_SYSTEM_PROMPT = """\
Você avalia respostas de um assistente que tira dúvidas sobre a documentação da API do Asaas.

Compare a resposta do assistente com a resposta de referência, escrita por uma pessoa que \
conferiu a documentação. A referência é o gabarito: avalie os fatos, não o estilo.

- correct = true quando a resposta transmite os fatos essenciais da referência, mesmo com \
outras palavras ou com detalhes extras que não a contradizem.
- correct = false quando falta um fato essencial, quando a resposta contradiz a referência \
ou quando o assistente não respondeu à pergunta.
"""


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Justificativa do veredito em uma ou duas frases.")
    correct: bool


class Judge(Protocol):
    def grade(self, *, question: str, expected: str, actual: str) -> JudgeVerdict: ...


class ClaudeJudge:
    def __init__(self, client: anthropic.Anthropic, *, model: str, effort: Effort) -> None:
        self._client = client
        self._model = model
        self._effort = effort

    def grade(self, *, question: str, expected: str, actual: str) -> JudgeVerdict:
        user = (
            f"<pergunta>\n{question}\n</pergunta>\n\n"
            f"<resposta_de_referencia>\n{expected}\n</resposta_de_referencia>\n\n"
            f"<resposta_do_assistente>\n{actual}\n</resposta_do_assistente>"
        )
        completion = complete_structured(
            self._client,
            model=self._model,
            effort=self._effort,
            system=JUDGE_SYSTEM_PROMPT,
            user=user,
            output_format=JudgeVerdict,
        )
        return completion.output
