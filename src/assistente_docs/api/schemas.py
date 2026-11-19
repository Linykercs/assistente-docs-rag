"""Request and response models of the HTTP API."""

from dataclasses import asdict
from typing import Literal, Self

from pydantic import BaseModel, Field

from assistente_docs.models import AnswerStatus, RagAnswer


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        examples=["Como crio uma cobrança via Pix?"],
    )


class SourceOut(BaseModel):
    id: int = Field(description="Número usado nas citações [n] do texto da resposta.")
    url: str
    title: str
    section: str
    score: float = Field(description="Similaridade de cosseno entre a pergunta e o trecho.")


class UsageOut(BaseModel):
    input_tokens: int = Field(
        description="Tokens enviados ao LLM (instruções + trechos + pergunta)."
    )
    output_tokens: int = Field(description="Tokens gerados pelo LLM.")
    retrieve_ms: float = Field(description="Tempo da busca dos trechos, em milissegundos.")
    answer_ms: float = Field(description="Tempo da chamada ao LLM, em milissegundos.")


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str
    sources: list[SourceOut]
    model: str | None
    usage: UsageOut

    @classmethod
    def from_answer(cls, answer: RagAnswer) -> Self:
        return cls(
            status=answer.status,
            answer=answer.answer,
            sources=[SourceOut(**asdict(source)) for source in answer.sources],
            model=answer.model,
            usage=UsageOut(**asdict(answer.usage)),
        )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
