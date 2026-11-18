"""FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from assistente_docs import __version__
from assistente_docs.api.ratelimit import RateLimiter, client_key
from assistente_docs.api.schemas import AskRequest, AskResponse, HealthResponse
from assistente_docs.bootstrap import open_rag_service
from assistente_docs.config import Settings, get_settings
from assistente_docs.rag.llm import LlmError
from assistente_docs.rag.service import AskService


def create_app(service: AskService | None = None, settings: Settings | None = None) -> FastAPI:
    """Create the app. Tests pass a fake `service`; otherwise the real one is built at startup."""
    limits = settings or get_settings()
    limiter = RateLimiter(
        per_hour=limits.demo_questions_per_hour, per_day=limits.demo_questions_per_day
    )
    page = resources.files("assistente_docs").joinpath("api", "static", "index.html")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.service = service
            yield
            return
        with open_rag_service(get_settings()) as rag_service:
            app.state.service = rag_service
            yield

    app = FastAPI(
        title="Assistente da documentação do Asaas",
        description="Responde dúvidas sobre a API do Asaas com base na documentação oficial.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> str:
        return page.read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/ask")
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        refused = limiter.check(client_key(request))
        if refused:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=refused)
        ask_service: AskService = request.app.state.service
        try:
            answer = ask_service.ask(payload.question)
        except LlmError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
        return AskResponse.from_answer(answer)

    return app


app = create_app()
