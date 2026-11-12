from pathlib import Path

import httpx

from assistente_docs.ingestion.fetcher import PageFetcher
from assistente_docs.ingestion.pipeline import IngestReport, ingest, load_document
from assistente_docs.ingestion.sources import load_source_urls
from fakes import FakeEmbedder, InMemoryChunkStore

PAGE_URL = "https://docs.asaas.com/docs/sandbox"
MARKDOWN_PAGE = "---\nupdatedAt: 2026-01-01\n---\n\n# Sandbox\n\nAmbiente de testes.\n"
HTML_PAGE = "<html><body><main><h1>Sandbox</h1><p>Ambiente de testes.</p></main></body></html>"


class FakeDocsSite:
    """An httpx transport that serves canned pages and records the requested URLs."""

    def __init__(self, pages: dict[str, tuple[str, str]]) -> None:
        self.pages = pages
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        page = self.pages.get(str(request.url))
        if page is None:
            return httpx.Response(404)
        content_type, body = page
        return httpx.Response(200, headers={"content-type": content_type}, text=body)

    def fetcher(self, cache_dir: Path, *, refresh: bool = False) -> PageFetcher:
        client = httpx.Client(
            transport=httpx.MockTransport(self), headers={"User-Agent": "test-agent/1.0"}
        )
        return PageFetcher(client, cache_dir, delay_seconds=0, refresh=refresh)


def test_fetcher_identifies_itself_and_caches_pages(tmp_path: Path) -> None:
    site = FakeDocsSite({f"{PAGE_URL}.md": ("text/markdown", MARKDOWN_PAGE)})

    first = site.fetcher(tmp_path).fetch(f"{PAGE_URL}.md")
    second = site.fetcher(tmp_path).fetch(f"{PAGE_URL}.md")

    assert first == second
    assert first.body == MARKDOWN_PAGE
    assert not first.is_html
    assert len(site.requests) == 1
    assert site.requests[0].headers["user-agent"] == "test-agent/1.0"


def test_fetcher_refresh_bypasses_the_cache(tmp_path: Path) -> None:
    site = FakeDocsSite({f"{PAGE_URL}.md": ("text/markdown", MARKDOWN_PAGE)})

    site.fetcher(tmp_path).fetch(f"{PAGE_URL}.md")
    site.fetcher(tmp_path, refresh=True).fetch(f"{PAGE_URL}.md")

    assert len(site.requests) == 2


def test_load_document_prefers_the_markdown_version(tmp_path: Path) -> None:
    site = FakeDocsSite({f"{PAGE_URL}.md": ("text/markdown; charset=utf-8", MARKDOWN_PAGE)})

    document = load_document(site.fetcher(tmp_path), PAGE_URL)

    assert document.url == PAGE_URL
    assert document.title == "Sandbox"
    assert document.markdown == "# Sandbox\n\nAmbiente de testes.\n"


def test_load_document_falls_back_to_html(tmp_path: Path) -> None:
    site = FakeDocsSite({PAGE_URL: ("text/html; charset=utf-8", HTML_PAGE)})

    document = load_document(site.fetcher(tmp_path), PAGE_URL)

    assert document.url == PAGE_URL
    assert document.title == "Sandbox"
    assert "Ambiente de testes." in document.markdown


def test_ingest_is_idempotent_and_removes_stale_chunks(tmp_path: Path) -> None:
    other_url = "https://docs.asaas.com/docs/pix"
    site = FakeDocsSite(
        {
            f"{PAGE_URL}.md": ("text/markdown", MARKDOWN_PAGE),
            f"{other_url}.md": ("text/markdown", "# Pix\n\nPagamento instantâneo.\n"),
        }
    )
    store = InMemoryChunkStore()

    def run(urls: list[str], *, refresh: bool = False) -> IngestReport:
        return ingest(
            urls,
            fetcher=site.fetcher(tmp_path, refresh=refresh),
            embedder=FakeEmbedder(),
            store=store,
            max_chars=500,
            overlap_chars=0,
        )

    first = run([PAGE_URL, other_url])
    assert (first.documents, first.chunks, first.embedded, first.deleted) == (2, 2, 2, 0)

    second = run([PAGE_URL, other_url])
    assert (second.embedded, second.deleted) == (0, 0)

    site.pages[f"{PAGE_URL}.md"] = ("text/markdown", "# Sandbox\n\nTexto novo.\n")
    third = run([PAGE_URL], refresh=True)
    assert (third.chunks, third.embedded, third.deleted) == (1, 1, 2)
    assert [chunk.content for chunk in store.chunks.values()] == ["Texto novo."]


def test_load_source_urls_deduplicates_and_keeps_order(tmp_path: Path) -> None:
    sources_file = tmp_path / "sources.yaml"
    sources_file.write_text(
        "urls:\n"
        "  - https://docs.asaas.com/docs/sandbox\n"
        "  - https://docs.asaas.com/docs/pix/\n"
        "  - https://docs.asaas.com/docs/sandbox\n",
        encoding="utf-8",
    )

    assert load_source_urls(sources_file) == [
        "https://docs.asaas.com/docs/sandbox",
        "https://docs.asaas.com/docs/pix",
    ]


def test_bundled_sources_file_is_valid() -> None:
    urls = load_source_urls(Path(__file__).parents[1] / "sources.yaml")

    assert urls
    assert all(url.startswith("https://docs.asaas.com/") for url in urls)
