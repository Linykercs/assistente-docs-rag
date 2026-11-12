from assistente_docs.ingestion.cleaning import clean_markdown, extract_title, html_to_markdown

README_PAGE = """---
---
updatedAt: 2026-10-13T13:37:54.000Z
agentTools:
  projectIndex: https://docs.asaas.com/llms.txt
---

# Autenticação

Envie a chave no header `access_token`.<br />Ela identifica a sua conta.

<Callout icon="📘" theme="info">
  ### Dica

  Consulte o <Anchor target="_blank" href="https://docs.asaas.com/docs/sandbox">Sandbox</Anchor>.
</Callout>

```json
{
  "tag": "<Callout>",

  "access_token": "sua_chave"
}
```

```mermaid
flowchart LR
  A --> B
```

# OpenAPI definition

```json
{"openapi": "3.0.0"}
```

# Sibling pages

- [Chaves de API](https://docs.asaas.com/docs/chaves-de-api.md)
"""


def test_clean_markdown_strips_frontmatter() -> None:
    cleaned = clean_markdown(README_PAGE)

    assert cleaned.startswith("# Autenticação\n")
    assert "updatedAt" not in cleaned
    assert "projectIndex" not in cleaned


def test_clean_markdown_converts_components_to_plain_markdown() -> None:
    cleaned = clean_markdown(README_PAGE)

    assert "[Sandbox](https://docs.asaas.com/docs/sandbox)" in cleaned
    assert "<Callout" not in cleaned.split("```json")[0]
    assert "</Callout>" not in cleaned
    assert "<br" not in cleaned
    assert "\n### Dica\n" in cleaned


def test_clean_markdown_keeps_code_blocks_untouched() -> None:
    cleaned = clean_markdown(README_PAGE)

    assert '"tag": "<Callout>",\n\n  "access_token": "sua_chave"' in cleaned


def test_clean_markdown_drops_diagrams_and_generated_sections() -> None:
    cleaned = clean_markdown(README_PAGE)

    assert "mermaid" not in cleaned
    assert "flowchart" not in cleaned
    assert "OpenAPI definition" not in cleaned
    assert "openapi" not in cleaned
    assert "Sibling pages" not in cleaned
    assert "chaves-de-api" not in cleaned


def test_clean_markdown_keeps_horizontal_rules_in_pages_without_frontmatter() -> None:
    cleaned = clean_markdown("# Título\n\nAntes.\n\n---\n\nDepois.\n")

    assert cleaned == "# Título\n\nAntes.\n\n---\n\nDepois.\n"


def test_html_to_markdown_keeps_main_content_only() -> None:
    html = """
    <html>
      <head><title>Docs</title><style>body { color: red; }</style></head>
      <body>
        <nav><a href="/">Menu principal</a></nav>
        <main>
          <h1>Webhooks</h1>
          <p>Receba <strong>eventos</strong> no seu endpoint.</p>
          <script>console.log("tracking")</script>
          <footer>Rodapé</footer>
        </main>
      </body>
    </html>
    """

    markdown = html_to_markdown(html)

    assert markdown.startswith("# Webhooks")
    assert "Receba **eventos** no seu endpoint." in markdown
    for noise in ("Menu principal", "tracking", "Rodapé", "color: red"):
        assert noise not in markdown


def test_extract_title_uses_first_h1_or_fallback() -> None:
    assert extract_title("# Sandbox\n\nTexto.\n\n# Outro\n", fallback="slug") == "Sandbox"
    assert extract_title("Texto sem título.\n", fallback="slug") == "slug"
