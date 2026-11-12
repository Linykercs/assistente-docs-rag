from assistente_docs.ingestion.chunking import chunk_document
from assistente_docs.models import Document

URL = "https://docs.asaas.com/docs/sobre-os-webhooks"


def make_document(markdown: str, title: str = "Webhooks") -> Document:
    return Document(url=URL, title=title, markdown=markdown)


def test_chunks_carry_the_heading_trail_without_the_page_title() -> None:
    document = make_document(
        "# Webhooks\n\nIntrodução.\n\n"
        "## Criar\n\nPasso um.\n\n"
        "### Pela API\n\nUse POST.\n\n"
        "## Eventos\n\nLista de eventos.\n"
    )

    chunks = chunk_document(document, max_chars=500, overlap_chars=50)

    assert [(chunk.heading, chunk.content) for chunk in chunks] == [
        ("", "Introdução."),
        ("Criar", "Passo um."),
        ("Criar > Pela API", "Use POST."),
        ("Eventos", "Lista de eventos."),
    ]
    assert all(chunk.source_url == URL and chunk.title == "Webhooks" for chunk in chunks)


def test_sections_without_body_produce_no_chunks() -> None:
    document = make_document("# Webhooks\n\n## Vazia\n\n## Cheia\n\nConteúdo.\n")

    chunks = chunk_document(document, max_chars=500, overlap_chars=0)

    assert [chunk.heading for chunk in chunks] == ["Cheia"]


def test_long_sections_are_split_at_paragraphs_with_overlap() -> None:
    first, second, third = "a" * 40, "b" * 40, "c" * 40
    document = make_document(f"## Seção\n\n{first}\n\n{second}\n\n{third}\n")

    chunks = chunk_document(document, max_chars=90, overlap_chars=45)

    assert [chunk.content for chunk in chunks] == [
        f"{first}\n\n{second}",
        f"{second}\n\n{third}",
    ]
    assert {chunk.heading for chunk in chunks} == {"Seção"}


def test_no_overlap_when_overlap_is_zero() -> None:
    first, second, third = "a" * 40, "b" * 40, "c" * 40
    document = make_document(f"{first}\n\n{second}\n\n{third}\n")

    chunks = chunk_document(document, max_chars=90, overlap_chars=0)

    assert [chunk.content for chunk in chunks] == [f"{first}\n\n{second}", third]


def test_code_blocks_are_not_split_at_blank_lines_or_read_as_headings() -> None:
    code = '```python\n# não é um título\n\nprint("oi")\n```'
    document = make_document(f"## Exemplo\n\nVeja o código:\n\n{code}\n")

    chunks = chunk_document(document, max_chars=500, overlap_chars=0)

    assert len(chunks) == 1
    assert chunks[0].heading == "Exemplo"
    assert chunks[0].content == f"Veja o código:\n\n{code}"


def test_oversized_blocks_are_split_at_line_breaks() -> None:
    rows = [f"| evento_{index:02d} | descrição do evento |" for index in range(10)]
    document = make_document("\n".join(rows) + "\n")

    chunks = chunk_document(document, max_chars=120, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 120 for chunk in chunks)
    assert "\n".join(chunk.content for chunk in chunks).split("\n") == rows


def test_content_hash_is_stable_and_sensitive_to_content() -> None:
    document = make_document("## Seção\n\nTexto.\n")
    changed = make_document("## Seção\n\nTexto alterado.\n")

    [chunk] = chunk_document(document, max_chars=500, overlap_chars=0)
    [same] = chunk_document(document, max_chars=500, overlap_chars=0)
    [different] = chunk_document(changed, max_chars=500, overlap_chars=0)

    assert chunk.content_hash == same.content_hash
    assert chunk.content_hash != different.content_hash


def test_embedding_text_includes_title_and_heading() -> None:
    [chunk] = chunk_document(make_document("## Eventos\n\nLista.\n"), max_chars=500, overlap_chars=0)

    assert chunk.embedding_text == "Webhooks > Eventos\n\nLista."
