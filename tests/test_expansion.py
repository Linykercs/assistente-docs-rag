from assistente_docs.ingestion.chunking import chunk_document
from assistente_docs.models import Chunk, Document, RetrievedChunk
from assistente_docs.rag.expansion import ExpandingRetriever, expansion_group, top_section
from fakes import FakeRetriever

URL = "https://docs.asaas.com/docs/webhook-para-cobrancas"


def chunk(position: int, heading: str, content: str = "", url: str = URL) -> Chunk:
    return Chunk(
        source_url=url,
        title="Eventos",
        heading=heading,
        content=content or f"Conteúdo {position}.",
        position=position,
    )


# Uma página como a de eventos: a lista está quebrada em três seções de "Eventos
# disponíveis", e o payload é outra seção de topo.
PAGE = [
    chunk(0, ""),
    chunk(1, "Eventos disponíveis > Ciclo da cobrança", "PAYMENT_CREATED"),
    chunk(2, "Eventos disponíveis > Estornos", "PAYMENT_REFUNDED"),
    chunk(3, "Eventos disponíveis > Boleto", "PAYMENT_BANK_SLIP_VIEWED"),
    chunk(4, "Payload do evento", "payload"),
    chunk(5, "Payload do evento > Campos opcionais", "campos"),
    chunk(6, "Próximos passos", "links"),
]


class FakePages:
    def __init__(self, pages: dict[str, list[Chunk]]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def page_chunks(self, source_url: str) -> list[Chunk]:
        self.requested.append(source_url)
        return self.pages.get(source_url, [])


def test_chunks_are_numbered_in_page_order() -> None:
    document = Document(url=URL, title="T", markdown="# T\n\nA\n\n## B\n\nB\n\n## C\n\nC")

    chunks = chunk_document(document, max_chars=1500, overlap_chars=0)

    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_position_is_not_part_of_the_chunk_identity() -> None:
    assert chunk(1, "X", "igual").content_hash == chunk(7, "X", "igual").content_hash


def test_top_section_is_the_first_heading_of_the_trail() -> None:
    assert top_section(PAGE[2]) == "Eventos disponíveis"
    assert top_section(PAGE[0]) == ""


def test_group_pulls_the_whole_top_level_section_and_the_neighbors() -> None:
    group = expansion_group(PAGE[1], PAGE, max_chars=10_000)

    # Section siblings 2 and 3, plus neighbor 0 before it.
    assert [c.position for c in group] == [0, 1, 2, 3]


def test_group_without_a_section_takes_only_the_neighbors() -> None:
    steps = [chunk(0, "1. Acesse"), chunk(1, "2. Configure"), chunk(2, "3. Selecione")]

    # Step-by-step: each step is its own top-level section, so the question that matches
    # step 2 still gets step 1 (where "Criar Webhook" is) through the neighbor rule.
    group = expansion_group(steps[1], steps, max_chars=10_000)

    assert [c.position for c in group] == [0, 1, 2]


def test_group_respects_the_budget_keeping_the_closest_chunks() -> None:
    page = [chunk(i, "Eventos disponíveis > X", f"{i}" + "x" * 99) for i in range(6)]

    group = expansion_group(page[3], page, max_chars=300)

    assert [c.position for c in group] == [2, 3, 4]


def test_the_seed_survives_even_when_it_alone_exceeds_the_budget() -> None:
    page = [chunk(0, "A", "x" * 500), chunk(1, "A", "y" * 10)]

    group = expansion_group(page[0], page, max_chars=100)

    assert page[0] in group


def test_retriever_expands_only_the_best_chunks_and_drops_duplicates() -> None:
    other = chunk(0, "Outra página", url="https://docs.asaas.com/docs/outra")
    retrieved = [
        RetrievedChunk(chunk=PAGE[2], score=0.9),
        RetrievedChunk(chunk=PAGE[3], score=0.8),  # already pulled in by the first expansion
        RetrievedChunk(chunk=other, score=0.7),
    ]
    pages = FakePages({URL: PAGE})
    retriever = ExpandingRetriever(FakeRetriever(retrieved), pages, expand_top=1, max_chars=10_000)

    result = retriever.retrieve("Quais eventos existem?")

    assert [(r.chunk.source_url, r.chunk.position) for r in result] == [
        (URL, 1),
        (URL, 2),
        (URL, 3),
        (other.source_url, 0),
    ]
    # Expanded chunks borrow the score of the chunk that pulled them in.
    assert {r.score for r in result[:3]} == {0.9}
    assert pages.requested == [URL]


def test_retriever_with_expand_top_zero_returns_the_inner_result() -> None:
    retrieved = [RetrievedChunk(chunk=PAGE[2], score=0.9)]
    retriever = ExpandingRetriever(
        FakeRetriever(retrieved), FakePages({URL: PAGE}), expand_top=0, max_chars=10_000
    )

    assert retriever.retrieve("pergunta") == retrieved
