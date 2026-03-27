from __future__ import annotations

from types import SimpleNamespace

from app.core.context import select_context
from app.core.rerank import rerank_items
from app.core.retrieval import RetrievalItem


def _settings(**overrides):
    base = {
        "max_context_tokens": 6,
        "context_max_item_tokens": 0,
        "context_use_mmr": False,
        "context_mmr_lambda": 0.7,
        "openai_generation_model": "dummy",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _item(
    *,
    source_id: str,
    doc_id: str,
    text: str,
    score: float,
    page_start: int = 1,
    page_end: int = 1,
    section_title: str | None = None,
    route: str = "standard",
    embedding: list[float] | None = None,
):
    return RetrievalItem(
        source_id=source_id,
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        file_url=None,
        source_url=None,
        text=text,
        score=score,
        page_start=page_start,
        page_end=page_end,
        section_title=section_title,
        route=route,
        embedding=embedding,
    )


def test_select_context_skips_oversized_items_instead_of_stopping():
    items = [
        _item(
            source_id="a",
            doc_id="doc-a",
            text="one two three four five six seven",
            score=0.9,
        ),
        _item(
            source_id="b",
            doc_id="doc-b",
            text="short useful evidence",
            score=0.8,
        ),
    ]

    selected, context_items = select_context(items=items, settings=_settings())

    assert [item.source_id for item in selected] == ["b"]
    assert [item["doc_id"] for item in context_items] == ["doc-b"]


def test_select_context_preserves_same_text_across_docs():
    items = [
        _item(
            source_id="a",
            doc_id="doc-a",
            text="shared corroborating passage",
            score=0.9,
        ),
        _item(
            source_id="b",
            doc_id="doc-b",
            text="shared corroborating passage",
            score=0.85,
        ),
    ]

    selected, context_items = select_context(items=items, settings=_settings(max_context_tokens=20))

    assert [item.source_id for item in selected] == ["a", "b"]
    assert [item["doc_id"] for item in context_items] == ["doc-a", "doc-b"]


def test_rerank_prefers_a_useful_nonredundant_second_item():
    items = [
        _item(
            source_id="a",
            doc_id="doc-a",
            text="alpha passage",
            score=0.98,
            embedding=[1.0, 0.0],
        ),
        _item(
            source_id="b",
            doc_id="doc-b",
            text="alpha passage",
            score=0.97,
            embedding=[1.0, 0.0],
        ),
        _item(
            source_id="c",
            doc_id="doc-c",
            text="different but still relevant passage",
            score=0.95,
            page_end=1,
            section_title="Relevant section",
            route="tree",
            embedding=[0.9, 0.1],
        ),
    ]

    reranked = rerank_items(
        query_embedding=[1.0, 0.0],
        items=items,
        top_k=2,
        candidate_k=3,
    )

    assert [item.source_id for item in reranked] == ["c", "a"]
    assert reranked[0].score > reranked[1].score
