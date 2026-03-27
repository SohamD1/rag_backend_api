from __future__ import annotations

from app.config import settings
from app.core.pdf_text import PageText
from app.services import tree_index


def _build_tree_with_fake_toc(monkeypatch, pages: list[PageText], toc_items: list[dict[str, object]]):
    monkeypatch.setattr(tree_index, "_find_toc_pages", lambda pages, settings: [])
    monkeypatch.setattr(tree_index, "_extract_toc_items", lambda pages, settings: list(toc_items))
    monkeypatch.setattr(tree_index, "_verify_and_fix_toc_items", lambda toc_items, pages, settings: toc_items)
    monkeypatch.setattr(tree_index, "_summarize_heading_nodes", lambda **kwargs: None)
    monkeypatch.setattr(tree_index, "_split_large_nodes_recursively", lambda **kwargs: None)
    return tree_index.build_tree("doc-1", pages, settings)


def _paragraph_parent_map(nodes):
    return {
        node.text_span: node.parent_id
        for node in nodes.values()
        if node.level == "paragraph"
    }


def test_build_tree_switches_heading_mid_page(monkeypatch):
    pages = [
        PageText(
            page_num=1,
            text=(
                "Section One\n\n"
                "Alpha paragraph.\n\n"
                "Section Two\n\n"
                "Beta paragraph."
            ),
        )
    ]
    nodes = _build_tree_with_fake_toc(
        monkeypatch,
        pages,
        [
            {"structure": "1", "title": "Section One", "physical_index": 1},
            {"structure": "2", "title": "Section Two", "physical_index": 1},
        ],
    )

    parents = _paragraph_parent_map(nodes)
    assert parents["Section One"] == "doc-1:sec:1"
    assert parents["Alpha paragraph."] == "doc-1:sec:1"
    assert parents["Section Two"] == "doc-1:sec:2"
    assert parents["Beta paragraph."] == "doc-1:sec:2"


def test_build_tree_matches_numbered_heading_prefixes(monkeypatch):
    pages = [
        PageText(
            page_num=1,
            text=(
                "1. Overview\n\n"
                "Intro paragraph.\n\n"
                "2. Details\n\n"
                "Follow-up paragraph."
            ),
        )
    ]
    nodes = _build_tree_with_fake_toc(
        monkeypatch,
        pages,
        [
            {"structure": "1", "title": "Overview", "physical_index": 1},
            {"structure": "2", "title": "Details", "physical_index": 1},
        ],
    )

    parents = _paragraph_parent_map(nodes)
    assert parents["1. Overview"] == "doc-1:sec:1"
    assert parents["Intro paragraph."] == "doc-1:sec:1"
    assert parents["2. Details"] == "doc-1:sec:2"
    assert parents["Follow-up paragraph."] == "doc-1:sec:2"
