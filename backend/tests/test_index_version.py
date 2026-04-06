from __future__ import annotations

from dataclasses import replace

from app.config import settings
from app.core.chunking import chunk_document
from app.core.index_version import compute_index_version
from app.core.pdf_text import PageText


def test_index_version_changes_when_ocr_settings_change():
    base = compute_index_version(settings=settings, route="standard")
    changed = compute_index_version(
        settings=replace(settings, mistral_ocr_model="mistral-ocr-next"),
        route="standard",
    )
    assert changed != base


def test_index_version_changes_when_chunking_logic_changes():
    base = compute_index_version(settings=settings, route="standard")
    changed = compute_index_version(
        settings=replace(settings, chunking_logic_version="v3_layout_tuned"),
        route="standard",
    )
    assert changed != base


def test_index_version_changes_when_doc_summary_strategy_changes():
    base = compute_index_version(settings=settings, route="standard")
    changed = compute_index_version(
        settings=replace(settings, doc_summary_strategy_version="multi_vector_v2"),
        route="standard",
    )
    assert changed != base


def test_chunking_preserves_list_layout_for_ocrish_pages():
    chunks = chunk_document(
        doc_id="doc1",
        pages=[
            PageText(
                page_num=1,
                text="Item one\nItem two\nItem three\nItem four",
            )
        ],
        model=settings.openai_embedding_model,
        min_tokens=1,
        target_tokens=100,
        max_tokens=200,
        overlap_tokens=0,
    )

    assert len(chunks) == 1
    assert chunks[0].layout_mode == "preserve_lines"
    assert "Item one\n\nItem two" in chunks[0].text
