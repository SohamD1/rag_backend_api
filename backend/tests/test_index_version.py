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


def test_chunking_splits_on_heading_boundaries():
    chunks = chunk_document(
        doc_id="doc1",
        pages=[
            PageText(
                page_num=1,
                text=(
                    "Benefits Overview\n\n"
                    "Eligibility begins on the first of the month.\n\n"
                    "Enrollment Rules\n\n"
                    "Open enrollment changes must be submitted within thirty days."
                ),
            )
        ],
        model=settings.openai_embedding_model,
        min_tokens=1,
        target_tokens=200,
        max_tokens=300,
        overlap_tokens=0,
    )

    assert len(chunks) == 2
    assert chunks[0].section_title == "Benefits Overview"
    assert chunks[0].text.startswith("Benefits Overview\n\n")
    assert "Eligibility begins on the first of the month." in chunks[0].text
    assert chunks[1].section_title == "Enrollment Rules"
    assert chunks[1].text.startswith("Enrollment Rules\n\n")
    assert "Open enrollment changes must be submitted within thirty days." in chunks[1].text


def test_chunking_keeps_active_heading_across_pages_until_replaced():
    chunks = chunk_document(
        doc_id="doc1",
        pages=[
            PageText(
                page_num=1,
                text=(
                    "Benefits Overview\n\n"
                    "Eligibility begins on the first of the month."
                ),
            ),
            PageText(
                page_num=2,
                text="Coverage remains active while employed.",
            ),
        ],
        model=settings.openai_embedding_model,
        min_tokens=1,
        target_tokens=200,
        max_tokens=300,
        overlap_tokens=0,
    )

    assert len(chunks) == 1
    assert chunks[0].section_title == "Benefits Overview"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].text.startswith("Benefits Overview\n\n")
    assert "Coverage remains active while employed." in chunks[0].text
