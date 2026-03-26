from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

from app.api.v1 import documents
from app.config import settings
from app.core.index_version import compute_index_version
from app.core.pdf_text import PageText
from app.core.vector_namespace import vector_namespace
from app.storage.local_store import StoredFile
from app.storage.registry import DocMeta


class FakeRegistry:
    def __init__(self, meta: DocMeta | None):
        self._meta = meta

    def get(self, doc_id: str):
        if self._meta and self._meta.doc_id == doc_id:
            return self._meta
        return None

    def save(self, meta: DocMeta) -> None:
        self._meta = meta

    def delete(self, doc_id: str) -> bool:
        if self._meta and self._meta.doc_id == doc_id:
            self._meta = None
            return True
        return False

    def list(self):
        return [self._meta] if self._meta else []


class SaveThenFailRegistry(FakeRegistry):
    def __init__(self, meta: DocMeta | None):
        super().__init__(meta)
        self.saved: list[DocMeta] = []

    def save(self, meta: DocMeta) -> None:
        self.saved.append(meta)
        raise RuntimeError("registry write failed")


class RecordingVectorStore:
    def __init__(self, previous_doc_summary: list[float] | None = None):
        self.previous_doc_summary = previous_doc_summary or [0.8, 0.2]
        self.upserts: list[tuple[str, list[tuple[str, dict]]]] = []
        self.cleared: list[str] = []
        self.deleted: list[tuple[str, list[str]]] = []

    def upsert(self, items, namespace: str) -> None:
        serialized = [(item["id"], dict(item.get("metadata", {}))) for item in items]
        self.upserts.append((namespace, serialized))

    def clear_namespace(self, namespace: str) -> None:
        self.cleared.append(namespace)

    def delete_ids(self, ids, namespace: str) -> None:
        self.deleted.append((namespace, list(ids)))

    def fetch(self, *, ids, namespace: str):
        if namespace == settings.doc_summary_namespace and ids:
            return {ids[0]: list(self.previous_doc_summary)}
        return {}


def _stored_file(tmp_path: Path, doc_id: str, slug: str = "doc") -> StoredFile:
    pdf_path = Path(__file__).resolve()
    return StoredFile(
        doc_id=doc_id,
        slug=slug,
        path=pdf_path,
        filename=f"{slug}.pdf",
        checksum="abc123",
        size_bytes=12,
        created=False,
    )


def _meta(*, doc_id: str, storage_path: Path, source_url: str, page_count: int = 1, route: str = "standard", index_version: str | None = None) -> DocMeta:
    version = index_version or compute_index_version(settings=settings, route=route)
    return DocMeta(
        doc_id=doc_id,
        slug="doc",
        filename="doc.pdf",
        source_url=source_url,
        checksum="abc123",
        size_bytes=12,
        page_count=page_count,
        token_count=42,
        route=route,
        storage_path=str(storage_path),
        index_version=version,
        created_at="2026-03-26T00:00:00+00:00",
    )


def test_exact_reupload_skips_ocr_when_unchanged(client, monkeypatch, work_tmp):
    stored = _stored_file(work_tmp, "doc__abc12345")
    existing = _meta(
        doc_id=stored.doc_id,
        storage_path=stored.path,
        source_url="https://example.com/source.pdf",
    )
    fake_registry = FakeRegistry(existing)

    monkeypatch.setattr(documents, "registry", fake_registry)
    monkeypatch.setattr(documents, "save_upload", lambda file, storage_dir: stored)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ocr_pdf_to_pages should not run for an exact no-op reupload")

    monkeypatch.setattr(documents, "ocr_pdf_to_pages", fail_if_called)

    response = client.post(
        "/api/v1/documents",
        data={"source_url": existing.source_url},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == existing.doc_id
    assert payload["index_version"] == existing.index_version
    assert payload["source_url"] == existing.source_url


def test_registry_failure_cleans_only_new_namespace(client, monkeypatch, work_tmp):
    stored = _stored_file(work_tmp, "doc__abc12345")
    existing = _meta(
        doc_id=stored.doc_id,
        storage_path=stored.path,
        source_url="https://example.com/old.pdf",
        index_version="oldver123456",
    )
    fake_registry = SaveThenFailRegistry(existing)
    vector_store = RecordingVectorStore()

    monkeypatch.setattr(documents, "registry", fake_registry)
    monkeypatch.setattr(documents, "save_upload", lambda file, storage_dir: stored)
    monkeypatch.setattr(
        documents,
        "ocr_pdf_to_pages",
        lambda **kwargs: [PageText(page_num=1, text="Intro paragraph.\n\nAnother paragraph.")],
    )
    monkeypatch.setattr(documents, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.core.indexing.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])

    response = client.post(
        "/api/v1/documents",
        data={"source_url": "https://example.com/new.pdf"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 500
    new_version = compute_index_version(settings=settings, route="standard")
    assert vector_store.cleared == [vector_namespace(stored.doc_id, new_version)]
    assert vector_namespace(stored.doc_id, existing.index_version) not in vector_store.cleared


def test_doc_summary_failure_restores_previous_registry_and_cleans_new_namespace(
    client, monkeypatch, work_tmp
):
    stored = _stored_file(work_tmp, "doc__abc12345")
    previous_meta = _meta(
        doc_id=stored.doc_id,
        storage_path=stored.path,
        source_url="https://example.com/old.pdf",
        index_version="oldver123456",
    )
    fake_registry = FakeRegistry(previous_meta)
    vector_store = RecordingVectorStore(previous_doc_summary=[0.4, 0.6])

    monkeypatch.setattr(documents, "registry", fake_registry)
    monkeypatch.setattr(documents, "save_upload", lambda file, storage_dir: stored)
    monkeypatch.setattr(
        documents,
        "ocr_pdf_to_pages",
        lambda **kwargs: [PageText(page_num=1, text="Intro paragraph.\n\nAnother paragraph.")],
    )
    monkeypatch.setattr(documents, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.core.indexing.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])

    def fail_doc_summary(**kwargs):
        raise RuntimeError("doc summary write failed")

    monkeypatch.setattr(documents, "upsert_doc_centroid", fail_doc_summary)

    response = client.post(
        "/api/v1/documents",
        data={"source_url": "https://example.com/new.pdf"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 502
    restored = fake_registry.get(stored.doc_id)
    assert restored is not None
    assert restored.index_version == previous_meta.index_version
    assert restored.source_url == previous_meta.source_url

    new_version = compute_index_version(settings=settings, route="standard")
    assert vector_namespace(stored.doc_id, new_version) in vector_store.cleared
    assert vector_namespace(stored.doc_id, previous_meta.index_version) not in vector_store.cleared
