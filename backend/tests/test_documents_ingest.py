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

    def find_by_checksum(self, checksum: str):
        if (
            self._meta
            and str(self._meta.checksum or "").lower() == str(checksum or "").lower()
        ):
            return self._meta
        return None


class SaveThenFailRegistry(FakeRegistry):
    def __init__(self, meta: DocMeta | None):
        super().__init__(meta)
        self.saved: list[DocMeta] = []

    def save(self, meta: DocMeta) -> None:
        self.saved.append(meta)
        raise RuntimeError("registry write failed")


class RecordingVectorStore:
    def __init__(self, previous_doc_summary_records: dict[str, dict] | None = None):
        self.previous_doc_summary_records = previous_doc_summary_records or {
            "doc__abc12345:profile": {
                "id": "doc__abc12345:profile",
                "values": [0.8, 0.2],
                "metadata": {"doc_id": "doc__abc12345", "summary_kind": "profile"},
            }
        }
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
            first_id = ids[0]
            payload = self.previous_doc_summary_records.get(first_id)
            if payload:
                return {first_id: list(payload.get("values") or [])}
        return {}

    def fetch_records(self, *, ids, namespace: str):
        if namespace != settings.doc_summary_namespace:
            return {}
        return {
            record_id: dict(payload)
            for record_id, payload in self.previous_doc_summary_records.items()
            if record_id in ids
        }


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


def _meta(*, doc_id: str, storage_path: Path | None, source_url: str, page_count: int = 1, route: str = "standard", index_version: str | None = None) -> DocMeta:
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
        storage_path=str(storage_path) if storage_path is not None else None,
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
    assert payload["file_url"] is None


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
    monkeypatch.setattr("app.core.doc_summaries.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])

    response = client.post(
        "/api/v1/documents",
        data={"source_url": "https://example.com/new.pdf"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 500
    new_version = compute_index_version(settings=settings, route="standard")
    assert vector_store.cleared == [vector_namespace(stored.doc_id, new_version)]
    assert vector_namespace(stored.doc_id, existing.index_version) not in vector_store.cleared


def test_successful_ingest_persists_index_only_metadata(client, monkeypatch, work_tmp):
    stored = _stored_file(work_tmp, "doc__abc12345")
    fake_registry = FakeRegistry(None)
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
    monkeypatch.setattr("app.core.doc_summaries.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])

    response = client.post(
        "/api/v1/documents",
        data={"source_url": "https://example.com/new.pdf"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["file_url"] is None

    saved = fake_registry.get(stored.doc_id)
    assert saved is not None
    assert saved.storage_path is None

    all_metadata = [
        metadata
        for _namespace, items in vector_store.upserts
        for _item_id, metadata in items
    ]
    assert all("file_url" not in metadata for metadata in all_metadata)


def test_successful_ingest_keeps_legacy_centroid_and_multi_vector_summaries(
    client, monkeypatch, work_tmp
):
    stored = _stored_file(work_tmp, "doc__abc12345")
    fake_registry = FakeRegistry(None)
    vector_store = RecordingVectorStore()

    monkeypatch.setattr(documents, "registry", fake_registry)
    monkeypatch.setattr(documents, "save_upload", lambda file, storage_dir: stored)
    monkeypatch.setattr(
        documents,
        "ocr_pdf_to_pages",
        lambda **kwargs: [PageText(page_num=1, text="Benefits overview.\n\nEligibility details.")],
    )
    monkeypatch.setattr(documents, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.core.indexing.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr("app.core.doc_summaries.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])

    response = client.post(
        "/api/v1/documents",
        data={"source_url": "https://example.com/new.pdf"},
        files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 200
    summary_upserts = [
        items
        for namespace, items in vector_store.upserts
        if namespace == settings.doc_summary_namespace
    ]
    assert summary_upserts

    summary_ids = [item_id for items in summary_upserts for item_id, _metadata in items]
    assert stored.doc_id in summary_ids
    assert f"{stored.doc_id}:profile" in summary_ids
    assert f"{stored.doc_id}:headings" in summary_ids


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
    vector_store = RecordingVectorStore(
        previous_doc_summary_records={
            f"{stored.doc_id}:profile": {
                "id": f"{stored.doc_id}:profile",
                "values": [0.4, 0.6],
                "metadata": {"doc_id": stored.doc_id, "summary_kind": "profile"},
            }
        }
    )

    monkeypatch.setattr(documents, "registry", fake_registry)
    monkeypatch.setattr(documents, "save_upload", lambda file, storage_dir: stored)
    monkeypatch.setattr(
        documents,
        "ocr_pdf_to_pages",
        lambda **kwargs: [PageText(page_num=1, text="Intro paragraph.\n\nAnother paragraph.")],
    )
    monkeypatch.setattr(documents, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr("app.core.indexing.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr("app.core.doc_summaries.embed_texts", lambda texts, settings: [[0.1, 0.2] for _ in texts])

    def fail_doc_summary(**kwargs):
        raise RuntimeError("doc summary write failed")

    monkeypatch.setattr(documents, "upsert_doc_summaries", fail_doc_summary)

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
    assert restored.storage_path == previous_meta.storage_path

    new_version = compute_index_version(settings=settings, route="standard")
    assert vector_namespace(stored.doc_id, new_version) in vector_store.cleared
    assert vector_namespace(stored.doc_id, previous_meta.index_version) not in vector_store.cleared


def test_checksum_duplicate_reuses_existing_doc_identity(client, monkeypatch, work_tmp):
    stored = _stored_file(work_tmp, "renamed__abc12345", slug="renamed")
    existing = _meta(
        doc_id="doc__abc12345",
        storage_path=stored.path,
        source_url="https://example.com/source.pdf",
    )
    fake_registry = FakeRegistry(existing)

    monkeypatch.setattr(documents, "registry", fake_registry)
    monkeypatch.setattr(documents, "save_upload", lambda file, storage_dir: stored)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("ocr_pdf_to_pages should not run for a checksum-identical reupload")

    monkeypatch.setattr(documents, "ocr_pdf_to_pages", fail_if_called)

    response = client.post(
        "/api/v1/documents",
        data={"source_url": existing.source_url},
        files={"file": ("renamed.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == existing.doc_id
    assert payload["source_url"] == existing.source_url
