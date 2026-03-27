from __future__ import annotations

from dataclasses import replace

from app.config import settings
from app.core.pipeline import run_chat
from app.core.retrieval import RetrievalItem
from app.storage.registry import DocMeta


class FakeRegistry:
    def __init__(self, metas):
        self._metas = list(metas)

    def list(self):
        return list(self._metas)


class RecordingCache:
    def __init__(self):
        self._data = {}
        self.set_calls = []

    def get(self, key):
        value = self._data.get(key)
        if value is None:
            return None
        return type("CacheEntry", (), {"key": key, "value": value})()

    def set(self, key, value):
        self.set_calls.append((key, value))
        self._data[key] = value


class FetchingVectorStore:
    def __init__(self, embedding_by_id):
        self.embedding_by_id = dict(embedding_by_id)

    def fetch(self, *, ids, namespace):
        return {
            source_id: self.embedding_by_id[source_id]
            for source_id in ids
            if source_id in self.embedding_by_id
        }


def _meta(doc_id: str) -> DocMeta:
    return DocMeta(
        doc_id=doc_id,
        slug=doc_id,
        filename=f"{doc_id}.pdf",
        source_url=f"https://example.com/{doc_id}.pdf",
        checksum="abc123",
        size_bytes=1,
        page_count=1,
        token_count=10,
        route="standard",
        storage_path=None,
        index_version="ver1",
        created_at="2026-03-26T00:00:00+00:00",
    )


def test_degraded_retrieval_does_not_write_caches(monkeypatch):
    from app.core import pipeline

    retrieval_cache = RecordingCache()
    response_cache = RecordingCache()
    registry = FakeRegistry([_meta("doc1"), _meta("doc2")])
    test_settings = replace(settings, rag_generate_answers_enabled=False)

    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, settings: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline,
        "select_docs_with_rewrite_retry",
        lambda **kwargs: (["doc1", "doc2"], False, kwargs["query"], kwargs["query_embedding"]),
    )

    def fake_retrieve_standard_for_doc(**kwargs):
        if kwargs["doc_id"] == "doc2":
            raise RuntimeError("vector outage")
        return [
            RetrievalItem(
                source_id="doc1:c1",
                doc_id="doc1",
                filename="doc1.pdf",
                file_url=None,
                source_url="https://example.com/doc1.pdf",
                text="useful evidence",
                score=0.9,
                page_start=1,
                page_end=1,
                section_title="Intro",
                route="standard",
                vector_namespace="doc1::ver1",
            )
        ]

    monkeypatch.setattr(pipeline, "retrieve_standard_for_doc", fake_retrieve_standard_for_doc)
    monkeypatch.setattr(pipeline, "retrieve_tree_for_doc", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "select_context",
        lambda **kwargs: (
            kwargs["items"],
            [{"header": "doc=doc1 pages=1-1", "text": "useful evidence"}],
        ),
    )
    monkeypatch.setattr(pipeline, "generate_answer_and_summary", lambda **kwargs: ("", None))

    payload = run_chat(
        query="what is useful",
        debug_enabled=True,
        settings=test_settings,
        registry=registry,
        vector_store=object(),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["chunks"]
    assert payload["debug"]["retrieval_degraded"]["stages"][0]["stage"] == "retrieve"
    assert retrieval_cache.set_calls == []
    assert response_cache.set_calls == []


def test_clean_retrieval_writes_caches(monkeypatch):
    from app.core import pipeline

    retrieval_cache = RecordingCache()
    response_cache = RecordingCache()
    registry = FakeRegistry([_meta("doc1")])
    test_settings = replace(settings, rag_generate_answers_enabled=False)

    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, settings: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline,
        "select_docs_with_rewrite_retry",
        lambda **kwargs: (["doc1"], False, kwargs["query"], kwargs["query_embedding"]),
    )
    monkeypatch.setattr(
        pipeline,
        "retrieve_standard_for_doc",
        lambda **kwargs: [
            RetrievalItem(
                source_id="doc1:c1",
                doc_id="doc1",
                filename="doc1.pdf",
                file_url=None,
                source_url="https://example.com/doc1.pdf",
                text="useful evidence",
                score=0.9,
                page_start=1,
                page_end=1,
                section_title="Intro",
                route="standard",
                vector_namespace="doc1::ver1",
            )
        ],
    )
    monkeypatch.setattr(pipeline, "retrieve_tree_for_doc", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "select_context",
        lambda **kwargs: (
            kwargs["items"],
            [{"header": "doc=doc1 pages=1-1", "text": "useful evidence"}],
        ),
    )
    monkeypatch.setattr(pipeline, "generate_answer_and_summary", lambda **kwargs: ("", None))

    vector_store = FetchingVectorStore({"doc1:c1": [0.1, 0.2]})
    payload = run_chat(
        query="what is useful",
        debug_enabled=True,
        settings=test_settings,
        registry=registry,
        vector_store=vector_store,
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["chunks"]
    assert retrieval_cache.set_calls
    assert response_cache.set_calls
