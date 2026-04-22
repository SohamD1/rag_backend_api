from __future__ import annotations

from dataclasses import replace

from app.config import settings
from app.core.doc_summaries import select_doc_ids_from_matches
from app.core.pipeline import run_chat, select_docs_with_rewrite_retry
from app.core.retrieval import RetrievalItem, retrieve_tree_for_doc
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


class SummaryRecordVectorStore:
    def __init__(self, records):
        self.records = dict(records)

    def fetch_records(self, *, ids, namespace):
        return {
            record_id: dict(self.records[record_id])
            for record_id in ids
            if record_id in self.records
        }


class NoopVectorStore:
    pass


class SummaryFetchGuardVectorStore:
    def fetch_records(self, *, ids, namespace):
        raise AssertionError("summary fetch should not run for strong doc selection")


class TreeQueryVectorStore:
    def __init__(self):
        self.calls = []

    def query(self, *, vector, top_k, namespace, filter=None):
        self.calls.append({"top_k": top_k, "namespace": namespace, "filter": dict(filter or {})})
        level = (filter or {}).get("level", {}).get("$eq")
        if level == "section":
            raise RuntimeError("section heading query failed")
        if level == "subsection":
            return [
                {
                    "id": "node-1",
                    "score": 0.82,
                    "metadata": {
                        "doc_id": "doc-tree",
                        "node_id": "node-1",
                        "section_id": "section-1",
                        "level": "subsection",
                        "title": "Coverage",
                    },
                }
            ]
        if level == "subsubsection":
            return []
        if level == "paragraph" and (filter or {}).get("section_id", {}).get("$eq") == "section-1":
            return [
                {
                    "id": "para-1",
                    "score": 0.77,
                    "metadata": {
                        "doc_id": "doc-tree",
                        "node_id": "para-1",
                        "text": "Coverage details",
                        "page_start": 2,
                        "page_end": 2,
                        "breadcrumb": "Coverage",
                    },
                }
            ]
        if level == "paragraph":
            return []
        raise AssertionError(f"unexpected filter: {filter}")


class TreeDefinitionBiasVectorStore:
    def query(self, *, vector, top_k, namespace, filter=None):
        level = (filter or {}).get("level", {}).get("$eq")
        if level == "section":
            return [
                {
                    "id": "sec-law",
                    "score": 0.86,
                    "metadata": {
                        "doc_id": "doc-tree",
                        "node_id": "sec-law",
                        "section_id": "sec-law",
                        "level": "section",
                        "title": "Current state of the law",
                        "summary": "This section explains the legal framework and current law.",
                        "page_start": 6,
                        "page_end": 8,
                    },
                }
            ]
        if level == "subsection":
            return [
                {
                    "id": "sub-definition",
                    "score": 0.72,
                    "metadata": {
                        "doc_id": "doc-tree",
                        "node_id": "sub-definition",
                        "section_id": "sec-overview",
                        "level": "subsection",
                        "title": "What are widgets?",
                        "summary": "Widgets fall into two categories and this section lists examples.",
                        "page_start": 3,
                        "page_end": 4,
                    },
                }
            ]
        if level == "subsubsection":
            return []
        if level == "paragraph":
            return []
        raise AssertionError(f"unexpected filter: {filter}")


class TreeNoParagraphsVectorStore:
    def query(self, *, vector, top_k, namespace, filter=None):
        level = (filter or {}).get("level", {}).get("$eq")
        if level == "section":
            return []
        if level == "subsection":
            return [
                {
                    "id": "sub-definition",
                    "score": 0.74,
                    "metadata": {
                        "doc_id": "doc-tree",
                        "node_id": "sub-definition",
                        "section_id": "sec-overview",
                        "level": "subsection",
                        "title": "What are widgets?",
                        "summary": "Widgets fall into two categories: local and hosted.",
                        "page_start": 3,
                        "page_end": 4,
                    },
                }
            ]
        if level == "subsubsection":
            return []
        if level == "paragraph":
            return []
        raise AssertionError(f"unexpected filter: {filter}")


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
    monkeypatch.setattr(pipeline, "generate_answer", lambda **kwargs: "")

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
    monkeypatch.setattr(pipeline, "generate_answer", lambda **kwargs: "")

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


def test_run_chat_hides_selected_doc_ids_without_debug(monkeypatch):
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
    monkeypatch.setattr(pipeline, "generate_answer", lambda **kwargs: "")

    payload = run_chat(
        query="what is useful",
        debug_enabled=False,
        settings=test_settings,
        registry=registry,
        vector_store=FetchingVectorStore({"doc1:c1": [0.1, 0.2]}),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["chunks"]
    assert payload["selected_doc_ids"] is None
    assert payload["debug"] is None


def test_doc_selection_aggregates_summary_kinds_and_initial_rewrite(monkeypatch):
    from app.core import pipeline

    test_settings = replace(settings, doc_rewrite_max_attempts=1)
    metas = [_meta("doc1"), _meta("doc2")]
    rewrite_embedding = [0.9, 0.1]

    monkeypatch.setattr(pipeline, "rewrite_query_for_doc_selection", lambda query, settings: "benefits enrollment")
    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, settings: [rewrite_embedding for _ in texts],
    )

    def fake_query_doc_summaries(**kwargs):
        if kwargs["query_embedding"] == rewrite_embedding:
            return [
                {
                    "id": "doc1:headings",
                    "score": 0.34,
                    "metadata": {"doc_id": "doc1", "summary_kind": "headings"},
                }
            ]
        return [
            {
                "id": "doc1:profile",
                "score": 0.37,
                "metadata": {"doc_id": "doc1", "summary_kind": "profile"},
            },
            {
                "id": "doc2:profile",
                "score": 0.36,
                "metadata": {"doc_id": "doc2", "summary_kind": "profile"},
            },
        ]

    monkeypatch.setattr(pipeline, "query_doc_summaries", fake_query_doc_summaries)

    selected_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query="what are the benefits",
        query_embedding=[0.1, 0.2],
        metas=metas,
        settings=test_settings,
        vector_store=NoopVectorStore(),
        debug={},
        allow_rewrite=True,
    )

    assert selected_ids == ["doc1"]
    assert strong is True
    assert retrieval_query == "benefits enrollment"
    assert retrieval_embedding == rewrite_embedding


def test_doc_selection_skips_summary_fetch_on_strong_match(monkeypatch):
    from app.core import pipeline

    test_settings = replace(settings, doc_rewrite_max_attempts=0)
    metas = [_meta("doc1"), _meta("doc2")]

    monkeypatch.setattr(
        pipeline,
        "query_doc_summaries",
        lambda **kwargs: [
            {
                "id": "doc1:profile",
                "score": 0.38,
                "metadata": {"doc_id": "doc1", "summary_kind": "profile"},
            },
            {
                "id": "doc1:keywords",
                "score": 0.37,
                "metadata": {"doc_id": "doc1", "summary_kind": "keywords"},
            },
            {
                "id": "doc2:profile",
                "score": 0.32,
                "metadata": {"doc_id": "doc2", "summary_kind": "profile"},
            },
        ],
    )

    selected_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query="benefits details",
        query_embedding=[0.1, 0.2],
        metas=metas,
        settings=test_settings,
        vector_store=SummaryFetchGuardVectorStore(),
        debug={},
        allow_rewrite=True,
    )

    assert selected_ids == ["doc1"]
    assert strong is True
    assert retrieval_query == "benefits details"
    assert retrieval_embedding == [0.1, 0.2]


def test_doc_selection_expands_to_multiple_docs_for_broad_strong_query(monkeypatch):
    from app.core import pipeline

    test_settings = replace(settings, doc_rewrite_max_attempts=0, doc_summary_top_k=3)
    metas = [_meta("doc1"), _meta("doc2"), _meta("doc3")]

    monkeypatch.setattr(
        pipeline,
        "query_doc_summaries",
        lambda **kwargs: [
            {
                "id": "doc1:profile",
                "score": 0.50,
                "metadata": {"doc_id": "doc1", "summary_kind": "profile"},
            },
            {
                "id": "doc1:headings",
                "score": 0.47,
                "metadata": {"doc_id": "doc1", "summary_kind": "headings"},
            },
            {
                "id": "doc2:profile",
                "score": 0.40,
                "metadata": {"doc_id": "doc2", "summary_kind": "profile"},
            },
            {
                "id": "doc2:headings",
                "score": 0.39,
                "metadata": {"doc_id": "doc2", "summary_kind": "headings"},
            },
            {
                "id": "doc3:profile",
                "score": 0.18,
                "metadata": {"doc_id": "doc3", "summary_kind": "profile"},
            },
        ],
    )

    selected_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query="What counts as digital assets? Give the definition, categories, and examples.",
        query_embedding=[0.1, 0.2],
        metas=metas,
        settings=test_settings,
        vector_store=NoopVectorStore(),
        debug={},
        allow_rewrite=True,
    )

    assert selected_ids == ["doc1", "doc2"]
    assert strong is True
    assert retrieval_query == "What counts as digital assets? Give the definition, categories, and examples."
    assert retrieval_embedding == [0.1, 0.2]


def test_doc_selection_can_win_on_keywords_signal(monkeypatch):
    from app.core import pipeline

    test_settings = replace(settings, doc_rewrite_max_attempts=0)
    metas = [_meta("doc1"), _meta("doc2")]

    monkeypatch.setattr(
        pipeline,
        "query_doc_summaries",
        lambda **kwargs: [
            {
                "id": "doc1:profile",
                "score": 0.35,
                "metadata": {"doc_id": "doc1", "summary_kind": "profile"},
            },
            {
                "id": "doc1:keywords",
                "score": 0.41,
                "metadata": {"doc_id": "doc1", "summary_kind": "keywords"},
            },
            {
                "id": "doc2:profile",
                "score": 0.36,
                "metadata": {"doc_id": "doc2", "summary_kind": "profile"},
            },
        ],
    )

    selected_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query="cobra continuation coverage",
        query_embedding=[0.1, 0.2],
        metas=metas,
        settings=test_settings,
        vector_store=NoopVectorStore(),
        debug={},
        allow_rewrite=True,
    )

    assert selected_ids == ["doc1"]
    assert strong is True
    assert retrieval_query == "cobra continuation coverage"
    assert retrieval_embedding == [0.1, 0.2]


def test_doc_selection_does_not_mark_aggregate_only_win_as_strong():
    selected_ids, strong, debug = select_doc_ids_from_matches(
        [
            {
                "id": "doc-digital:profile",
                "score": 0.380580872,
                "metadata": {"doc_id": "doc-digital", "summary_kind": "profile"},
            },
            {
                "id": "doc-digital:headings",
                "score": 0.372947812,
                "metadata": {"doc_id": "doc-digital", "summary_kind": "headings"},
            },
            {
                "id": "doc-tax:profile",
                "score": 0.385362387,
                "metadata": {"doc_id": "doc-tax", "summary_kind": "profile"},
            },
            {
                "id": "doc-tax:headings",
                "score": 0.240419984,
                "metadata": {"doc_id": "doc-tax", "summary_kind": "headings"},
            },
        ],
        settings=settings,
        top_k_fallback=3,
        lexical_scores={},
    )

    assert strong is False
    assert selected_ids[:2] == ["doc-digital", "doc-tax"]
    assert debug["ranked"][0]["doc_id"] == "doc-digital"
    assert debug["ranked"][0]["best_vector_score"] < debug["ranked"][1]["best_vector_score"]


def test_doc_selection_falls_back_to_lexical_metadata(monkeypatch):
    from app.core import pipeline

    test_settings = replace(settings, doc_rewrite_max_attempts=0)
    metas = [
        _meta("doc1"),
        replace(
            _meta("employee-handbook"),
            filename="Employee Handbook 2026.pdf",
            source_url="https://example.com/hr/employee-handbook-2026.pdf",
        ),
    ]

    monkeypatch.setattr(pipeline, "query_doc_summaries", lambda **kwargs: [])

    selected_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query="employee handbook",
        query_embedding=[0.1, 0.2],
        metas=metas,
        settings=test_settings,
        vector_store=NoopVectorStore(),
        debug={},
        allow_rewrite=True,
    )

    assert selected_ids == ["employee-handbook"]
    assert strong is False
    assert retrieval_query == "employee handbook"
    assert retrieval_embedding == [0.1, 0.2]


def test_doc_selection_lexical_fallback_uses_summary_text(monkeypatch):
    from app.core import pipeline

    test_settings = replace(settings, doc_rewrite_max_attempts=0)
    metas = [
        replace(
            _meta("doc1"),
            filename="Benefits Guide.pdf",
            source_url="https://example.com/hr/benefits-guide.pdf",
        ),
        replace(
            _meta("doc2"),
            filename="General Policies.pdf",
            source_url="https://example.com/hr/general-policies.pdf",
        ),
    ]

    monkeypatch.setattr(pipeline, "query_doc_summaries", lambda **kwargs: [])

    vector_store = SummaryRecordVectorStore(
        {
            "doc1:headings": {
                "id": "doc1:headings",
                "metadata": {
                    "doc_id": "doc1",
                    "summary_kind": "headings",
                    "summary_text": "Document headings Title: Benefits Guide Headings: COBRA enrollment and continuation coverage",
                },
            }
        }
    )

    selected_ids, strong, retrieval_query, retrieval_embedding = select_docs_with_rewrite_retry(
        query="cobra enrollment",
        query_embedding=[0.1, 0.2],
        metas=metas,
        settings=test_settings,
        vector_store=vector_store,
        debug={},
        allow_rewrite=True,
    )

    assert selected_ids == ["doc1"]
    assert strong is False
    assert retrieval_query == "cobra enrollment"
    assert retrieval_embedding == [0.1, 0.2]


def test_tree_retrieval_marks_partial_failures_as_degraded():
    debug = {}
    items = retrieve_tree_for_doc(
        doc_id="doc-tree",
        query="coverage details",
        query_embedding=[0.1, 0.2],
        settings=replace(settings, tree_node_selection_mode="vector_only", retrieve_top_k=3),
        vector_store=TreeQueryVectorStore(),
        top_k=3,
        debug=debug,
        tree_dir=None,
        index_version="ver1",
    )

    assert items
    assert debug["degraded"] is True
    assert debug["tree"]["degraded"] is True
    assert debug["tree"]["failures"][0]["stage"] == "query_headings"


def test_tree_retrieval_prioritizes_query_matching_candidates_for_llm(monkeypatch):
    captured = {}

    def fake_select_tree_nodes_with_llm(*, query, candidates, settings, top_n):
        captured["candidates"] = candidates
        return ["sub-definition"], "picked exact definition node"

    monkeypatch.setattr(
        "app.core.retrieval.select_tree_nodes_with_llm",
        fake_select_tree_nodes_with_llm,
    )

    items = retrieve_tree_for_doc(
        doc_id="doc-tree",
        query="widget definition categories examples",
        query_embedding=[0.1, 0.2],
        settings=replace(settings, tree_node_selection_mode="vector_then_llm", retrieve_top_k=3),
        vector_store=TreeDefinitionBiasVectorStore(),
        top_k=3,
        debug={},
        tree_dir=None,
        index_version="ver1",
    )

    assert captured["candidates"][0]["node_id"] == "sub-definition"
    assert items
    assert items[0].text.startswith("Widgets fall into two categories")


def test_tree_retrieval_includes_selected_node_summary_when_paragraphs_miss(monkeypatch):
    monkeypatch.setattr(
        "app.core.retrieval.select_tree_nodes_with_llm",
        lambda **kwargs: (["sub-definition"], "picked exact definition node"),
    )

    items = retrieve_tree_for_doc(
        doc_id="doc-tree",
        query="widget definition",
        query_embedding=[0.1, 0.2],
        settings=replace(settings, tree_node_selection_mode="vector_then_llm", retrieve_top_k=3),
        vector_store=TreeNoParagraphsVectorStore(),
        top_k=3,
        debug={},
        tree_dir=None,
        index_version="ver1",
    )

    assert items
    assert items[0].source_id == "sub-definition"
    assert "two categories" in items[0].text
    assert items[0].section_title == "What are widgets?"


def test_run_chat_does_not_cache_partial_tree_degradation(monkeypatch):
    from app.core import pipeline

    retrieval_cache = RecordingCache()
    response_cache = RecordingCache()
    registry = FakeRegistry([replace(_meta("doc-tree"), route="tree")])
    test_settings = replace(settings, rag_generate_answers_enabled=False)

    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, settings: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline,
        "select_docs_with_rewrite_retry",
        lambda **kwargs: (["doc-tree"], False, kwargs["query"], kwargs["query_embedding"]),
    )

    def fake_retrieve_tree_for_doc(**kwargs):
        kwargs["debug"]["degraded"] = True
        kwargs["debug"]["tree"] = {"mode": "vector_only", "degraded": True}
        return [
            RetrievalItem(
                source_id="doc-tree:c1",
                doc_id="doc-tree",
                filename="doc-tree.pdf",
                file_url=None,
                source_url="https://example.com/doc-tree.pdf",
                text="tree evidence",
                score=0.9,
                page_start=1,
                page_end=1,
                section_title="Intro",
                route="tree",
                vector_namespace="doc-tree::ver1",
            )
        ]

    monkeypatch.setattr(pipeline, "retrieve_tree_for_doc", fake_retrieve_tree_for_doc)
    monkeypatch.setattr(pipeline, "retrieve_standard_for_doc", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "select_context",
        lambda **kwargs: (
            kwargs["items"],
            [{"header": "doc=doc-tree pages=1-1", "text": "tree evidence"}],
        ),
    )

    payload = run_chat(
        query="show me tree evidence",
        debug_enabled=True,
        settings=test_settings,
        registry=registry,
        vector_store=FetchingVectorStore({"doc-tree:c1": [0.1, 0.2]}),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["chunks"]
    assert payload["debug"]["retrieval_degraded"]["stages"][0]["degraded_doc_ids"] == ["doc-tree"]
    assert retrieval_cache.set_calls == []
    assert response_cache.set_calls == []


def test_run_chat_reuses_initial_rewrite_for_low_confidence_retry(monkeypatch):
    from app.core import pipeline

    retrieval_cache = RecordingCache()
    response_cache = RecordingCache()
    registry = FakeRegistry([_meta("doc1")])
    test_settings = replace(
        settings,
        rag_generate_answers_enabled=False,
        doc_rewrite_max_attempts=1,
        low_confidence_threshold=0.95,
    )
    rewrite_calls = []
    raw_embedding = [0.1, 0.2]
    rewrite_embedding = [0.9, 0.1]

    def fake_rewrite(query, settings):
        rewrite_calls.append(query)
        return "benefits enrollment"

    def fake_embed_texts(texts, settings):
        return [rewrite_embedding if text == "benefits enrollment" else raw_embedding for text in texts]

    def fake_query_doc_summaries(**kwargs):
        embedding = kwargs["query_embedding"]
        if embedding == rewrite_embedding:
            return [
                {
                    "id": "doc1:profile",
                    "score": 0.39,
                    "metadata": {"doc_id": "doc1", "summary_kind": "profile"},
                }
            ]
        return [
            {
                "id": "doc1:profile",
                "score": 0.34,
                "metadata": {"doc_id": "doc1", "summary_kind": "profile"},
            }
        ]

    monkeypatch.setattr(pipeline, "rewrite_query_for_doc_selection", fake_rewrite)
    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(pipeline, "query_doc_summaries", fake_query_doc_summaries)
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
                score=0.2,
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

    payload = run_chat(
        query="what are the benefits",
        debug_enabled=True,
        settings=test_settings,
        registry=registry,
        vector_store=FetchingVectorStore({"doc1:c1": [0.1, 0.2]}),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["chunks"]
    assert rewrite_calls == ["what are the benefits"]
    assert payload["debug"]["rewrite_retry"]["reused_initial_rewrite"] is True


def test_run_chat_broadens_doc_scope_after_low_confidence_retrieval(monkeypatch):
    from app.core import pipeline

    retrieval_cache = RecordingCache()
    response_cache = RecordingCache()
    registry = FakeRegistry([_meta("doc1"), _meta("doc2"), _meta("doc3")])
    test_settings = replace(
        settings,
        rag_generate_answers_enabled=False,
        doc_rewrite_max_attempts=0,
        low_confidence_threshold=0.35,
    )

    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, settings: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline,
        "select_docs_with_rewrite_retry",
        lambda **kwargs: (["doc1"], True, kwargs["query"], kwargs["query_embedding"]),
    )

    def fake_retrieve_standard_for_doc(**kwargs):
        doc_id = kwargs["doc_id"]
        if doc_id == "doc1":
            return [
                RetrievalItem(
                    source_id="doc1:c1",
                    doc_id="doc1",
                    filename="doc1.pdf",
                    file_url=None,
                    source_url="https://example.com/doc1.pdf",
                    text="weak evidence",
                    score=0.2,
                    page_start=1,
                    page_end=1,
                    section_title="Intro",
                    route="standard",
                    vector_namespace="doc1::ver1",
                )
            ]
        if doc_id == "doc2":
            return [
                RetrievalItem(
                    source_id="doc2:c1",
                    doc_id="doc2",
                    filename="doc2.pdf",
                    file_url=None,
                    source_url="https://example.com/doc2.pdf",
                    text="strong evidence",
                    score=0.62,
                    page_start=2,
                    page_end=2,
                    section_title="Coverage",
                    route="standard",
                    vector_namespace="doc2::ver1",
                )
            ]
        return []

    monkeypatch.setattr(pipeline, "retrieve_standard_for_doc", fake_retrieve_standard_for_doc)
    monkeypatch.setattr(pipeline, "retrieve_tree_for_doc", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "select_context",
        lambda **kwargs: (
            kwargs["items"],
            [
                {"header": f"doc={item.doc_id} pages={item.page_start}-{item.page_end}", "text": item.text}
                for item in kwargs["items"]
            ],
        ),
    )

    payload = run_chat(
        query="coverage details",
        debug_enabled=True,
        settings=test_settings,
        registry=registry,
        vector_store=FetchingVectorStore(
            {
                "doc1:c1": [0.2, -0.9],
                "doc2:c1": [0.2, 0.9],
            }
        ),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["chunks"]
    assert payload["selected_doc_ids"] == ["doc1", "doc2"]
    assert payload["chunks"][0]["doc_id"] == "doc2"
    assert payload["debug"]["broad_retrieval_retry"]["applied"] is True


def test_run_chat_returns_chunk_level_citations_filtered_to_cited_docs(monkeypatch):
    from app.core import pipeline

    retrieval_cache = RecordingCache()
    response_cache = RecordingCache()
    registry = FakeRegistry([_meta("doc1"), _meta("doc2"), _meta("doc3")])
    test_settings = replace(settings, rag_generate_answers_enabled=True, context_use_mmr=False)

    monkeypatch.setattr(
        pipeline,
        "embed_texts",
        lambda texts, settings: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        pipeline,
        "select_docs_with_rewrite_retry",
        lambda **kwargs: (
            ["doc1", "doc2", "doc3"],
            False,
            kwargs["query"],
            kwargs["query_embedding"],
        ),
    )

    def fake_retrieve_standard_for_doc(**kwargs):
        doc_id = kwargs["doc_id"]
        if doc_id == "doc1":
            return [
                RetrievalItem(
                    source_id="doc1:c1",
                    doc_id="doc1",
                    filename="doc1.pdf",
                    file_url=None,
                    source_url="https://example.com/doc1.pdf",
                    text="doc1 evidence",
                    score=0.94,
                    page_start=2,
                    page_end=2,
                    section_title="Definition",
                    route="standard",
                    vector_namespace="doc1::ver1",
                ),
                RetrievalItem(
                    source_id="doc1:c2",
                    doc_id="doc1",
                    filename="doc1.pdf",
                    file_url=None,
                    source_url="https://example.com/doc1.pdf",
                    text="doc1 example",
                    score=0.90,
                    page_start=4,
                    page_end=4,
                    section_title="Examples",
                    route="standard",
                    vector_namespace="doc1::ver1",
                ),
            ]
        if doc_id == "doc2":
            return [
                RetrievalItem(
                    source_id="doc2:c1",
                    doc_id="doc2",
                    filename="doc2.pdf",
                    file_url=None,
                    source_url="https://example.com/doc2.pdf",
                    text="doc2 supporting evidence",
                    score=0.88,
                    page_start=8,
                    page_end=9,
                    section_title="Planning implications",
                    route="standard",
                    vector_namespace="doc2::ver1",
                )
            ]
        return [
            RetrievalItem(
                source_id="doc3:c1",
                doc_id="doc3",
                filename="doc3.pdf",
                file_url=None,
                source_url="https://example.com/doc3.pdf",
                text="doc3 unrelated evidence",
                score=0.60,
                page_start=10,
                page_end=10,
                section_title="Other topic",
                route="standard",
                vector_namespace="doc3::ver1",
            )
        ]

    monkeypatch.setattr(pipeline, "retrieve_standard_for_doc", fake_retrieve_standard_for_doc)
    monkeypatch.setattr(pipeline, "retrieve_tree_for_doc", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda **kwargs: "Digital assets include stored records and online accounts [1]. They also create planning challenges [2].",
    )

    payload = run_chat(
        query="What counts as digital assets and why do they matter?",
        debug_enabled=True,
        settings=test_settings,
        registry=registry,
        vector_store=FetchingVectorStore(
            {
                "doc1:c1": [0.1, 0.2],
                "doc1:c2": [0.2, 0.1],
                "doc2:c1": [0.3, 0.4],
                "doc3:c1": [0.4, 0.5],
            }
        ),
        retrieval_cache=retrieval_cache,
        response_cache=response_cache,
        tree_dir=None,
    )

    assert payload["answer"]
    assert {citation["doc_id"] for citation in payload["citations"]} == {"doc1", "doc2"}
    assert {citation["source_id"] for citation in payload["citations"]} == {"doc1:c1", "doc1:c2", "doc2:c1"}
    assert {
        (citation["source_id"], citation["page_start"], citation["page_end"])
        for citation in payload["citations"]
    } == {
        ("doc1:c1", 2, 2),
        ("doc1:c2", 4, 4),
        ("doc2:c1", 8, 9),
    }
