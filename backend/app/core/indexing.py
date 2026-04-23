from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

from app.adapters.embeddings import embed_texts
from app.adapters.pinecone_store import PineconeVectorStore
from app.config import Settings
from app.core.chunking import chunk_document
from app.core.doc_summaries import (
    build_doc_summary_texts,
    extract_heading_candidates_from_texts,
)
from app.core.pdf_text import PageText
from app.core.vector_namespace import vector_namespace
from app.services.tree_index import TreeNode, build_tree, save_headings, save_tree, trace_path


@dataclass(frozen=True)
class IndexBuildResult:
    indexed_count: int
    doc_summary_texts: Dict[str, str]
    namespace: str


def build_standard_index(
    *,
    doc_id: str,
    slug: str,
    filename: str,
    source_url: str,
    page_count: int,
    token_count: int,
    pages: List[PageText],
    settings: Settings,
    vector_store: PineconeVectorStore,
    index_version: str,
) -> IndexBuildResult:
    chunks = chunk_document(
        doc_id=doc_id,
        pages=pages,
        model=settings.openai_embedding_model,
        min_tokens=settings.chunk_min_tokens,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, settings)

    heading_candidates = extract_heading_candidates_from_texts([page.text for page in pages])
    doc_summary_texts = build_doc_summary_texts(
        slug=slug,
        filename=filename,
        source_url=source_url,
        route="standard",
        page_count=page_count,
        token_count=token_count,
        abstract_fragments=texts[:2],
        headings=heading_candidates,
    )

    namespace = vector_namespace(doc_id, index_version)
    items = []
    for chunk, emb in zip(chunks, embeddings):
        items.append(
            {
                "id": chunk.chunk_id,
                "values": emb,
                "metadata": {
                    "doc_id": doc_id,
                    "slug": slug,
                    "filename": filename,
                    "source_url": source_url,
                    "chunk_id": chunk.chunk_id,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_title": chunk.section_title,
                    "text": chunk.text,
                    "layout_mode": chunk.layout_mode,
                    "ocr_used": True,
                    "route": "standard",
                    "index_version": index_version,
                },
            }
        )

    vector_store.upsert(items, namespace=namespace)
    return IndexBuildResult(
        indexed_count=len(items),
        doc_summary_texts=doc_summary_texts,
        namespace=namespace,
    )


def build_tree_index(
    *,
    doc_id: str,
    slug: str,
    filename: str,
    source_url: str,
    page_count: int,
    token_count: int,
    pages: List[PageText],
    settings: Settings,
    vector_store: PineconeVectorStore,
    tree_dir: Path,
    index_version: str,
) -> IndexBuildResult:
    nodes = build_tree(doc_id, pages, settings)
    save_tree(doc_id, nodes, tree_dir, index_version=index_version)
    save_headings(doc_id, nodes, tree_dir, index_version=index_version)
    namespace = vector_namespace(doc_id, index_version)

    def breadcrumb_for(node_id: str) -> str:
        path = trace_path(nodes, node_id)
        titles = [n.title for n in path if n.level in {"section", "subsection", "subsubsection"} and n.title]
        return " > ".join(titles)

    def section_id_for(node_id: str) -> str:
        path = trace_path(nodes, node_id)
        for n in path:
            if n.level == "section":
                return n.node_id
        return f"{doc_id}:root"

    nodes_to_embed: List[TreeNode] = [
        n for n in nodes.values() if n.level not in {"root"} and (n.text_span or n.title)
    ]

    embed_inputs: List[str] = []
    metadatas: List[Dict] = []
    for node in nodes_to_embed:
        breadcrumb = breadcrumb_for(node.node_id)
        if node.level == "paragraph":
            embed_text = node.text_span or ""
        else:
            embed_text = (node.summary or "").strip() or node.title or node.text_span or ""
        if breadcrumb:
            embed_text = f"{breadcrumb}\n\n{embed_text}"
        embed_inputs.append(embed_text)
        metadatas.append(
            {
                "doc_id": node.doc_id,
                "slug": slug,
                "filename": filename,
                "source_url": source_url,
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "level": node.level,
                "title": node.title,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "text": node.text_span,
                "summary": (node.summary or "") if node.level != "paragraph" else "",
                "breadcrumb": breadcrumb,
                "section_id": section_id_for(node.node_id),
                "ocr_used": True,
                "route": "tree",
                "index_version": index_version,
            }
        )

    embeddings = embed_texts(embed_inputs, settings)

    heading_titles = [
        node.title
        for node in nodes_to_embed
        if node.level in {"section", "subsection", "subsubsection"} and node.title
    ]
    abstract_fragments = [
        (node.summary or "").strip() or (node.text_span or "").strip()
        for node in nodes_to_embed
        if node.level in {"section", "subsection", "subsubsection"}
    ]
    doc_summary_texts = build_doc_summary_texts(
        slug=slug,
        filename=filename,
        source_url=source_url,
        route="tree",
        page_count=page_count,
        token_count=token_count,
        abstract_fragments=abstract_fragments[:4],
        headings=heading_titles,
    )

    items = [
        {"id": node.node_id, "values": emb, "metadata": meta}
        for node, emb, meta in zip(nodes_to_embed, embeddings, metadatas)
    ]

    vector_store.upsert(items, namespace=namespace)
    return IndexBuildResult(
        indexed_count=len(items),
        doc_summary_texts=doc_summary_texts,
        namespace=namespace,
    )
