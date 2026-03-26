from __future__ import annotations


def vector_namespace(doc_id: str, index_version: str) -> str:
    """
    Keep each logical index version isolated so rebuilds can stage safely
    without destroying the currently live vectors for a document.
    """
    return f"{doc_id}::{index_version}"
