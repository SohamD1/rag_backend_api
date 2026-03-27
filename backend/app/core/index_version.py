from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Dict

from app.config import Settings


def compute_index_version(*, settings: Settings, route: str) -> str:
    snapshot: Dict[str, Any] = {
        "schema": settings.index_schema_version,
        "route": route,
        "ocr": {
            "base_url": settings.mistral_base_url,
            "model": settings.mistral_ocr_model,
            "timeout_seconds": settings.mistral_ocr_timeout_seconds,
            "delete_uploaded_file": settings.mistral_ocr_delete_uploaded_file,
        },
        "embedding_model": settings.openai_embedding_model,
        "embedding_dimensions": settings.openai_embedding_dimensions,
        "tree_model": settings.openai_tree_model,
        "tree_reasoning_effort": getattr(settings, "openai_tree_reasoning_effort", "high"),
        "chunking": {
            "logic_version": settings.chunking_logic_version,
            "target": settings.chunk_target_tokens,
            "min": settings.chunk_min_tokens,
            "max": settings.chunk_max_tokens,
            "overlap": settings.chunk_overlap_tokens,
        },
        "tree": {
            "heading_preview_tokens": 250,
            "leaf_max_tokens": 900,
            "leaf_split_overlap_tokens": 80,
            "toc_check_page_num": settings.toc_check_page_num,
            "max_pages_per_node": settings.max_pages_per_node,
            "max_tokens_per_node": settings.max_tokens_per_node,
            "overlap_pages": settings.overlap_pages,
            "pageindex_toc_enabled": getattr(settings, "tree_pageindex_toc_enabled", True),
            "verify_toc_enabled": getattr(settings, "tree_verify_toc_enabled", True),
            "verify_toc_max_items": getattr(settings, "tree_verify_toc_max_items", 25),
            "fix_toc_enabled": getattr(settings, "tree_fix_toc_enabled", True),
            "fix_toc_max_attempts": getattr(settings, "tree_fix_toc_max_attempts", 2),
            "fix_toc_window_pages": getattr(settings, "tree_fix_toc_window_pages", 8),
            "split_large_nodes_enabled": getattr(settings, "tree_split_large_nodes_enabled", True),
            "split_large_nodes_max_depth": getattr(settings, "tree_split_large_nodes_max_depth", 2),
            "node_summaries_enabled": getattr(settings, "tree_node_summaries_enabled", True),
            "node_summary_max_tokens": getattr(settings, "tree_node_summary_max_tokens", 1200),
            "node_summary_token_threshold": getattr(settings, "tree_node_summary_token_threshold", 200),
            "node_summary_max_nodes": getattr(settings, "tree_node_summary_max_nodes", 200),
        },
    }
    raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(raw.encode("utf-8")).hexdigest()[:12]
