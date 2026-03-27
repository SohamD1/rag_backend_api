from __future__ import annotations

import logging
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.adapters.openai_client import chat_completions_create, get_openai_client
from app.config import Settings
from app.core.json_utils import extract_json_payload
from app.core.pdf_text import PageText
from app.core.tokens import estimate_tokens, get_token_encoder


HEADING_PREVIEW_TOKENS = 250
LEAF_MAX_TOKENS = 900
LEAF_SPLIT_OVERLAP_TOKENS = 80
TOC_PAGE_TEXT_MAX_TOKENS = 1200
TOC_ALIGN_TEXT_MAX_TOKENS = 8000
TOC_FIX_TEXT_MAX_TOKENS = 8000


logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@dataclass
class TreeNode:
    node_id: str
    doc_id: str
    parent_id: Optional[str]
    child_ids: List[str]
    level: str
    title: str
    text_span: str
    page_start: int
    page_end: int
    summary: str = ""


def _llm_chat(settings: Settings, prompt: str, *, max_tokens: int = 4096) -> str:
    response = chat_completions_create(
        settings,
        model=settings.openai_tree_model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=getattr(settings, "openai_tree_reasoning_effort", "high"),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _llm_json(
    settings: Settings,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> Any:
    response = chat_completions_create(
        settings,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        reasoning_effort=getattr(settings, "openai_tree_reasoning_effort", "high"),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    raw = (response.choices[0].message.content or "").strip()
    return extract_json_payload(raw)


def _convert_physical_index_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    m = re.search(r"<physical_index_(\d+)>", text)
    if m:
        return int(m.group(1))
    try:
        return int(text)
    except Exception:
        return None


def _tagged_page_text(page: PageText) -> str:
    idx = page.page_num
    return f"<physical_index_{idx}>\n{page.text or ''}\n<physical_index_{idx}>\n\n"


def _group_pages(
    pages: List[PageText],
    *,
    settings: Settings,
) -> List[str]:
    """
    Deterministic page grouping to keep LLM prompts bounded, with overlap by pages.
    """
    max_tokens = int(settings.max_tokens_per_node)
    max_pages = int(settings.max_pages_per_node)
    overlap_pages = max(0, int(settings.overlap_pages))
    if max_pages > 0:
        overlap_pages = min(overlap_pages, max_pages - 1)

    page_texts = [_tagged_page_text(p) for p in pages]
    token_lengths = [estimate_tokens(t, settings.openai_tree_model) for t in page_texts]

    groups: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for i, (t, tt) in enumerate(zip(page_texts, token_lengths)):
        if cur and (
            cur_tokens + tt > max_tokens or (max_pages > 0 and len(cur) >= max_pages)
        ):
            groups.append("".join(cur))
            # overlap by pages
            start = max(0, i - overlap_pages)
            cur = page_texts[start:i]
            cur_tokens = sum(token_lengths[start:i])

        cur.append(t)
        cur_tokens += tt

    if cur:
        groups.append("".join(cur))
    return groups


_TOC_TITLE_RE = re.compile(r"\b(table of contents|contents)\b", re.IGNORECASE)
_TOC_DOT_LEADER_RE = re.compile(r"\.{4,}\s*\d+\s*$", re.MULTILINE)


def _looks_like_toc(pages: List[PageText]) -> bool:
    text = "\n".join((p.text or "") for p in (pages or []))
    if not text.strip():
        return False
    if _TOC_TITLE_RE.search(text):
        return True
    if _TOC_DOT_LEADER_RE.search(text):
        return True
    return False


_TOC_DOTS_TO_COLON_RE = re.compile(r"(?:\.\s*){5,}")


def _transform_toc_text(text: str) -> str:
    # Make dot leaders easier for the LLM to parse.
    return _TOC_DOTS_TO_COLON_RE.sub(": ", text or "")


def _llm_is_toc_page(page: PageText, settings: Settings) -> bool:
    content = _truncate_tokens(
        page.text or "",
        model=settings.openai_tree_model,
        max_tokens=TOC_PAGE_TEXT_MAX_TOKENS,
    )
    if not content.strip():
        return False
    system_prompt = (
        "You detect whether a PDF page is part of the document's Table of Contents.\n"
        "Return STRICT JSON only.\n"
        'Schema: {"toc_detected": "yes"|"no"}\n'
        "Notes:\n"
        "- 'List of figures/tables', 'notation', 'abbreviations', and 'abstract' are NOT TOC.\n"
        "- A TOC page usually contains many section titles with page numbers.\n"
    )
    user_prompt = f"Page text:\n{content}\n\nJSON:"
    try:
        payload = _llm_json(
            settings,
            model=settings.openai_tree_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
        )
        detected = str((payload or {}).get("toc_detected") or "").strip().lower()
        return detected in {"yes", "y", "true", "1"}
    except Exception:
        # Conservative fallback.
        return False


def _find_toc_pages(pages: List[PageText], settings: Settings) -> List[int]:
    """
    Returns a list of *physical page numbers* (1-based) that appear to be TOC pages.
    """
    if not pages:
        return []
    max_check = max(0, int(settings.toc_check_page_num))
    if max_check <= 0:
        return []

    toc_pages: List[int] = []
    last_page_was_toc = False
    for page in pages[: min(max_check, len(pages))]:
        # Fast heuristic gate; still confirm with LLM when enabled.
        heur = _looks_like_toc([page])
        is_toc = False
        if getattr(settings, "tree_pageindex_toc_enabled", True):
            # If heuristic is strongly negative and we haven't started the TOC block,
            # we still ask the model (TOCs aren't always dot-leader formatted).
            is_toc = _llm_is_toc_page(page, settings)
        else:
            is_toc = heur

        if is_toc:
            toc_pages.append(int(page.page_num))
            last_page_was_toc = True
            continue

        if toc_pages and last_page_was_toc:
            # Stop at first non-TOC after entering the TOC block.
            break
        last_page_was_toc = False

    return toc_pages


def _llm_parse_toc_entries(toc_text: str, settings: Settings) -> List[Dict[str, Any]]:
    toc_text = _transform_toc_text(toc_text or "")
    toc_text = _truncate_tokens(
        toc_text,
        model=settings.openai_tree_model,
        max_tokens=TOC_ALIGN_TEXT_MAX_TOKENS,
    )
    if not toc_text.strip():
        return []

    system_prompt = (
        "You extract a Table of Contents into a flat JSON list in reading order.\n"
        "Return STRICT JSON only.\n"
        "Each item schema:\n"
        '{ "structure": "1"|"1.1"|null, "title": string, "page": int|null }\n'
        "Rules:\n"
        "- Preserve original title wording (fix spacing only).\n"
        "- If a numeric hierarchy label is not shown, set structure=null.\n"
        "- If an Arabic page number is shown, set page to that integer; otherwise page=null.\n"
        "- Exclude non-TOC lists (figures/tables) and boilerplate.\n"
    )
    user_prompt = f"TOC text:\n{toc_text}\n\nJSON:"
    try:
        payload = _llm_json(
            settings,
            model=settings.openai_tree_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=4096,
        )
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    out: List[Dict[str, Any]] = []
    top_level_counter = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        structure = item.get("structure")
        structure_s = str(structure or "").strip()
        page = item.get("page")
        page_i: Optional[int] = None
        if isinstance(page, int):
            page_i = page
        else:
            try:
                page_i = int(str(page).strip())
            except Exception:
                page_i = None

        if not structure_s:
            top_level_counter += 1
            structure_s = str(top_level_counter)

        out.append({"structure": structure_s, "title": title, "page": page_i})

    # De-dupe in order.
    seen: set[tuple[str, str, Optional[int]]] = set()
    deduped: List[Dict[str, Any]] = []
    for item in out:
        key = (item["structure"], item["title"], item.get("page"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _llm_extract_physical_indices_for_titles(
    *,
    toc_entries: List[Dict[str, Any]],
    tagged_pages_text: str,
    settings: Settings,
) -> List[Dict[str, Any]]:
    if not toc_entries or not tagged_pages_text.strip():
        return []
    tagged_pages_text = _truncate_tokens(
        tagged_pages_text,
        model=settings.openai_tree_model,
        max_tokens=TOC_ALIGN_TEXT_MAX_TOKENS,
    )
    # Keep prompt bounded.
    small = []
    for e in toc_entries[:60]:
        title = str(e.get("title") or "").strip()
        page = e.get("page")
        if not title or page is None:
            continue
        small.append({"title": title, "page": int(page)})
        if len(small) >= 40:
            break
    if not small:
        return []

    system_prompt = (
        "You align TOC printed page numbers to physical PDF pages.\n"
        "You are given TOC entries (title, printed page) and a partial document.\n"
        "The document contains tags like <physical_index_X> marking physical pages.\n"
        "Return STRICT JSON only as a list of matches for entries that START in the given pages.\n"
        'Schema: [{"title": string, "page": int, "physical_index": "<physical_index_X>"}]\n'
        "Do not hallucinate physical_index values; only use tags present in the text.\n"
    )
    user_prompt = (
        f"TOC entries:\n{json.dumps(small, ensure_ascii=True, indent=2)}\n\n"
        f"Document pages:\n{tagged_pages_text}\n\nJSON:"
    )
    try:
        payload = _llm_json(
            settings,
            model=settings.openai_tree_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048,
        )
    except Exception:
        return []

    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        page = item.get("page")
        physical = _convert_physical_index_to_int(item.get("physical_index"))
        try:
            page_i = int(page)
        except Exception:
            page_i = None
        if not title or page_i is None or physical is None:
            continue
        out.append({"title": title, "page": page_i, "physical_index": physical})
    return out


def _mode_int(values: List[int]) -> Optional[int]:
    if not values:
        return None
    c = Counter(values)
    best, best_count = None, 0
    for k, count in c.items():
        if count > best_count:
            best, best_count = k, count
        elif count == best_count and best is not None:
            # Tie-break: prefer smaller absolute offset.
            if abs(k) < abs(best):
                best = k
    return best


def _heuristic_appear_start(title: str, page_text: str) -> str:
    """
    Best-effort: "yes" if the title appears near the beginning of the page, else "no".
    """
    t = " ".join((title or "").strip().split()).lower()
    if not t:
        return "no"
    head = " ".join((page_text or "").strip().split()).lower()
    head = head[:800]
    if t in head[:200]:
        return "yes"
    # Allow numbered headings like "1. Title"
    if re.search(rf"(^|\\s)\\d+(?:\\.\\d+)*\\s*[:.\\-]?\\s*{re.escape(t)}", head[:250]):
        return "yes"
    return "no"


def _llm_check_title_on_page(
    *,
    title: str,
    page_text: str,
    settings: Settings,
) -> Tuple[bool, str]:
    """
    Returns (appears_or_starts, appear_start_yes_no).
    """
    content = _truncate_tokens(
        page_text or "",
        model=settings.openai_tree_model,
        max_tokens=TOC_PAGE_TEXT_MAX_TOKENS,
    )
    if not content.strip() or not (title or "").strip():
        return False, "no"

    system_prompt = (
        "You check whether a given section title appears or starts on a given PDF page.\n"
        "Also check whether the section starts at the beginning of the page (no other content before it).\n"
        "Return STRICT JSON only.\n"
        'Schema: {"answer":"yes"|"no","start_begin":"yes"|"no"}\n'
        "Use fuzzy matching; ignore spacing inconsistencies.\n"
    )
    user_prompt = f"Section title:\n{title}\n\nPage text:\n{content}\n\nJSON:"
    try:
        payload = _llm_json(
            settings,
            model=settings.openai_tree_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
        )
        ans = str((payload or {}).get("answer") or "").strip().lower()
        sb = str((payload or {}).get("start_begin") or "").strip().lower()
        ok = ans in {"yes", "y", "true", "1"}
        start_begin = "yes" if sb in {"yes", "y", "true", "1"} else "no"
        return ok, start_begin
    except Exception:
        return False, _heuristic_appear_start(title, page_text)


def _llm_find_title_start_page(
    *,
    title: str,
    tagged_pages_text: str,
    settings: Settings,
) -> Optional[int]:
    tagged_pages_text = _truncate_tokens(
        tagged_pages_text or "",
        model=settings.openai_tree_model,
        max_tokens=TOC_FIX_TEXT_MAX_TOKENS,
    )
    if not tagged_pages_text.strip() or not (title or "").strip():
        return None

    system_prompt = (
        "You locate where a section starts in a set of tagged PDF pages.\n"
        "The text contains tags like <physical_index_X>.\n"
        "Return STRICT JSON only.\n"
        'Schema: {"physical_index":"<physical_index_X>"|null}\n'
        "Only output a physical_index that appears in the provided text.\n"
    )
    user_prompt = (
        f"Section title:\n{title}\n\nTagged pages:\n{tagged_pages_text}\n\nJSON:"
    )
    try:
        payload = _llm_json(
            settings,
            model=settings.openai_tree_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=256,
        )
        return _convert_physical_index_to_int((payload or {}).get("physical_index"))
    except Exception:
        return None


def _depth_for_level(level: str) -> int:
    if level == "section":
        return 1
    if level == "subsection":
        return 2
    if level == "subsubsection":
        return 3
    return 0


def _is_heading(level: str) -> bool:
    return level in {"section", "subsection", "subsubsection"}


def _sorted_toc_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        items,
        key=lambda x: (
            int(x.get("physical_index") or 0),
            _structure_depth(str(x.get("structure") or "")),
            str(x.get("structure") or ""),
            str(x.get("title") or ""),
        ),
    )


def _ensure_preface(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = list(items or [])
    if not items:
        return items
    first = items[0].get("physical_index")
    try:
        first_i = int(first)
    except Exception:
        return items
    if first_i > 1:
        items.insert(
            0,
            {
                "structure": "0",
                "title": "Preface",
                "physical_index": 1,
                "appear_start": "yes",
            },
        )
    return items


def _verify_and_fix_toc_items(
    toc_items: List[Dict[str, Any]],
    pages: List[PageText],
    settings: Settings,
) -> List[Dict[str, Any]]:
    if not toc_items:
        return []

    page_count = len(pages)

    # Initialize appear_start heuristically for all items.
    for item in toc_items:
        phys = item.get("physical_index")
        try:
            p = int(phys)
        except Exception:
            item["appear_start"] = "no"
            continue
        if p < 1 or p > page_count:
            item["physical_index"] = None
            item["appear_start"] = "no"
            continue
        item["appear_start"] = _heuristic_appear_start(item.get("title", ""), pages[p - 1].text or "")

    if not getattr(settings, "tree_verify_toc_enabled", True):
        return toc_items

    max_items = max(0, int(getattr(settings, "tree_verify_toc_max_items", 25)))
    if max_items <= 0:
        return toc_items

    valid_idxs = [i for i, it in enumerate(toc_items) if isinstance(it.get("physical_index"), int)]
    if not valid_idxs:
        return toc_items

    # Deterministic evenly-spaced sample.
    if len(valid_idxs) <= max_items:
        sample = valid_idxs
    else:
        k = max_items
        n = len(valid_idxs)
        if k <= 1:
            sample = [valid_idxs[0]]
        else:
            sample = [valid_idxs[round(j * (n - 1) / (k - 1))] for j in range(k)]
        sample = sorted(set(sample))

    incorrect: List[int] = []
    for idx in sample:
        item = toc_items[idx]
        phys = int(item["physical_index"])
        ok, start_begin = _llm_check_title_on_page(
            title=str(item.get("title") or ""),
            page_text=pages[phys - 1].text or "",
            settings=settings,
        )
        if ok:
            item["appear_start"] = start_begin
        else:
            incorrect.append(idx)

    if not incorrect or not getattr(settings, "tree_fix_toc_enabled", True):
        return toc_items

    max_attempts = max(0, int(getattr(settings, "tree_fix_toc_max_attempts", 2)))
    window = max(1, int(getattr(settings, "tree_fix_toc_window_pages", 8)))
    if max_attempts <= 0:
        return toc_items

    for attempt in range(max_attempts):
        still_bad: List[int] = []
        for idx in incorrect:
            item = toc_items[idx]
            title = str(item.get("title") or "").strip()
            phys = item.get("physical_index")
            if not title or not isinstance(phys, int) or phys < 1 or phys > page_count:
                continue
            start = max(1, phys - window)
            end = min(page_count, phys + window)
            tagged = "".join(_tagged_page_text(p) for p in pages[start - 1 : end])
            found = _llm_find_title_start_page(title=title, tagged_pages_text=tagged, settings=settings)
            if found is not None and 1 <= found <= page_count:
                item["physical_index"] = int(found)
                item["appear_start"] = _heuristic_appear_start(title, pages[found - 1].text or "")
            else:
                still_bad.append(idx)
        incorrect = still_bad
        if not incorrect:
            break
        logger.info("tree_toc_fix attempt=%d remaining=%d", attempt + 1, len(incorrect))

    return toc_items


def _split_large_nodes_recursively(
    *,
    doc_id: str,
    nodes: Dict[str, TreeNode],
    pages: List[PageText],
    settings: Settings,
    counters: Dict[str, int],
    depth_remaining: int,
) -> None:
    if depth_remaining <= 0:
        return

    page_count = len(pages)
    max_pages = max(0, int(settings.max_pages_per_node))
    max_tokens = max(0, int(settings.max_tokens_per_node))
    if max_pages <= 0 or max_tokens <= 0:
        return

    # Precompute token counts per page once.
    page_tokens = [estimate_tokens(p.text or "", settings.openai_tree_model) for p in pages]

    # Find heading leaf nodes to consider (no heading children).
    heading_ids = [nid for nid, n in nodes.items() if _is_heading(n.level)]

    # Process in descending size so we split the biggest spans first.
    heading_ids.sort(
        key=lambda nid: (nodes[nid].page_end - nodes[nid].page_start, nodes[nid].page_start),
        reverse=True,
    )

    for nid in heading_ids:
        node = nodes.get(nid)
        if node is None:
            continue
        if not _is_heading(node.level):
            continue
        heading_children = []
        for cid in node.child_ids:
            child = nodes.get(cid)
            if child and _is_heading(child.level):
                heading_children.append(cid)
        if heading_children:
            continue

        span_pages = (node.page_end - node.page_start + 1) if node.page_end >= node.page_start else 0
        if span_pages <= max_pages:
            continue

        start = max(1, node.page_start)
        end = min(page_count, node.page_end)
        span_tokens = sum(page_tokens[start - 1 : end])
        if span_tokens < max_tokens:
            continue

        subset = pages[start - 1 : end]
        try:
            subtoc = _extract_toc_items(subset, settings)
        except Exception:
            continue
        if not subtoc:
            continue

        # Drop a self-referential first item if it matches this node.
        parent_title = " ".join((node.title or "").strip().split()).lower()
        filtered: List[Dict[str, Any]] = []
        for item in subtoc:
            t = " ".join((str(item.get("title") or "")).strip().split()).lower()
            phys = item.get("physical_index")
            if parent_title and t == parent_title and int(phys) == start:
                continue
            if isinstance(phys, int) and start <= phys <= end:
                filtered.append(item)
        subtoc = filtered
        if len(subtoc) < 2:
            continue

        # Create child heading nodes under this node.
        min_depth = min(_structure_depth(str(i.get("structure") or "")) for i in subtoc)
        parent_depth = _depth_for_level(node.level)
        stack: List[Tuple[str, int]] = [(nid, parent_depth)]
        created: List[Tuple[str, int, str]] = []

        for item in subtoc:
            structure = str(item.get("structure") or "").strip()
            depth = _structure_depth(structure)
            rel = max(1, depth - min_depth + 1)
            abs_depth = min(3, max(1, parent_depth + rel))
            level = _level_for_depth(abs_depth)

            counters[level] = counters.get(level, 0) + 1
            prefix = {"section": "sec", "subsection": "sub", "subsubsection": "subsub"}[level]
            child_id = f"{doc_id}:{prefix}:{counters[level]}"

            while stack and stack[-1][1] >= abs_depth:
                stack.pop()
            parent_id = stack[-1][0] if stack else nid

            title = str(item.get("title") or "").strip()
            pstart = int(item.get("physical_index") or start)
            child = TreeNode(
                node_id=child_id,
                doc_id=doc_id,
                parent_id=parent_id,
                child_ids=[],
                level=level,
                title=title,
                text_span="",
                page_start=pstart,
                page_end=pstart,
            )
            nodes[child_id] = child
            nodes[parent_id].child_ids.append(child_id)
            stack.append((child_id, abs_depth))

            appear = _heuristic_appear_start(title, pages[pstart - 1].text or "")
            created.append((child_id, abs_depth, appear))

        # Compute page_end for created nodes within the parent's range.
        for i, (child_id, abs_depth, _appear) in enumerate(created):
            next_start = None
            next_appear = "yes"
            for j in range(i + 1, len(created)):
                nid2, d2, ap2 = created[j]
                if d2 <= abs_depth:
                    next_start = nodes[nid2].page_start
                    next_appear = ap2
                    break
            if next_start is None:
                end_idx = end
            else:
                end_idx = next_start - 1
                if next_appear != "yes":
                    end_idx = next_start
            nodes[child_id].page_end = max(nodes[child_id].page_start, min(end_idx, end))

        # Recurse into newly created children.
        _split_large_nodes_recursively(
            doc_id=doc_id,
            nodes=nodes,
            pages=pages,
            settings=settings,
            counters=counters,
            depth_remaining=depth_remaining - 1,
        )


def _summarize_heading_nodes(
    *,
    nodes: Dict[str, TreeNode],
    pages: List[PageText],
    settings: Settings,
) -> None:
    if not getattr(settings, "tree_node_summaries_enabled", True):
        return
    max_workers = max(1, int(getattr(settings, "tree_node_summary_max_workers", 4)))
    max_tokens = max(100, int(getattr(settings, "tree_node_summary_max_tokens", 1200)))
    token_threshold = max(0, int(getattr(settings, "tree_node_summary_token_threshold", 200)))

    page_count = len(pages)

    def node_text(n: TreeNode) -> str:
        start = max(1, min(page_count, int(n.page_start)))
        end = max(1, min(page_count, int(n.page_end)))
        if end < start:
            start, end = end, start
        raw = "\n\n".join((pages[i - 1].text or "") for i in range(start, end + 1))
        return _truncate_tokens(raw, model=settings.openai_tree_model, max_tokens=max_tokens)

    def summarize_one(nid: str) -> Tuple[str, str]:
        n = nodes[nid]
        text = node_text(n)
        if not text.strip():
            return nid, ""
        if estimate_tokens(text, settings.openai_tree_model) <= token_threshold:
            return nid, text.strip()
        system_prompt = (
            "You summarize document sections.\n"
            "Return a concise, factual summary (1-3 sentences) of the main points.\n"
            "Do not add facts not present in the text.\n"
        )
        user_prompt = f"Section title:\n{n.title}\n\nSection text:\n{text}\n\nSummary:"
        try:
            response = chat_completions_create(
                settings,
                model=settings.openai_tree_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                reasoning_effort=getattr(settings, "openai_tree_reasoning_effort", "high"),
                temperature=0.0,
                max_tokens=256,
            )
            summary = (response.choices[0].message.content or "").strip()
            return nid, summary
        except Exception:
            return nid, ""

    heading_ids = [nid for nid, n in nodes.items() if _is_heading(n.level) and n.title]
    # More important nodes first.
    heading_ids.sort(key=lambda nid: (nodes[nid].page_end - nodes[nid].page_start, nid), reverse=True)
    max_nodes = max(0, int(getattr(settings, "tree_node_summary_max_nodes", 200)))
    if max_nodes > 0:
        heading_ids = heading_ids[:max_nodes]

    # Keep concurrency conservative; OpenAI rate limits are environment-dependent.
    if max_workers <= 1 or len(heading_ids) <= 1:
        for nid in heading_ids:
            _, summary = summarize_one(nid)
            nodes[nid].summary = summary
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=min(max_workers, 6)) as executor:
        futures = {executor.submit(summarize_one, nid): nid for nid in heading_ids}
        for future in as_completed(futures):
            nid = futures[future]
            try:
                _nid, summary = future.result()
            except Exception:
                continue
            nodes[nid].summary = summary


def _extract_toc_items(pages: List[PageText], settings: Settings) -> List[Dict[str, Any]]:
    groups = _group_pages(pages, settings=settings)
    toc_items: List[Dict[str, Any]] = []
    if groups:
        toc_items = _generate_toc_init(groups[0], settings)
        for part in groups[1:]:
            additional = _generate_toc_continue(toc_items, part, settings)
            toc_items.extend(additional)
    toc_items = _dedupe_toc_items(toc_items)
    # Ensure items are ordered by physical index; stable tie-break by structure/title.
    return sorted(
        toc_items,
        key=lambda x: (
            int(x["physical_index"]),
            _structure_depth(x["structure"]),
            x["structure"],
            x["title"],
        ),
    )


def _extract_json_list(text: str) -> List[Dict[str, Any]]:
    payload = extract_json_payload(text)
    if isinstance(payload, dict) and "table_of_contents" in payload:
        payload = payload["table_of_contents"]
    if not isinstance(payload, list):
        raise ValueError("Expected JSON list")
    out: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def _generate_toc_init(part: str, settings: Settings) -> List[Dict[str, Any]]:
    prompt = (
        "You are an expert at extracting a hierarchical table of contents from long documents.\n\n"
        "You will be given a partial document with physical page tags like <physical_index_X>.\n"
        "Your task: generate the tree structure of the document as a FLAT LIST in reading order.\n\n"
        "Rules:\n"
        "- structure is a numeric hierarchy string: 1, 1.1, 1.2, 2, 2.1, ...\n"
        "- title must preserve the original wording (fix spacing only)\n"
        "- physical_index is the tag <physical_index_X> for where the section starts\n"
        "- Only include real sections (exclude abstract, acknowledgements, etc. unless they are clearly a main section)\n\n"
        "Return STRICT JSON as a list:\n"
        "[\n"
        "  {\"structure\": \"1\", \"title\": \"...\", \"physical_index\": \"<physical_index_12>\"},\n"
        "  {\"structure\": \"1.1\", \"title\": \"...\", \"physical_index\": \"<physical_index_14>\"}\n"
        "]\n\n"
        f"Document:\n{part}\n\nJSON:"
    )
    return _extract_json_list(_llm_chat(settings, prompt))


def _generate_toc_continue(
    previous: List[Dict[str, Any]],
    part: str,
    settings: Settings,
) -> List[Dict[str, Any]]:
    prompt = (
        "You are an expert at extracting a hierarchical table of contents from long documents.\n\n"
        "You are given:\n"
        "1) a flat JSON list representing the TOC extracted so far\n"
        "2) the next partial document chunk (with <physical_index_X> tags)\n\n"
        "Your task: output ONLY the additional TOC entries that appear in the new part.\n"
        "Do NOT repeat entries that were already listed.\n"
        "Keep structure numbering consistent and in reading order.\n\n"
        "Return STRICT JSON as a list:\n"
        "[{\"structure\":\"...\",\"title\":\"...\",\"physical_index\":\"<physical_index_X>\"}]\n\n"
        f"Previous TOC:\n{json.dumps(previous, indent=2)}\n\n"
        f"Document:\n{part}\n\nJSON:"
    )
    return _extract_json_list(_llm_chat(settings, prompt))


def _dedupe_toc_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, str, Optional[int]]] = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        structure = str(item.get("structure") or "").strip()
        title = str(item.get("title") or "").strip()
        physical = _convert_physical_index_to_int(item.get("physical_index"))
        key = (structure, title, physical)
        if key in seen:
            continue
        seen.add(key)
        if not structure or not title or physical is None:
            continue
        out.append({"structure": structure, "title": title, "physical_index": physical})
    # Preserve reading order as provided.
    return out


def _structure_depth(structure: str) -> int:
    if not structure:
        return 1
    parts = [p for p in str(structure).split(".") if p.strip()]
    return max(1, len(parts))


def _level_for_depth(depth: int) -> str:
    if depth <= 1:
        return "section"
    if depth == 2:
        return "subsection"
    return "subsubsection"


def _truncate_tokens(text: str, *, model: str, max_tokens: int) -> str:
    if not text:
        return ""
    enc = get_token_encoder(model)
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text.strip()
    return enc.decode(toks[:max_tokens]).strip()


def _normalize_heading_match_text(text: str) -> str:
    text = " ".join((text or "").replace("\r", " ").replace("\n", " ").split()).lower()
    if not text:
        return ""
    # Strip common numbered heading prefixes so "1.2 Title" can match "Title".
    text = re.sub(r"^\s*(?:\(?[ivxlcdm]+\)?|\d+(?:\.\d+)*)(?:[\)\].:-]|\s)+", "", text)
    text = re.sub(r"^\s*(?:chapter|section|part)\s+\d+(?:\.\d+)*\s*[:.\-]?\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _paragraph_starts_heading(title: str, paragraph: str) -> bool:
    title_norm = _normalize_heading_match_text(title)
    para_norm = _normalize_heading_match_text(paragraph)
    if not title_norm or not para_norm:
        return False
    return para_norm == title_norm or para_norm.startswith(f"{title_norm} ")


def build_tree(doc_id: str, pages: List[PageText], settings: Settings) -> Dict[str, TreeNode]:
    """
    PageIndex-style: build a hierarchical TOC (via LLM, temp=0) then create paragraph leaves.
    Deterministic grouping/limits; no randomness.
    """
    if not pages:
        raise ValueError("No pages to index")

    page_count = len(pages)
    toc_check_page_num = max(0, int(settings.toc_check_page_num))

    toc_items: List[Dict[str, Any]] = []
    toc_pages = _find_toc_pages(pages, settings) if toc_check_page_num > 0 else []
    if toc_pages:
        toc_text = "\n\n".join((pages[p - 1].text or "") for p in toc_pages if 1 <= p <= page_count)
        parsed = _llm_parse_toc_entries(toc_text, settings)
        has_printed_pages = any(e.get("page") is not None for e in parsed)

        if has_printed_pages:
            last_toc = max(toc_pages)
            # Use a small window after the TOC to align printed pages to physical pages.
            start = min(page_count, last_toc + 1)
            end = min(page_count, start + max(1, toc_check_page_num) - 1)
            tagged = "".join(_tagged_page_text(p) for p in pages[start - 1 : end])
            alignments = _llm_extract_physical_indices_for_titles(
                toc_entries=parsed,
                tagged_pages_text=tagged,
                settings=settings,
            )
            diffs = [int(a["physical_index"]) - int(a["page"]) for a in alignments]
            offset = _mode_int(diffs)
            if offset is None:
                offset = 0
            for e in parsed:
                page_num = e.get("page")
                if page_num is None:
                    continue
                phys = int(page_num) + int(offset)
                if 1 <= phys <= page_count:
                    toc_items.append(
                        {
                            "structure": e.get("structure") or "",
                            "title": e.get("title") or "",
                            "physical_index": phys,
                        }
                    )

        # Fallback: build TOC directly from the document if TOC parsing/alignment failed.
        if not toc_items:
            toc_items = _extract_toc_items(pages, settings)
    else:
        # No TOC found; build TOC directly from the document.
        toc_items = _extract_toc_items(pages, settings)

    toc_items = _sorted_toc_items(toc_items)
    toc_items = _ensure_preface(toc_items)
    toc_items = _verify_and_fix_toc_items(toc_items, pages, settings)
    # Re-sort after potential fixes.
    toc_items = _sorted_toc_items([t for t in toc_items if isinstance(t.get("physical_index"), int)])

    root_id = f"{doc_id}:root"
    nodes: Dict[str, TreeNode] = {
        root_id: TreeNode(
            node_id=root_id,
            doc_id=doc_id,
            parent_id=None,
            child_ids=[],
            level="root",
            title="Root",
            text_span="",
            page_start=1,
            page_end=len(pages),
        )
    }

    # Create heading nodes in reading order.
    stack: List[str] = [root_id]  # node_id per depth, root at depth 0
    heading_order: List[Tuple[str, int, str]] = []
    counters = {"section": 0, "subsection": 0, "subsubsection": 0}
    for item in toc_items:
        structure = item["structure"]
        depth = min(3, _structure_depth(structure))
        level = _level_for_depth(depth)
        counters[level] += 1
        prefix = {"section": "sec", "subsection": "sub", "subsubsection": "subsub"}[level]
        node_id = f"{doc_id}:{prefix}:{counters[level]}"

        while len(stack) > depth:
            stack.pop()
        parent_id = stack[-1] if stack else root_id
        stack.append(node_id)

        node = TreeNode(
            node_id=node_id,
            doc_id=doc_id,
            parent_id=parent_id,
            child_ids=[],
            level=level,
            title=item["title"],
            text_span="",
            page_start=int(item["physical_index"]),
            page_end=int(item["physical_index"]),
        )
        nodes[node_id] = node
        nodes[parent_id].child_ids.append(node_id)
        heading_order.append((node_id, depth, str(item.get("appear_start") or "no")))

    # Compute page_end for headings by next heading at same-or-higher depth.
    last_page = len(pages)
    appear_by_id = {nid: ap for nid, _d, ap in heading_order}
    for i, (node_id, depth, _appear) in enumerate(heading_order):
        next_start = None
        next_appear = "yes"
        for j in range(i + 1, len(heading_order)):
            next_id, next_depth, _ap2 = heading_order[j]
            if next_depth <= depth:
                next_start = nodes[next_id].page_start
                next_appear = appear_by_id.get(next_id, "yes")
                break
        end = (next_start - 1) if next_start is not None else last_page
        if next_start is not None and next_appear != "yes":
            end = next_start
        nodes[node_id].page_end = max(nodes[node_id].page_start, min(end, last_page))

    # Optionally split oversized nodes (PageIndex-style recursive refinement).
    if getattr(settings, "tree_split_large_nodes_enabled", True):
        _split_large_nodes_recursively(
            doc_id=doc_id,
            nodes=nodes,
            pages=pages,
            settings=settings,
            counters=counters,
            depth_remaining=max(0, int(getattr(settings, "tree_split_large_nodes_max_depth", 2))),
        )

    # Add a deterministic preview text_span for heading nodes (for embeddings).
    for n in nodes.values():
        if not _is_heading(n.level):
            continue
        page_text = pages[max(0, n.page_start - 1)].text or ""
        n.text_span = _truncate_tokens(
            page_text,
            model=settings.openai_embedding_model,
            max_tokens=HEADING_PREVIEW_TOKENS,
        )

    # Generate summaries for heading nodes (used for retrieval + embeddings).
    _summarize_heading_nodes(nodes=nodes, pages=pages, settings=settings)

    # Precompute deepest heading per page and section ancestor per heading.
    heading_nodes = [(nid, _depth_for_level(nodes[nid].level)) for nid in nodes if _is_heading(nodes[nid].level)]
    heading_nodes = sorted(
        heading_nodes,
        key=lambda x: (nodes[x[0]].page_start, x[1], x[0]),
    )
    heading_order_index = {nid: idx for idx, (nid, _depth) in enumerate(heading_nodes)}

    def section_ancestor(nid: str) -> str:
        cur = nodes.get(nid)
        while cur and cur.parent_id is not None:
            if cur.level == "section":
                return cur.node_id
            cur = nodes.get(cur.parent_id)
        return root_id

    section_id_by_heading: Dict[str, str] = {
        nid: section_ancestor(nid) for nid, _ in heading_nodes
    }

    page_heading_candidates: Dict[int, List[str]] = {}
    for nid, _depth in heading_nodes:
        page_num = nodes[nid].page_start
        page_heading_candidates.setdefault(page_num, []).append(nid)
    for page_num, candidates in page_heading_candidates.items():
        candidates.sort(
            key=lambda nid: (
                -_depth_for_level(nodes[nid].level),
                heading_order_index.get(nid, 0),
                nid,
            )
        )

    deepest_by_page: Dict[int, str] = {}
    for page_num in range(1, last_page + 1):
        best_id = root_id
        best_depth = 0
        for nid, depth in heading_nodes:
            n = nodes[nid]
            if n.page_start <= page_num <= n.page_end:
                if depth >= best_depth:
                    best_id = nid
                    best_depth = depth
        deepest_by_page[page_num] = best_id

    # Create paragraph leaves.
    para_counters: Dict[int, int] = {}
    for page in pages:
        active_heading = deepest_by_page.get(page.page_num, root_id)
        section_id = section_id_by_heading.get(active_heading, root_id)
        candidates = page_heading_candidates.get(page.page_num, [])

        normalized = (page.text or "").replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
        if not paragraphs:
            continue

        for para in paragraphs:
            for candidate in candidates:
                if _paragraph_starts_heading(nodes[candidate].title, para):
                    active_heading = candidate
                    section_id = section_id_by_heading.get(active_heading, root_id)
                    break

            parts = [para]
            if estimate_tokens(para, settings.openai_embedding_model) > LEAF_MAX_TOKENS:
                # Split oversized paragraphs by tokens with overlap.
                enc = get_token_encoder(settings.openai_embedding_model)
                toks = enc.encode(para)
                parts = []
                start = 0
                while start < len(toks):
                    end = min(len(toks), start + LEAF_MAX_TOKENS)
                    parts.append(enc.decode(toks[start:end]).strip())
                    if end >= len(toks):
                        break
                    next_start = end - LEAF_SPLIT_OVERLAP_TOKENS
                    if next_start <= start:
                        next_start = end
                    start = next_start

            for part in parts:
                para_counters[page.page_num] = para_counters.get(page.page_num, 0) + 1
                leaf_id = f"{doc_id}:p:{page.page_num}:{para_counters[page.page_num]}"
                leaf = TreeNode(
                    node_id=leaf_id,
                    doc_id=doc_id,
                    parent_id=active_heading,
                    child_ids=[],
                    level="paragraph",
                    title="",
                    text_span=part,
                    page_start=page.page_num,
                    page_end=page.page_num,
                )
                nodes[leaf_id] = leaf
                nodes[active_heading].child_ids.append(leaf_id)

                # Store section_id on the node itself for later metadata usage (not part of TreeNode schema).
                # We encode it in the node_id format via indexing metadata instead.
                _ = section_id

    return nodes


def save_tree(doc_id: str, nodes: Dict[str, TreeNode], base_dir: Path, index_version: str) -> Path:
    out_dir = base_dir / doc_id / index_version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tree.json"
    data = {node_id: asdict(node) for node_id, node in nodes.items()}
    _atomic_write_json(path, data)
    return path


def save_headings(doc_id: str, nodes: Dict[str, TreeNode], base_dir: Path, index_version: str) -> Path:
    """
    Save a lightweight headings-only artifact for LLM node selection at retrieval time.
    """
    out_dir = base_dir / doc_id / index_version
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "headings.json"

    def section_ancestor(nid: str) -> str:
        cur = nodes.get(nid)
        while cur and cur.parent_id is not None:
            if cur.level == "section":
                return cur.node_id
            cur = nodes.get(cur.parent_id)
        return f"{doc_id}:root"

    data: Dict[str, Any] = {}
    for nid, n in nodes.items():
        if not _is_heading(n.level):
            continue
        heading_child_ids = [cid for cid in (n.child_ids or []) if _is_heading(nodes.get(cid).level if nodes.get(cid) else "")]
        data[nid] = {
            "node_id": n.node_id,
            "parent_id": n.parent_id,
            "child_ids": heading_child_ids,
            "level": n.level,
            "title": n.title,
            "page_start": n.page_start,
            "page_end": n.page_end,
            "summary": n.summary,
            "section_id": section_ancestor(nid),
        }

    _atomic_write_json(path, data)
    return path


def load_headings(doc_id: str, base_dir: Path, index_version: str) -> Dict[str, Dict[str, Any]]:
    path = base_dir / doc_id / index_version / "headings.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(data, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for nid, payload in data.items():
            if isinstance(payload, dict):
                out[str(nid)] = payload
        return out
    return {}


def load_tree(doc_id: str, base_dir: Path, index_version: str) -> Dict[str, TreeNode]:
    path = base_dir / doc_id / index_version / "tree.json"
    data = json.loads(path.read_text())
    return {node_id: TreeNode(**payload) for node_id, payload in data.items()}


def trace_path(nodes: Dict[str, TreeNode], node_id: str) -> List[TreeNode]:
    path: List[TreeNode] = []
    cur = nodes.get(node_id)
    while cur:
        path.append(cur)
        if cur.parent_id is None:
            break
        cur = nodes.get(cur.parent_id)
    return list(reversed(path))
