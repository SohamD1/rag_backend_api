from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional

from app.core.pdf_text import PageText
from app.core.tokens import estimate_tokens, get_token_encoder


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_start: int
    page_end: int
    section_title: Optional[str]
    layout_mode: str


@dataclass(frozen=True)
class _Block:
    text: str
    page_num: int
    layout_mode: str


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|[0-9]+[.)]|[A-Za-z][.)])\s+")


_HEADING_PREFIX_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[A-Z]|[IVXLC]+)[).:\-\s]+")


def _normalize(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines to a single paragraph break.
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw


def _clean_heading_candidate(text: str) -> Optional[str]:
    candidate = " ".join((text or "").split()).strip(" -:\t")
    if not candidate:
        return None
    if len(candidate) < 4 or len(candidate) > 120:
        return None
    if len(candidate.split()) > 14:
        return None
    if candidate.count("|") >= 2 or re.fullmatch(r"[\d\W]+", candidate):
        return None

    normalized = _HEADING_PREFIX_RE.sub("", candidate).strip()
    if len(normalized) < 4:
        return None
    if normalized.endswith((".", "!", "?")):
        return None

    alpha_chars = sum(1 for ch in normalized if ch.isalpha())
    if alpha_chars < max(3, len(normalized) // 4):
        return None

    words = normalized.split()
    title_ratio = sum(1 for word in words if word[:1].isupper()) / max(1, len(words))
    looks_heading = (
        normalized.isupper()
        or title_ratio >= 0.6
        or bool(re.match(r"^\d+(?:\.\d+)*\s+\S+", candidate))
        or candidate.endswith(":")
    )
    if not looks_heading:
        return None
    return normalized.rstrip(":")


def _should_preserve_line_breaks(text: str) -> bool:
    lines = [line.strip() for line in (text or "").split("\n") if line.strip()]
    if len(lines) < 3:
        return False

    short_lines = sum(1 for line in lines if len(line) <= 80)
    bullet_lines = sum(1 for line in lines if _BULLET_RE.match(line))
    tableish_lines = sum(
        1
        for line in lines
        if "|" in line
        or "\t" in line
        or re.search(r"\S\s{2,}\S", line)
    )
    short_ratio = short_lines / max(1, len(lines))
    return bullet_lines >= 2 or tableish_lines >= 2 or short_ratio >= 0.6


def _iter_paragraphs(page: PageText) -> Iterable[_Block]:
    normalized = _normalize(page.text)
    if _should_preserve_line_breaks(normalized):
        parts = [p.strip() for p in normalized.split("\n") if p.strip()]
        layout_mode = "preserve_lines"
    else:
        # PDFs often contain hard line breaks; treat single newlines as spaces while
        # preserving explicit paragraph breaks (blank lines).
        normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized)
        parts = [p.strip() for p in normalized.split("\n\n")]
        layout_mode = "flow_text"
    for part in parts:
        if not part:
            continue
        yield _Block(text=part, page_num=page.page_num, layout_mode=layout_mode)


def _split_by_token_window(
    text: str, *, model: str, max_tokens: int, overlap_tokens: int = 0
) -> List[str]:
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    overlap_tokens = max(0, min(int(overlap_tokens), max_tokens - 1))
    enc = get_token_encoder(model)
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + max_tokens)
        out.append(enc.decode(toks[start:end]).strip())
        if end >= len(toks):
            break
        next_start = end - overlap_tokens
        if next_start <= start:
            next_start = end
        start = next_start
    return [s for s in out if s]


def _split_oversized_block(
    block: _Block, *, model: str, max_tokens: int
) -> List[_Block]:
    """
    Deterministic fallback when a single paragraph exceeds max_tokens.
    Prefer sentence boundaries; fall back to token-window splits as needed.
    """
    text = block.text.strip()
    if not text:
        return []
    if estimate_tokens(text, model) <= max_tokens:
        return [block]

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        parts = _split_by_token_window(text, model=model, max_tokens=max_tokens)
        return [_Block(text=p, page_num=block.page_num, layout_mode=block.layout_mode) for p in parts]

    parts: List[str] = []
    buf: List[str] = []
    buf_tokens = 0
    for sent in sentences:
        st = estimate_tokens(sent, model)
        if st > max_tokens:
            # Flush what we have, then split this sentence by tokens.
            if buf:
                parts.append(" ".join(buf).strip())
                buf = []
                buf_tokens = 0
            for p in _split_by_token_window(sent, model=model, max_tokens=max_tokens):
                parts.append(p)
            continue

        if buf and buf_tokens + st > max_tokens:
            parts.append(" ".join(buf).strip())
            buf = []
            buf_tokens = 0
        buf.append(sent)
        buf_tokens += st

    if buf:
        parts.append(" ".join(buf).strip())

    return [_Block(text=p, page_num=block.page_num, layout_mode=block.layout_mode) for p in parts if p]


def chunk_document(
    *,
    doc_id: str,
    pages: List[PageText],
    model: str,
    min_tokens: int,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:
    """
    Deterministic "semantic" chunking with lightweight heading awareness:
    - paragraph blocks from explicit blank lines
    - pack full paragraphs into ~target_tokens, bounded by [min_tokens, max_tokens]
    - flush chunk boundaries when OCR exposes likely section headings
    - 10% overlap implemented as carrying forward the tail blocks until overlap_tokens is met
    """
    if min_tokens <= 0 or target_tokens <= 0 or max_tokens <= 0:
        raise ValueError("min/target/max tokens must be > 0")
    if not (min_tokens <= target_tokens <= max_tokens):
        raise ValueError("Expected min_tokens <= target_tokens <= max_tokens")
    overlap_tokens = max(0, min(int(overlap_tokens), max_tokens - 1))

    blocks: List[_Block] = []
    for page in pages:
        blocks.extend(list(_iter_paragraphs(page)))

    expanded: List[_Block] = []
    for block in blocks:
        expanded.extend(_split_oversized_block(block, model=model, max_tokens=max_tokens))

    def overlap_tail(prev: List[_Block]) -> List[_Block]:
        if overlap_tokens <= 0 or not prev:
            return []
        total = 0
        kept: List[_Block] = []
        for b in reversed(prev):
            kept.append(b)
            total += estimate_tokens(b.text, model)
            if total >= overlap_tokens:
                break
        kept = list(reversed(kept))
        # If overlap would keep the entire previous chunk, drop it to ensure progress.
        if len(kept) >= len(prev):
            return []
        return kept

    buf_is_overlap = False
    current_section_title: Optional[str] = None

    def flush_buf() -> None:
        nonlocal buf, buf_tokens, buf_is_overlap
        text = "\n\n".join(b.text for b in buf).strip()
        if text:
            page_start = buf[0].page_num
            page_end = buf[-1].page_num
            section_title = current_section_title or (
                f"Page {page_start}"
                if page_start == page_end
                else f"Pages {page_start}-{page_end}"
            )
            chunk_text = text
            if current_section_title and not text.lower().startswith(current_section_title.lower()):
                chunk_text = f"{current_section_title}\n\n{text}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:c{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=section_title,
                    layout_mode=buf[0].layout_mode if buf else "flow_text",
                )
            )
        buf = overlap_tail(buf)
        buf_tokens = sum(estimate_tokens(b.text, model) for b in buf)
        buf_is_overlap = bool(buf)

    chunks: List[Chunk] = []
    buf: List[_Block] = []
    buf_tokens = 0
    for block in expanded:
        heading = _clean_heading_candidate(block.text)
        if heading is not None:
            if buf and not buf_is_overlap:
                flush_buf()
            # Never carry overlap across an explicit heading boundary.
            buf = []
            buf_tokens = 0
            buf_is_overlap = False
            current_section_title = heading
            continue

        bt = estimate_tokens(block.text, model)
        if buf and buf_tokens + bt > max_tokens:
            if buf_is_overlap:
                # Don't emit an "overlap-only" chunk. If overlap can't fit with the
                # next block, drop overlap and continue.
                buf = []
                buf_tokens = 0
                buf_is_overlap = False
            else:
                flush_buf()
                # If overlap is too large to fit alongside the next block, drop it.
                if buf and buf_tokens + bt > max_tokens:
                    buf = []
                    buf_tokens = 0
                    buf_is_overlap = False

        buf.append(block)
        buf_tokens += bt
        buf_is_overlap = False

        # If we're within range, end near target for more consistent chunk sizes.
        if buf_tokens >= target_tokens and buf_tokens >= min_tokens:
            flush_buf()

    if buf:
        # Final flush without overlap (end of doc).
        text = "\n\n".join(b.text for b in buf).strip()
        if text:
            page_start = buf[0].page_num
            page_end = buf[-1].page_num
            section_title = current_section_title or (
                f"Page {page_start}"
                if page_start == page_end
                else f"Pages {page_start}-{page_end}"
            )
            chunk_text = text
            if current_section_title and not text.lower().startswith(current_section_title.lower()):
                chunk_text = f"{current_section_title}\n\n{text}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:c{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=section_title,
                    layout_mode=buf[0].layout_mode if buf else "flow_text",
                )
            )
    return chunks
