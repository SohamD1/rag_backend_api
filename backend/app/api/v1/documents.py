import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.pinecone_store import PineconeVectorStore
from app.api.schemas import DocumentCreateResponse, DocumentInfo, DocumentListResponse
from app.api.security import KBAdminAuthDep
from app.api.rate_limit import admin_rate_limit, make_rate_limit_dep
from app.config import (
    resolve_docs_dir,
    resolve_storage_dir,
    resolve_tree_dir,
    settings,
)
from app.core.index_version import compute_index_version
from app.core.indexing import build_standard_index, build_tree_index
from app.core.pdf_text import PageText, extract_pages
from app.core.tokens import estimate_tokens
from app.storage.local_store import save_upload
from app.storage.registry import DocMeta, DocRegistry, now_utc_iso
from app.adapters.mistral_ocr import MistralOcrError, ocr_pdf_to_pages


router = APIRouter()
logger = logging.getLogger(__name__)

DOCS_DIR = resolve_docs_dir(settings)
TREE_DIR = resolve_tree_dir(settings)
STORAGE_DIR = resolve_storage_dir(settings)

vector_store = PineconeVectorStore(settings)
registry = DocRegistry(DOCS_DIR)


def _file_url(doc_id: str) -> str:
    return f"/api/v1/documents/{doc_id}/file"


def _tree_path(doc_id: str, index_version: str) -> Path:
    return TREE_DIR / doc_id / index_version / "tree.json"


def _headings_path(doc_id: str, index_version: str) -> Path:
    return TREE_DIR / doc_id / index_version / "headings.json"

def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _should_ocr(pages: list[PageText]) -> bool:
    if not pages:
        return True
    min_chars = max(0, _int_env("KB_OCR_MIN_CHARS_PER_PAGE", 50))
    min_ratio = max(0.0, min(1.0, _float_env("KB_OCR_MIN_TEXT_PAGE_RATIO", 0.4)))

    text_pages = 0
    for p in pages:
        c = len("".join((p.text or "").split()))
        if c >= min_chars:
            text_pages += 1

    ratio = text_pages / max(1, len(pages))
    return ratio < min_ratio


def _has_mistral_key() -> bool:
    # Mistral OCR adapter reads the env var directly; keep behavior consistent here.
    return bool((os.getenv("MISTRAL_API_KEY") or "").strip())


@router.post(
    "/documents",
    response_model=DocumentCreateResponse,
    dependencies=[KBAdminAuthDep, Depends(make_rate_limit_dep(admin_rate_limit))],
)
def create_document(file: UploadFile = File(...), source_url: str = Form(...)):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    source_url = (source_url or "").strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="source_url is required.")

    try:
        stored = save_upload(file, STORAGE_DIR)
    except Exception as exc:
        logger.exception("upload_failed stage=save_upload")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    try:
        extracted = extract_pages(str(stored.path))
    except Exception as exc:
        logger.exception("upload_failed stage=extract_pages doc_id=%s", stored.doc_id)
        raise HTTPException(status_code=400, detail=f"Failed to read PDF text: {exc}") from exc

    ocr_mode = str(os.getenv("KB_OCR_MODE", "auto") or "auto").strip().lower()
    if ocr_mode not in {"auto", "always", "never"}:
        ocr_mode = "auto"

    pages = extracted
    use_ocr = bool(ocr_mode == "always" or (ocr_mode == "auto" and _should_ocr(extracted)))
    if use_ocr:
        if not _has_mistral_key():
            # In auto mode, prefer a soft-fail: accept whatever text extraction produced.
            # In always mode, the caller explicitly demanded OCR, so hard-fail.
            if ocr_mode == "always":
                raise HTTPException(
                    status_code=502,
                    detail="KB_OCR_MODE=always but MISTRAL_API_KEY is not set",
                )
            logger.warning(
                "ocr_skipped doc_id=%s reason=missing_mistral_api_key",
                stored.doc_id,
            )
            use_ocr = False
        else:
            try:
                pages = ocr_pdf_to_pages(pdf_path=stored.path, filename=stored.filename)
            except MistralOcrError as exc:
                logger.exception("upload_failed stage=ocr doc_id=%s", stored.doc_id)
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except Exception as exc:  # pragma: no cover
                logger.exception("upload_failed stage=ocr doc_id=%s", stored.doc_id)
                raise HTTPException(status_code=502, detail=f"Failed to OCR PDF: {exc}") from exc

    logger.info(
        "pdf_text_mode doc_id=%s mode=%s ocr=%s pages=%d",
        stored.doc_id,
        ocr_mode,
        bool(use_ocr),
        len(pages),
    )

    page_count = len(pages)
    token_count = sum(estimate_tokens(p.text, settings.openai_embedding_model) for p in pages)
    route = "tree" if page_count > settings.page_tree_threshold else "standard"
    index_version = compute_index_version(settings=settings, route=route)

    existing = registry.get(stored.doc_id)
    if existing and existing.index_version == index_version and existing.route == route:
        if route != "tree" or (
            _tree_path(stored.doc_id, index_version).exists()
            and _headings_path(stored.doc_id, index_version).exists()
        ):
            return DocumentCreateResponse(
                doc_id=existing.doc_id,
                slug=existing.slug,
                filename=existing.filename,
                source_url=existing.source_url,
                page_count=existing.page_count,
                token_count=existing.token_count,
                route=existing.route,
                index_version=existing.index_version,
                file_url=_file_url(existing.doc_id),
            )

    try:
        if route == "tree":
            indexed_count = build_tree_index(
                doc_id=stored.doc_id,
                slug=stored.slug,
                filename=stored.filename,
                source_url=source_url,
                page_count=page_count,
                token_count=token_count,
                pages=pages,
                settings=settings,
                vector_store=vector_store,
                tree_dir=TREE_DIR,
                index_version=index_version,
            )
        else:
            indexed_count = build_standard_index(
                doc_id=stored.doc_id,
                slug=stored.slug,
                filename=stored.filename,
                source_url=source_url,
                page_count=page_count,
                token_count=token_count,
                pages=pages,
                settings=settings,
                vector_store=vector_store,
                index_version=index_version,
            )
    except Exception as exc:
        logger.exception("upload_failed stage=index doc_id=%s route=%s", stored.doc_id, route)
        raise HTTPException(
            status_code=502,
            detail=f"Indexing failed ({route}). Check OpenAI/Pinecone connectivity: {exc}",
        ) from exc

    try:
        registry.save(
            DocMeta(
                doc_id=stored.doc_id,
                slug=stored.slug,
                filename=stored.filename,
                source_url=source_url,
                checksum=stored.checksum,
                size_bytes=stored.size_bytes,
                page_count=page_count,
                token_count=token_count,
                route=route,
                storage_path=str(stored.path),
                index_version=index_version,
                created_at=now_utc_iso(),
            )
        )
    except Exception as exc:
        logger.exception("upload_failed stage=registry_save doc_id=%s", stored.doc_id)
        raise HTTPException(status_code=500, detail=f"Failed to save document metadata: {exc}") from exc

    logger.info(
        "upload_complete doc_id=%s route=%s pages=%d est_tokens=%d indexed=%d",
        stored.doc_id,
        route,
        page_count,
        token_count,
        indexed_count,
    )

    return DocumentCreateResponse(
        doc_id=stored.doc_id,
        slug=stored.slug,
        filename=stored.filename,
        source_url=source_url,
        page_count=page_count,
        token_count=token_count,
        route=route,
        index_version=index_version,
        file_url=_file_url(stored.doc_id),
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[KBAdminAuthDep, Depends(make_rate_limit_dep(admin_rate_limit))],
)
def list_documents():
    metas = registry.list()
    docs = [
        DocumentInfo(
            doc_id=m.doc_id,
            slug=m.slug,
            filename=m.filename,
            source_url=m.source_url,
            page_count=m.page_count,
            token_count=m.token_count,
            route=m.route,
            index_version=m.index_version,
            file_url=_file_url(m.doc_id),
        )
        for m in metas
    ]
    docs = sorted(docs, key=lambda d: d.filename.lower())
    return DocumentListResponse(docs=docs)


@router.get(
    "/documents/{doc_id}/file",
    name="get_document_file",
    dependencies=[KBAdminAuthDep, Depends(make_rate_limit_dep(admin_rate_limit))],
)
def get_document_file(doc_id: str):
    meta = registry.get(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found.")
    path = Path(meta.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing from storage.")
    return FileResponse(path, media_type="application/pdf", filename=meta.filename)


@router.delete(
    "/documents/{doc_id}",
    dependencies=[KBAdminAuthDep, Depends(make_rate_limit_dep(admin_rate_limit))],
)
def delete_document(doc_id: str):
    meta = registry.get(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete vectors for the doc namespace.
    try:
        vector_store.clear_namespace(doc_id)
    except Exception:
        pass

    # Delete doc centroid in summary namespace.
    try:
        vector_store.delete_ids([doc_id], namespace=settings.doc_summary_namespace)
    except Exception:
        pass

    # Delete tree artifacts if present.
    if meta.route == "tree":
        tree_path = _tree_path(doc_id, meta.index_version)
        if tree_path.exists():
            tree_path.unlink()
        headings_path = _headings_path(doc_id, meta.index_version)
        if headings_path.exists():
            headings_path.unlink()
        # Best-effort cleanup of empty dirs.
        try:
            parent = tree_path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
            parent2 = parent.parent
            if parent2.exists() and not any(parent2.iterdir()):
                parent2.rmdir()
        except Exception:
            pass

    # Delete stored PDF.
    storage_path = Path(meta.storage_path)
    if storage_path.exists():
        storage_path.unlink()

    registry.delete(doc_id)
    return {"ok": True, "doc_id": doc_id}
