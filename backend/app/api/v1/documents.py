import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.pinecone_store import PineconeVectorStore
from app.api.deps import get_vector_store
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
from app.core.tokens import estimate_tokens
from app.storage.local_store import save_upload
from app.storage.registry import DocMeta, DocRegistry, now_utc_iso
from app.adapters.mistral_ocr import MistralOcrError, ocr_pdf_to_pages


router = APIRouter()
logger = logging.getLogger(__name__)

DOCS_DIR = resolve_docs_dir(settings)
TREE_DIR = resolve_tree_dir(settings)
STORAGE_DIR = resolve_storage_dir(settings)

registry = DocRegistry(DOCS_DIR)


@dataclass
class IngestState:
    doc_id: str | None = None
    storage_path: Path | None = None
    storage_created: bool = False
    had_existing_doc: bool = False
    route: str | None = None
    index_version: str | None = None
    target_tree_artifacts_existed: bool = False
    registry_saved: bool = False
    stage: str = "init"


def _file_url(doc_id: str) -> str:
    return f"/api/v1/documents/{doc_id}/file"


def _tree_path(doc_id: str, index_version: str) -> Path:
    return TREE_DIR / doc_id / index_version / "tree.json"


def _headings_path(doc_id: str, index_version: str) -> Path:
    return TREE_DIR / doc_id / index_version / "headings.json"


def _delete_tree_artifacts(doc_id: str, index_version: str) -> None:
    tree_path = _tree_path(doc_id, index_version)
    if tree_path.exists():
        tree_path.unlink()

    headings_path = _headings_path(doc_id, index_version)
    if headings_path.exists():
        headings_path.unlink()

    parent = tree_path.parent
    if parent.exists() and not any(parent.iterdir()):
        parent.rmdir()
    parent2 = parent.parent
    if parent2.exists() and not any(parent2.iterdir()):
        parent2.rmdir()


def _cleanup_failed_ingest(
    *,
    state: IngestState,
    vector_store: PineconeVectorStore | None,
) -> None:
    if not state.doc_id:
        return

    logger.warning(
        "ingest_cleanup_started doc_id=%s stage=%s route=%s index_version=%s",
        state.doc_id,
        state.stage,
        state.route,
        state.index_version,
    )

    if vector_store is not None and not state.had_existing_doc:
        try:
            vector_store.clear_namespace(state.doc_id)
        except Exception:
            logger.exception(
                "ingest_cleanup_failed step=clear_namespace doc_id=%s",
                state.doc_id,
            )

        try:
            vector_store.delete_ids([state.doc_id], namespace=settings.doc_summary_namespace)
        except Exception:
            logger.exception(
                "ingest_cleanup_failed step=delete_doc_summary doc_id=%s",
                state.doc_id,
            )
    elif vector_store is not None and state.had_existing_doc:
        logger.warning(
            "ingest_cleanup_preserved_existing_vectors doc_id=%s stage=%s",
            state.doc_id,
            state.stage,
        )

    if (
        state.route == "tree"
        and state.index_version
        and (not state.had_existing_doc or not state.target_tree_artifacts_existed)
    ):
        try:
            _delete_tree_artifacts(state.doc_id, state.index_version)
        except Exception:
            logger.exception(
                "ingest_cleanup_failed step=delete_tree_artifacts doc_id=%s index_version=%s",
                state.doc_id,
                state.index_version,
            )
    elif state.route == "tree" and state.index_version and state.target_tree_artifacts_existed:
        logger.warning(
            "ingest_cleanup_preserved_existing_tree_artifacts doc_id=%s index_version=%s",
            state.doc_id,
            state.index_version,
        )

    if state.storage_created and state.storage_path is not None:
        try:
            if state.storage_path.exists():
                state.storage_path.unlink()
        except Exception:
            logger.exception(
                "ingest_cleanup_failed step=delete_pdf doc_id=%s path=%s",
                state.doc_id,
                state.storage_path,
            )

    if state.registry_saved and not state.had_existing_doc:
        try:
            registry.delete(state.doc_id)
        except Exception:
            logger.exception(
                "ingest_cleanup_failed step=delete_registry doc_id=%s",
                state.doc_id,
            )
    elif state.registry_saved and state.had_existing_doc:
        logger.warning(
            "ingest_cleanup_preserved_existing_registry doc_id=%s",
            state.doc_id,
        )


def _delete_document_vectors(
    *,
    doc_id: str,
    vector_store: PineconeVectorStore,
) -> None:
    try:
        vector_store.clear_namespace(doc_id)
    except Exception as exc:
        logger.exception("delete_failed step=clear_namespace doc_id=%s", doc_id)
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Document delete aborted because vector cleanup failed.",
                "doc_id": doc_id,
                "failed_step": "clear_namespace",
                "cleanup_status": "local_state_preserved",
                "error": str(exc),
            },
        ) from exc

    try:
        vector_store.delete_ids([doc_id], namespace=settings.doc_summary_namespace)
    except Exception as exc:
        logger.exception("delete_failed step=delete_doc_summary doc_id=%s", doc_id)
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Document delete aborted because vector cleanup failed.",
                "doc_id": doc_id,
                "failed_step": "delete_doc_summary",
                "cleanup_status": "local_state_preserved",
                "error": str(exc),
            },
        ) from exc


def _delete_local_document_artifacts(meta: DocMeta) -> None:
    if meta.route == "tree":
        _delete_tree_artifacts(meta.doc_id, meta.index_version)

    storage_path = Path(meta.storage_path)
    if storage_path.exists():
        storage_path.unlink()

    registry.delete(meta.doc_id)


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

    state = IngestState(stage="save_upload")
    vector_store: PineconeVectorStore | None = None

    try:
        stored = save_upload(file, STORAGE_DIR)
    except Exception as exc:
        logger.exception("upload_failed stage=save_upload")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    state.doc_id = stored.doc_id
    state.storage_path = stored.path
    state.storage_created = bool(getattr(stored, "created", False))

    existing = registry.get(stored.doc_id)
    state.had_existing_doc = existing is not None

    try:
        state.stage = "ocr"
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
            "mistral_always",
            True,
            len(pages),
        )

        page_count = len(pages)
        token_count = sum(estimate_tokens(p.text, settings.openai_embedding_model) for p in pages)
        route = "tree" if page_count > settings.page_tree_threshold else "standard"
        index_version = compute_index_version(settings=settings, route=route)
        state.route = route
        state.index_version = index_version

        state.target_tree_artifacts_existed = bool(
            route == "tree"
            and _tree_path(stored.doc_id, index_version).exists()
            and _headings_path(stored.doc_id, index_version).exists()
        )

        if existing and existing.index_version == index_version and existing.route == route:
            if route != "tree" or state.target_tree_artifacts_existed:
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

        state.stage = "index"
        try:
            vector_store = get_vector_store()
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

        state.stage = "registry_save"
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

        state.registry_saved = True
        state.stage = "complete"
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
    except HTTPException:
        _cleanup_failed_ingest(state=state, vector_store=vector_store)
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception(
            "upload_failed stage=%s doc_id=%s unexpected_error",
            state.stage,
            state.doc_id,
        )
        _cleanup_failed_ingest(state=state, vector_store=vector_store)
        raise HTTPException(status_code=500, detail=f"Unexpected ingest failure: {exc}") from exc


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

    try:
        vector_store = get_vector_store()
    except HTTPException as exc:
        logger.exception("delete_failed step=get_vector_store doc_id=%s", doc_id)
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "message": "Document delete aborted because vector cleanup could not start.",
                "doc_id": doc_id,
                "failed_step": "get_vector_store",
                "cleanup_status": "local_state_preserved",
                "error": exc.detail,
            },
        ) from exc

    _delete_document_vectors(doc_id=doc_id, vector_store=vector_store)
    _delete_local_document_artifacts(meta)
    return {"ok": True, "doc_id": doc_id}
