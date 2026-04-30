import logging
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from fastapi import File, Form, HTTPException, UploadFile

from app.adapters.pinecone_store import PineconeVectorStore
from app.api.deps import get_vector_store
from app.api.schemas import DocumentCreateResponse, DocumentInfo, DocumentListResponse
from app.config import (
    resolve_docs_dir,
    resolve_storage_dir,
    resolve_tree_dir,
    settings,
)
from app.core.doc_summaries import (
    delete_doc_summary_records,
    doc_summary_record_ids,
    upsert_doc_summaries,
    upsert_doc_summary_records,
)
from app.core.index_version import compute_index_version
from app.core.indexing import IndexBuildResult, build_standard_index, build_tree_index
from app.core.tokens import estimate_tokens
from app.core.vector_namespace import vector_namespace
from app.storage.local_store import UploadRejectedError, save_upload
from app.storage.registry import DocMeta, DocRegistry, now_utc_iso
from app.adapters.mistral_ocr import MistralOcrError, ocr_pdf_to_pages


logger = logging.getLogger(__name__)

DOCS_DIR = resolve_docs_dir(settings)
TREE_DIR = resolve_tree_dir(settings)
STORAGE_DIR = resolve_storage_dir(settings)

registry = DocRegistry(DOCS_DIR)


@dataclass
class IngestState:
    doc_id: str | None = None
    storage_path: Path | None = None
    temp_upload_created: bool = False
    had_existing_doc: bool = False
    previous_meta: DocMeta | None = None
    previous_doc_summary_records: Dict[str, Dict] | None = None
    route: str | None = None
    index_version: str | None = None
    target_namespace: str | None = None
    target_tree_artifacts_existed: bool = False
    registry_saved: bool = False
    doc_summary_saved: bool = False
    stage: str = "init"


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

    previous_namespace = (
        vector_namespace(state.previous_meta.doc_id, state.previous_meta.index_version)
        if state.previous_meta is not None
        else None
    )

    if vector_store is not None and state.target_namespace:
        should_clear_target_namespace = (
            not state.had_existing_doc or state.target_namespace != previous_namespace
        )
        if should_clear_target_namespace:
            try:
                vector_store.clear_namespace(state.target_namespace)
            except Exception:
                logger.exception(
                    "ingest_cleanup_failed step=clear_namespace namespace=%s doc_id=%s",
                    state.target_namespace,
                    state.doc_id,
                )
        else:
            logger.warning(
                "ingest_cleanup_preserved_existing_vectors doc_id=%s stage=%s namespace=%s",
                state.doc_id,
                state.stage,
                state.target_namespace,
            )

        if not state.had_existing_doc:
            try:
                delete_doc_summary_records(
                    doc_id=state.doc_id,
                    settings=settings,
                    vector_store=vector_store,
                )
            except Exception:
                logger.exception(
                    "ingest_cleanup_failed step=delete_doc_summary doc_id=%s",
                    state.doc_id,
                )
        elif (
            state.doc_summary_saved
            and state.previous_meta
            and state.previous_doc_summary_records
        ):
            try:
                upsert_doc_summary_records(
                    records=list(state.previous_doc_summary_records.values()),
                    settings=settings,
                    vector_store=vector_store,
                )
            except Exception:
                logger.exception(
                    "ingest_cleanup_failed step=restore_doc_summary doc_id=%s",
                    state.doc_id,
                )
        elif state.doc_summary_saved and state.had_existing_doc:
            logger.warning(
                "ingest_cleanup_doc_summary_not_restored doc_id=%s reason=%s",
                state.doc_id,
                "missing_previous_summary",
            )

    if (
        state.route == "tree"
        and state.index_version
        and (
            not state.had_existing_doc
            or not state.previous_meta
            or state.previous_meta.index_version != state.index_version
            or not state.target_tree_artifacts_existed
        )
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

    if state.temp_upload_created and state.storage_path is not None:
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
    elif state.registry_saved and state.had_existing_doc and state.previous_meta is not None:
        try:
            registry.save(state.previous_meta)
        except Exception:
            logger.exception(
                "ingest_cleanup_failed step=restore_registry doc_id=%s",
                state.doc_id,
            )
    elif state.registry_saved and state.had_existing_doc:
        logger.warning("ingest_cleanup_preserved_existing_registry doc_id=%s", state.doc_id)


def _finalize_superseded_index(
    *,
    state: IngestState,
    vector_store: PineconeVectorStore | None,
) -> None:
    previous_meta = state.previous_meta
    if previous_meta is None or state.index_version is None:
        return
    if previous_meta.index_version == state.index_version:
        return

    old_namespace = vector_namespace(previous_meta.doc_id, previous_meta.index_version)
    if vector_store is not None:
        try:
            vector_store.clear_namespace(old_namespace)
        except Exception:
            logger.exception(
                "ingest_finalize_failed step=clear_old_namespace namespace=%s doc_id=%s",
                old_namespace,
                previous_meta.doc_id,
            )

    if previous_meta.route == "tree":
        try:
            _delete_tree_artifacts(previous_meta.doc_id, previous_meta.index_version)
        except Exception:
            logger.exception(
                "ingest_finalize_failed step=delete_old_tree_artifacts doc_id=%s index_version=%s",
                previous_meta.doc_id,
                previous_meta.index_version,
            )


def _delete_document_vectors(
    *,
    doc_id: str,
    index_version: str,
    vector_store: PineconeVectorStore,
) -> None:
    namespace = vector_namespace(doc_id, index_version)
    try:
        vector_store.clear_namespace(namespace)
    except Exception as exc:
        logger.exception(
            "delete_failed step=clear_namespace doc_id=%s namespace=%s",
            doc_id,
            namespace,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Document delete aborted because vector cleanup failed.",
                "doc_id": doc_id,
                "failed_step": "clear_namespace",
                "cleanup_status": "local_state_preserved",
                "namespace": namespace,
                "error": str(exc),
            },
        ) from exc

    try:
        delete_doc_summary_records(
            doc_id=doc_id,
            settings=settings,
            vector_store=vector_store,
        )
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
    doc_tree_dir = TREE_DIR / meta.doc_id
    if doc_tree_dir.exists():
        for version_dir in sorted(doc_tree_dir.glob("*"), reverse=True):
            if version_dir.is_dir():
                _delete_tree_artifacts(meta.doc_id, version_dir.name)

    registry.delete(meta.doc_id)


def create_document(file: UploadFile = File(...), source_url: str = Form(...)):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    source_url = (source_url or "").strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="source_url is required.")

    state = IngestState(stage="save_upload")
    vector_store: PineconeVectorStore | None = None

    try:
        stored = save_upload(
            file,
            STORAGE_DIR,
            max_size_bytes=max(1, int(settings.max_upload_bytes)),
        )
    except UploadRejectedError as exc:
        logger.info("upload_rejected stage=save_upload reason=%s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("upload_failed stage=save_upload")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    duplicate = getattr(registry, "find_by_checksum", lambda checksum: None)(stored.checksum)
    if duplicate is not None and duplicate.doc_id != stored.doc_id:
        stored = replace(stored, doc_id=duplicate.doc_id, slug=duplicate.slug)

    state.doc_id = stored.doc_id
    state.storage_path = stored.path
    state.temp_upload_created = bool(getattr(stored, "created", False))

    existing = registry.get(stored.doc_id)
    state.had_existing_doc = existing is not None
    state.previous_meta = existing

    try:
        if existing is not None:
            current_existing_route = (
                "tree" if existing.page_count > settings.page_tree_threshold else "standard"
            )
            current_existing_index_version = compute_index_version(
                settings=settings, route=current_existing_route
            )
            existing_tree_ready = bool(
                current_existing_route != "tree"
                or (
                    _tree_path(existing.doc_id, current_existing_index_version).exists()
                    and _headings_path(existing.doc_id, current_existing_index_version).exists()
                )
            )
            if (
                source_url == existing.source_url
                and existing.route == current_existing_route
                and existing.index_version == current_existing_index_version
                and existing_tree_ready
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
                    file_url=None,
                )

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
        state.target_namespace = vector_namespace(stored.doc_id, index_version)

        state.target_tree_artifacts_existed = bool(
            route == "tree"
            and _tree_path(stored.doc_id, index_version).exists()
            and _headings_path(stored.doc_id, index_version).exists()
        )

        if (
            existing
            and source_url == existing.source_url
            and existing.index_version == index_version
            and existing.route == route
        ):
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
                    file_url=None,
                )

        state.stage = "index"
        try:
            vector_store = get_vector_store()
            if existing is not None:
                try:
                    state.previous_doc_summary_records = vector_store.fetch_records(
                        ids=doc_summary_record_ids(stored.doc_id),
                        namespace=settings.doc_summary_namespace,
                    )
                except Exception:
                    logger.exception(
                        "upload_failed stage=fetch_previous_doc_summary doc_id=%s",
                        stored.doc_id,
                    )
            if route == "tree":
                build_result = build_tree_index(
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
                build_result = build_standard_index(
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
                    storage_path=None,
                    index_version=index_version,
                    created_at=now_utc_iso(),
                )
            )
        except Exception as exc:
            logger.exception("upload_failed stage=registry_save doc_id=%s", stored.doc_id)
            raise HTTPException(status_code=500, detail=f"Failed to save document metadata: {exc}") from exc

        state.registry_saved = True
        state.stage = "doc_summary"
        try:
            upsert_doc_summaries(
                doc_id=stored.doc_id,
                slug=stored.slug,
                filename=stored.filename,
                source_url=source_url,
                route=route,
                page_count=page_count,
                token_count=token_count,
                index_version=index_version,
                summary_texts=build_result.doc_summary_texts,
                settings=settings,
                vector_store=vector_store,
            )
        except Exception as exc:
            logger.exception("upload_failed stage=doc_summary doc_id=%s", stored.doc_id)
            raise HTTPException(
                status_code=502,
                detail=f"Failed to update document summary index: {exc}",
            ) from exc
        state.doc_summary_saved = True

        state.stage = "finalize"
        _finalize_superseded_index(state=state, vector_store=vector_store)

        state.stage = "complete"
        logger.info(
            "upload_complete doc_id=%s route=%s pages=%d est_tokens=%d indexed=%d",
            stored.doc_id,
            route,
            page_count,
            token_count,
            build_result.indexed_count,
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
            file_url=None,
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
    finally:
        if state.temp_upload_created and state.storage_path is not None:
            try:
                if state.storage_path.exists():
                    state.storage_path.unlink()
            except Exception:
                logger.exception(
                    "upload_failed stage=temp_cleanup doc_id=%s path=%s",
                    state.doc_id,
                    state.storage_path,
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
            file_url=None,
        )
        for m in metas
    ]
    docs = sorted(docs, key=lambda d: d.filename.lower())
    return DocumentListResponse(docs=docs)


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
    _delete_document_vectors(
        doc_id=doc_id,
        index_version=meta.index_version,
        vector_store=vector_store,
    )
    _delete_local_document_artifacts(meta)
    return {"ok": True, "doc_id": doc_id}
