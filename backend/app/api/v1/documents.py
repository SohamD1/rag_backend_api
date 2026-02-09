import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.pinecone_store import PineconeVectorStore
from app.api.schemas import DocumentCreateResponse, DocumentInfo, DocumentListResponse
from app.config import (
    resolve_docs_dir,
    resolve_ingest_dir,
    resolve_storage_dir,
    resolve_tree_dir,
    settings,
)
from app.core.index_version import compute_index_version
from app.core.indexing import build_standard_index, build_tree_index
from app.core.pdf_text import extract_pages
from app.core.tokens import estimate_tokens
from app.storage.local_store import save_local_file, save_upload
from app.storage.registry import DocMeta, DocRegistry, now_utc_iso


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


@router.post("/documents", response_model=DocumentCreateResponse)
def create_document(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    try:
        stored = save_upload(file, STORAGE_DIR)
    except Exception as exc:
        logger.exception("upload_failed stage=save_upload")
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    try:
        pages = extract_pages(str(stored.path))
    except Exception as exc:
        logger.exception("upload_failed stage=extract_pages doc_id=%s", stored.doc_id)
        raise HTTPException(status_code=400, detail=f"Failed to read PDF text: {exc}") from exc

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
        page_count=page_count,
        token_count=token_count,
        route=route,
        index_version=index_version,
        file_url=_file_url(stored.doc_id),
    )


@router.post("/documents/ingest-local")
def ingest_local_documents(limit: int | None = None, dry_run: bool = False):
    """
    Bulk ingest PDFs from a local directory on the server machine.

    Intended for local development: drop PDFs into LOCAL_INGEST_DIR (defaults to repo-root ./docs),
    then call this endpoint once.
    """
    ingest_dir = resolve_ingest_dir(settings)
    if not ingest_dir.exists() or not ingest_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Local ingest dir not found: {ingest_dir}. Create it or set LOCAL_INGEST_DIR.",
        )

    pdfs = sorted([p for p in ingest_dir.rglob("*.pdf") if p.is_file()])
    if limit is not None and limit > 0:
        pdfs = pdfs[: int(limit)]

    results: list[dict] = []
    errors: list[dict] = []

    for src_path in pdfs:
        try:
            if dry_run:
                results.append({"path": str(src_path), "status": "matched"})
                continue

            stored = save_local_file(src_path, STORAGE_DIR)

            pages = extract_pages(str(stored.path))
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
                    results.append(
                        {
                            "path": str(src_path),
                            "status": "skipped",
                            "doc_id": existing.doc_id,
                            "reason": "already_ingested",
                        }
                    )
                    continue

            if route == "tree":
                indexed_count = build_tree_index(
                    doc_id=stored.doc_id,
                    slug=stored.slug,
                    filename=stored.filename,
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
                    page_count=page_count,
                    token_count=token_count,
                    pages=pages,
                    settings=settings,
                    vector_store=vector_store,
                    index_version=index_version,
                )

            registry.save(
                DocMeta(
                    doc_id=stored.doc_id,
                    slug=stored.slug,
                    filename=stored.filename,
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

            logger.info(
                "local_ingest_complete path=%s doc_id=%s route=%s pages=%d est_tokens=%d indexed=%d",
                str(src_path),
                stored.doc_id,
                route,
                page_count,
                token_count,
                indexed_count,
            )

            results.append(
                {
                    "path": str(src_path),
                    "status": "ingested",
                    "doc": DocumentCreateResponse(
                        doc_id=stored.doc_id,
                        slug=stored.slug,
                        filename=stored.filename,
                        page_count=page_count,
                        token_count=token_count,
                        route=route,
                        index_version=index_version,
                        file_url=_file_url(stored.doc_id),
                    ).model_dump(),
                }
            )
        except Exception as exc:
            logger.exception("local_ingest_failed path=%s", str(src_path))
            errors.append({"path": str(src_path), "error": str(exc)})

    return {
        "ingest_dir": str(ingest_dir),
        "matched": len(pdfs),
        "results": results,
        "errors": errors,
        "dry_run": bool(dry_run),
    }


@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    metas = registry.list()
    docs = [
        DocumentInfo(
            doc_id=m.doc_id,
            slug=m.slug,
            filename=m.filename,
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


@router.get("/documents/{doc_id}/file", name="get_document_file")
def get_document_file(doc_id: str):
    meta = registry.get(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found.")
    path = Path(meta.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing from storage.")
    return FileResponse(path, media_type="application/pdf", filename=meta.filename)


@router.delete("/documents/{doc_id}")
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
