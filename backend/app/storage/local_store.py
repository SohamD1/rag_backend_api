from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import UploadFile

from app.core.slug import slugify


@dataclass(frozen=True)
class StoredFile:
    doc_id: str
    slug: str
    path: Path
    filename: str
    checksum: str
    size_bytes: int
    created: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_upload(upload: UploadFile, storage_dir: Path) -> StoredFile:
    """
    Save the upload to a temporary local path for OCR/indexing.
    The caller is responsible for deleting the file after ingest completes.
    """
    ensure_dir(storage_dir)

    suffix = Path(upload.filename or "document.pdf").suffix or ".pdf"
    stem = Path(upload.filename or "document").stem
    slug = slugify(stem)

    tmp_path = storage_dir / f".upload_{uuid4().hex}{suffix}"
    hasher = sha256()
    size_bytes = 0
    with tmp_path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            hasher.update(chunk)
            size_bytes += len(chunk)

    checksum = hasher.hexdigest()
    doc_id = f"{slug}__{checksum[:8]}"

    return StoredFile(
        doc_id=doc_id,
        slug=slug,
        path=tmp_path,
        filename=upload.filename or f"{doc_id}{suffix}",
        checksum=checksum,
        size_bytes=size_bytes,
        created=True,
    )


def save_local_file(source_path: Path, storage_dir: Path) -> StoredFile:
    """
    Copy a local file into the app storage dir and generate a stable doc_id.
    This supports bulk-ingesting PDFs that already exist on disk.
    """
    ensure_dir(storage_dir)

    suffix = source_path.suffix or ".pdf"
    stem = source_path.stem or "document"
    slug = slugify(stem)

    hasher = sha256()
    size_bytes = 0
    with source_path.open("rb") as src:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size_bytes += len(chunk)

    checksum = hasher.hexdigest()
    doc_id = f"{slug}__{checksum[:8]}"
    dest_path = storage_dir / f"{doc_id}{suffix}"
    created = not dest_path.exists()
    if not dest_path.exists():
        tmp_path = storage_dir / f".ingest_{uuid4().hex}{suffix}"
        with source_path.open("rb") as src, tmp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_path.replace(dest_path)

    return StoredFile(
        doc_id=doc_id,
        slug=slug,
        path=dest_path,
        filename=source_path.name,
        checksum=checksum,
        size_bytes=size_bytes,
        created=created,
    )
