from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
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
