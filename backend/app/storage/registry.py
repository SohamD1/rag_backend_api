from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    slug: str
    filename: str
    source_url: str
    checksum: str
    size_bytes: int
    page_count: int
    token_count: int
    route: str
    storage_path: str | None
    index_version: str
    created_at: str


class DocRegistry:
    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, doc_id: str) -> Path:
        return self.registry_dir / f"{doc_id}.json"

    def save(self, meta: DocMeta) -> None:
        data = {
            "doc_id": meta.doc_id,
            "slug": meta.slug,
            "filename": meta.filename,
            "source_url": meta.source_url,
            "checksum": meta.checksum,
            "size_bytes": meta.size_bytes,
            "page_count": meta.page_count,
            "token_count": meta.token_count,
            "route": meta.route,
            "storage_path": meta.storage_path,
            "index_version": meta.index_version,
            "created_at": meta.created_at,
        }
        path = self._path(meta.doc_id)
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            tmp_path.write_text(json.dumps(data, indent=2))
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def get(self, doc_id: str) -> Optional[DocMeta]:
        path = self._path(doc_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        # Backward-compat if schema evolves.
        data.setdefault(
            "created_at",
            datetime.now(timezone.utc).isoformat(),
        )
        data.setdefault("size_bytes", 0)
        data.setdefault("checksum", "")
        data.setdefault("slug", data.get("doc_id", "document"))
        data.setdefault("source_url", "")
        data.setdefault("storage_path", None)
        return DocMeta(**data)

    def list(self) -> List[DocMeta]:
        metas: List[DocMeta] = []
        for path in sorted(self.registry_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            data.setdefault(
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            )
            data.setdefault("size_bytes", 0)
            data.setdefault("checksum", "")
            data.setdefault("slug", data.get("doc_id", "document"))
            data.setdefault("source_url", "")
            data.setdefault("storage_path", None)
            try:
                metas.append(DocMeta(**data))
            except Exception:
                continue
        return metas

    def find_by_checksum(self, checksum: str) -> Optional[DocMeta]:
        checksum = str(checksum or "").strip().lower()
        if not checksum:
            return None
        for meta in self.list():
            if str(meta.checksum or "").strip().lower() == checksum:
                return meta
        return None

    def delete(self, doc_id: str) -> bool:
        path = self._path(doc_id)
        if not path.exists():
            return False
        path.unlink()
        return True


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
