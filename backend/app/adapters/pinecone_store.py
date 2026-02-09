from __future__ import annotations

from typing import Dict, List, Optional

from pinecone import Pinecone

from app.config import Settings


class PineconeVectorStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        if settings.pinecone_host:
            self.index = self.client.Index(settings.pinecone_index, host=settings.pinecone_host)
        else:
            self.index = self.client.Index(settings.pinecone_index)

    def _request_timeout(self):
        return (5.0, float(self.settings.pinecone_timeout_seconds))

    def upsert(self, items: List[Dict], namespace: str) -> None:
        if not items:
            return
        batch_size = max(1, int(self.settings.pinecone_upsert_batch_size))
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            vectors = [(i["id"], i["values"], i.get("metadata", {})) for i in batch]
            self.index.upsert(
                vectors=vectors,
                namespace=namespace,
                _request_timeout=self._request_timeout(),
            )

    def clear_namespace(self, namespace: str) -> None:
        self.index.delete(
            delete_all=True,
            namespace=namespace,
            _request_timeout=self._request_timeout(),
        )

    def delete_ids(self, ids: List[str], namespace: str) -> None:
        ids = [i for i in (ids or []) if i]
        if not ids:
            return
        self.index.delete(
            ids=ids,
            namespace=namespace,
            _request_timeout=self._request_timeout(),
        )

    def query(
        self,
        *,
        vector: List[float],
        top_k: int,
        namespace: str,
        filter: Optional[Dict] = None,
        include_values: bool = False,
    ) -> List[Dict]:
        response = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=True,
            include_values=bool(include_values),
            _request_timeout=self._request_timeout(),
        )
        matches: List[Dict] = []
        for match in response.get("matches", []) or []:
            item = {
                "id": match.get("id"),
                "score": float(match.get("score", 0.0)),
                "metadata": match.get("metadata", {}) or {},
            }
            if include_values and match.get("values") is not None:
                item["values"] = match.get("values")
            matches.append(item)
        return matches

    def fetch(self, *, ids: List[str], namespace: str) -> Dict[str, List[float]]:
        ids = [i for i in (ids or []) if i]
        if not ids:
            return {}
        response = self.index.fetch(
            ids=ids,
            namespace=namespace,
            _request_timeout=self._request_timeout(),
        )
        vectors = response.get("vectors", {}) or {}
        out: Dict[str, List[float]] = {}
        for vec_id, payload in vectors.items():
            values = payload.get("values")
            if values:
                out[str(vec_id)] = values
        return out

