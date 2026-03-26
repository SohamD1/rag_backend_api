from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class _DummyEncoder:
    def encode(self, text: str):
        return [tok for tok in (text or "").split() if tok]

    def decode(self, toks):
        return " ".join(toks)


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    from app.api.v1 import documents
    from app.core import chunking, tokens
    from app.services import tree_index

    encoder = _DummyEncoder()

    def estimate(text: str, model: str) -> int:
        return len(encoder.encode(text))

    monkeypatch.setattr(tokens, "get_token_encoder", lambda model: encoder)
    monkeypatch.setattr(tokens, "estimate_tokens", estimate)
    monkeypatch.setattr(chunking, "get_token_encoder", lambda model: encoder)
    monkeypatch.setattr(chunking, "estimate_tokens", estimate)
    monkeypatch.setattr(tree_index, "get_token_encoder", lambda model: encoder)
    monkeypatch.setattr(tree_index, "estimate_tokens", estimate)
    monkeypatch.setattr(documents, "estimate_tokens", estimate)


@pytest.fixture
def work_tmp():
    base = REPO_ROOT / ".tmp" / "pytest_work"
    base.mkdir(exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=base))
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def client():
    from app.api import rate_limit
    from app.api import deps
    from app.config import settings
    from app.main import app

    old_require_auth = settings.kb_require_auth
    rate_limit._admin_limiter = None
    rate_limit._chat_limiter = None
    deps._vector_store = None
    object.__setattr__(settings, "kb_require_auth", False)

    with TestClient(app) as test_client:
        yield test_client

    object.__setattr__(settings, "kb_require_auth", old_require_auth)
    rate_limit._admin_limiter = None
    rate_limit._chat_limiter = None
    deps._vector_store = None
