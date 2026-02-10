from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from app.core.pdf_text import PageText


class MistralOcrError(RuntimeError):
    pass


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _get_str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip()


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes"}


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _upload_pdf(*, client: httpx.Client, base_url: str, api_key: str, pdf_bytes: bytes, filename: str) -> str:
    url = f"{base_url}/v1/files"
    try:
        resp = client.post(
            url,
            headers=_headers(api_key),
            data={"purpose": "ocr"},
            files={"file": (filename or "document.pdf", pdf_bytes, "application/pdf")},
        )
    except Exception as exc:  # pragma: no cover
        raise MistralOcrError(f"Failed to upload PDF to Mistral files API: {exc}") from exc

    if resp.status_code >= 400:
        raise MistralOcrError(f"Mistral files API error {resp.status_code}: {resp.text}")
    payload = resp.json() if resp.content else {}
    file_id = str(payload.get("id") or "").strip()
    if not file_id:
        raise MistralOcrError("Mistral files API did not return a file id")
    return file_id


def _delete_file_best_effort(*, client: httpx.Client, base_url: str, api_key: str, file_id: str) -> None:
    if not file_id:
        return
    url = f"{base_url}/v1/files/{file_id}"
    try:
        client.delete(url, headers=_headers(api_key))
    except Exception:
        return


def _ocr_file_id(
    *, client: httpx.Client, base_url: str, api_key: str, model: str, file_id: str
) -> Dict[str, Any]:
    url = f"{base_url}/v1/ocr"
    body = {
        "model": model,
        "document": {"type": "file", "file_id": file_id},
        "include_image_base64": False,
    }
    try:
        resp = client.post(url, headers=_headers(api_key), json=body)
    except Exception as exc:  # pragma: no cover
        raise MistralOcrError(f"Failed to call Mistral OCR API: {exc}") from exc

    if resp.status_code >= 400:
        raise MistralOcrError(f"Mistral OCR API error {resp.status_code}: {resp.text}")
    return resp.json() if resp.content else {}


def ocr_pdf_to_pages(*, pdf_path: Path, filename: Optional[str] = None) -> List[PageText]:
    """
    OCR a PDF with Mistral OCR and return PageText list (1-indexed pages).

    Env vars:
    - MISTRAL_API_KEY (required)
    - MISTRAL_BASE_URL (default https://api.mistral.ai)
    - MISTRAL_OCR_MODEL (default mistral-ocr-latest)
    - MISTRAL_OCR_TIMEOUT_SECONDS (default 120)
    - MISTRAL_OCR_DELETE_UPLOADED_FILE (default true)
    """
    api_key = _get_str_env("MISTRAL_API_KEY", "")
    if not api_key:
        raise MistralOcrError("MISTRAL_API_KEY is not set")

    base_url = _get_str_env("MISTRAL_BASE_URL", "https://api.mistral.ai").rstrip("/")
    model = _get_str_env("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    timeout_s = _get_float_env("MISTRAL_OCR_TIMEOUT_SECONDS", 120.0)
    delete_uploaded = _get_bool_env("MISTRAL_OCR_DELETE_UPLOADED_FILE", True)

    pdf_bytes = Path(pdf_path).read_bytes()
    if not pdf_bytes:
        raise MistralOcrError("PDF is empty")

    timeout = httpx.Timeout(timeout_s, connect=10.0)
    with httpx.Client(timeout=timeout) as client:
        file_id = _upload_pdf(
            client=client,
            base_url=base_url,
            api_key=api_key,
            pdf_bytes=pdf_bytes,
            filename=filename or Path(pdf_path).name,
        )
        try:
            payload = _ocr_file_id(
                client=client, base_url=base_url, api_key=api_key, model=model, file_id=file_id
            )
        finally:
            if delete_uploaded:
                _delete_file_best_effort(
                    client=client, base_url=base_url, api_key=api_key, file_id=file_id
                )

    pages = payload.get("pages") or []
    if not isinstance(pages, list) or not pages:
        raise MistralOcrError("Mistral OCR returned no pages")

    out: List[PageText] = []
    for i, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        md = page.get("markdown")
        if md is None:
            # Some responses may use "text" or similar; keep a conservative fallback.
            md = page.get("text")
        text = str(md or "")
        out.append(PageText(page_num=i + 1, text=text))

    if not out:
        raise MistralOcrError("Mistral OCR returned pages but none were usable")
    return out

