from __future__ import annotations

import json
import re
from typing import Any, Dict


def strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if "```" not in cleaned:
        return cleaned
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def clean_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("None", "null")
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.replace(",]", "]").replace(",}", "}")
    return cleaned


def extract_json_payload(text: str) -> Any:
    cleaned = clean_json_text(strip_code_fences(text))
    if not cleaned:
        raise ValueError("Empty JSON payload")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_candidates = [cleaned.find("["), cleaned.find("{")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        raise
    start = min(start_candidates)

    end_candidates = [cleaned.rfind("]"), cleaned.rfind("}")]
    end_candidates = [i for i in end_candidates if i != -1]
    if not end_candidates:
        raise
    end = max(end_candidates) + 1

    fragment = clean_json_text(cleaned[start:end])
    return json.loads(fragment)


def extract_json_object(text: str) -> Dict[str, Any]:
    payload = extract_json_payload(text)
    if isinstance(payload, dict):
        return payload
    raise ValueError("Expected JSON object")

