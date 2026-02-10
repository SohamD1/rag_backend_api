from __future__ import annotations

from typing import Optional

from openai import OpenAI
from openai import BadRequestError

from app.config import Settings

_openai_client: Optional[OpenAI] = None


def get_openai_client(settings: Settings) -> OpenAI:
    global _openai_client
    if _openai_client is None:
        kwargs = {}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _openai_client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
            **kwargs,
        )
    return _openai_client


def chat_completions_create(settings: Settings, **kwargs):
    """
    Compatibility wrapper for model-specific parameter support.

    Some models (e.g. certain GPT-5 variants) reject non-default `temperature`.
    We prefer deterministic settings where possible, but fall back to the model
    default when the API rejects a provided temperature value.
    """
    client = get_openai_client(settings)
    try:
        return client.chat.completions.create(**kwargs)
    except BadRequestError as exc:
        # Narrow retry: only strip temperature when the server explicitly rejects it.
        msg = str(exc)
        if "temperature" in kwargs and (
            "temperature' does not support" in msg
            or "param': 'temperature'" in msg
            or "Only the default (1) value is supported" in msg
        ):
            kwargs2 = dict(kwargs)
            kwargs2.pop("temperature", None)
            return client.chat.completions.create(**kwargs2)
        raise
