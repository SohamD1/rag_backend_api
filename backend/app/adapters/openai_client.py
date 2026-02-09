from __future__ import annotations

from typing import Optional

from openai import OpenAI

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

