from __future__ import annotations

from typing import Optional

from openai import OpenAI
from openai import BadRequestError

from app.config import Settings

_openai_client: Optional[OpenAI] = None


def get_openai_client(settings: Settings) -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not getattr(settings, "openai_api_key", None):
            raise RuntimeError("OPENAI_API_KEY is not configured")
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
    request_kwargs = dict(kwargs)
    removed_temperature = False
    swapped_to_max_completion_tokens = False
    swapped_to_max_tokens = False

    while True:
        try:
            return client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            msg = str(exc)
            msg_l = msg.lower()

            # Retry 1: strip temperature when model only allows default temperature.
            if not removed_temperature and "temperature" in request_kwargs and (
                "temperature' does not support" in msg
                or "param': 'temperature'" in msg
                or "only the default (1) value is supported" in msg_l
            ):
                request_kwargs = dict(request_kwargs)
                request_kwargs.pop("temperature", None)
                removed_temperature = True
                continue

            # Retry 2: some models require max_completion_tokens instead of max_tokens.
            if (
                not swapped_to_max_completion_tokens
                and "max_tokens" in request_kwargs
                and "max_completion_tokens" not in request_kwargs
                and (
                    "unsupported parameter: 'max_tokens'" in msg_l
                    or "param': 'max_tokens'" in msg_l
                )
            ):
                request_kwargs = dict(request_kwargs)
                request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")
                swapped_to_max_completion_tokens = True
                continue

            # Retry 3: older/alternate providers may reject max_completion_tokens.
            if (
                not swapped_to_max_tokens
                and "max_completion_tokens" in request_kwargs
                and "max_tokens" not in request_kwargs
                and (
                    "unsupported parameter: 'max_completion_tokens'" in msg_l
                    or "param': 'max_completion_tokens'" in msg_l
                )
            ):
                request_kwargs = dict(request_kwargs)
                request_kwargs["max_tokens"] = request_kwargs.pop("max_completion_tokens")
                swapped_to_max_tokens = True
                continue

            raise
