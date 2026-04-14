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


def _normalize_gpt5_nano_chat_request(kwargs: dict) -> dict:
    """
    Normalize chat-completions kwargs for the OpenAI GPT-5 nano path we use in this repo.

    Current behavior:
    - prefer `max_completion_tokens` over legacy `max_tokens`
    - omit `temperature`; GPT-5 chat endpoints can reject non-default temperature values
    - keep `reasoning_effort` when provided, because tree indexing may use it
    """
    request_kwargs = dict(kwargs)

    if "max_tokens" in request_kwargs and "max_completion_tokens" not in request_kwargs:
        request_kwargs["max_completion_tokens"] = request_kwargs.pop("max_tokens")

    request_kwargs.pop("temperature", None)
    return request_kwargs


def chat_completions_create(settings: Settings, **kwargs):
    """
    OpenAI chat wrapper normalized for the GPT-5 nano setup used by this repo.

    We proactively shape requests for the GPT-5 chat API instead of carrying a
    broad compatibility layer for unrelated providers/models.
    """
    client = get_openai_client(settings)
    request_kwargs = _normalize_gpt5_nano_chat_request(kwargs)
    removed_reasoning_effort = False

    while True:
        try:
            return client.chat.completions.create(**request_kwargs)
        except TypeError as exc:
            msg = str(exc)
            msg_l = msg.lower()

            # Some client/API combinations reject this kwarg before making the request.
            if (
                not removed_reasoning_effort
                and "reasoning_effort" in request_kwargs
                and "reasoning_effort" in msg_l
                and "unexpected keyword argument" in msg_l
            ):
                request_kwargs = dict(request_kwargs)
                request_kwargs.pop("reasoning_effort", None)
                removed_reasoning_effort = True
                continue
            raise
        except BadRequestError as exc:
            msg = str(exc)
            msg_l = msg.lower()

            # Some client/API combinations still reject reasoning_effort outright.
            if not removed_reasoning_effort and "reasoning_effort" in request_kwargs and (
                "unsupported parameter: 'reasoning_effort'" in msg_l
                or "param': 'reasoning_effort'" in msg_l
            ):
                request_kwargs = dict(request_kwargs)
                request_kwargs.pop("reasoning_effort", None)
                removed_reasoning_effort = True
                continue

            raise
