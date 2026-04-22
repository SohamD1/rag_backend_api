from __future__ import annotations

from types import SimpleNamespace

from app.core.generation import generate_answer


def _settings(**overrides):
    base = {
        "openai_generation_model": "gpt-test",
        "openai_generation_max_completion_tokens": 1400,
        "log_payloads": False,
        "log_payload_max_chars": 0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_generate_answer_uses_explanatory_prompt_and_extended_token_budget(monkeypatch):
    captured = {}

    def fake_chat_completions_create(settings, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"answer":"A direct explanation [1].\\n\\nKey points:\\n- Detail [1]"}'
                    )
                )
            ]
        )

    monkeypatch.setattr("app.core.generation.chat_completions_create", fake_chat_completions_create)

    answer = generate_answer(
        query="Why do digital assets matter in estate planning?",
        context_items=[
            {
                "header": "doc=guide pages=2-3",
                "text": "Digital assets can have monetary and sentimental value.",
                "doc_id": "guide",
            }
        ],
        settings=_settings(),
    )

    system_prompt = captured["kwargs"]["messages"][0]["content"]
    assert "Prioritize giving the most complete supported answer" in system_prompt
    assert "start with 1 short paragraph" in system_prompt
    assert "Only say you do not have enough information" in system_prompt
    assert captured["kwargs"]["max_completion_tokens"] == 1400
    assert answer.startswith("A direct explanation [1].")

