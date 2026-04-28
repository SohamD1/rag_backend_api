from __future__ import annotations

from types import SimpleNamespace

from app.core.generation import _limit_bullets, generate_answer, review_answer


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


def test_review_answer_fixes_exclusion_and_checklist_misses(monkeypatch):
    captured = {}

    def fake_chat_completions_create(settings, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"answer":"Besides a will, the plan should include powers of attorney and incapacity planning [1]."}'
                    )
                )
            ]
        )

    monkeypatch.setattr("app.core.generation.chat_completions_create", fake_chat_completions_create)

    answer = review_answer(
        query="What are the common elements of an estate plan besides a will?",
        draft_answer="A will and powers of attorney are common estate planning tools [1].",
        context_items=[
            {
                "header": "doc=guide pages=5-6",
                "text": "Common elements include powers of attorney and planning for incapacity.",
                "doc_id": "guide",
            }
        ],
        settings=_settings(),
    )

    system_prompt = captured["kwargs"]["messages"][0]["content"]
    user_prompt = captured["kwargs"]["messages"][1]["content"]
    assert "improve answer completeness and instruction-following" in system_prompt
    assert "Do not include wills" in user_prompt
    assert answer.startswith("Besides a will")


def test_limit_bullets_handles_common_bullet_markers():
    answer = "\n".join(
        [
            "Intro.",
            "- one",
            "* two",
            "\u2022 three",
            "1. four",
            "2. five",
        ]
    )

    assert _limit_bullets(answer, max_bullets=3) == "Intro.\n- one\n* two\n\u2022 three"
