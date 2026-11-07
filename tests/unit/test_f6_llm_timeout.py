"""F-6: единый бюджет таймаута LLM (LLM_TIMEOUT, по умолчанию 90 c)."""

from __future__ import annotations

import httpx

from astro_app.grounded_qa import _default_llm_timeout, lmstudio_chat
from astro_app.query import _llm_timeout, reformulate_query


def test_default_timeout_is_ninety():
    assert _default_llm_timeout() == 90
    assert _llm_timeout() == 90


def test_env_overrides_timeout(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "60")
    assert _default_llm_timeout() == 60
    assert _llm_timeout() == 60


def test_invalid_env_falls_back_to_ninety(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "много")
    assert _default_llm_timeout() == 90
    assert _llm_timeout() == 90


def test_lmstudio_chat_uses_timeout_budget(monkeypatch):
    captured = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ОТВЕТ: тест [1]"}}]}

    def fake_post(url, json=None, timeout=None):
        captured["timeout"] = timeout
        return Resp()

    class Health:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Health())
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("LLM_TIMEOUT", "45")

    assert lmstudio_chat([{"role": "user", "content": "вопрос"}]) == "ОТВЕТ: тест [1]"
    assert captured["timeout"] == 45


def test_reformulate_post_uses_timeout_budget(monkeypatch):
    captured = {}

    class Resp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": "число гуа 1 направление"},
                        "finish_reason": "stop",
                    }
                ]
            }

    def fake_post(url, json=None, timeout=None):
        captured["timeout"] = timeout
        return Resp()

    class Health:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Health())
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("LLM_TIMEOUT", "75")

    result = reformulate_query(object(), "гуа 1")
    assert captured["timeout"] == 75
    assert result == "число гуа 1 направление"
