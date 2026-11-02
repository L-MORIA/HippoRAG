"""Регрессия: режим 19 (GraphRAG Agent) обязан возвращать ответ.

Найденный на полном прогоне дефект: функция `search_graphrag_agent` выполняла
маршрутизацию, каскадный поиск и вызов LLM, но заканчивалась без `return` —
всегда отдавала None, и меню печатало «Ничего не найдено». Дефект был ещё до
правок P1-6…P1-9 (проверено `git show HEAD:main.py`).
"""

import pytest


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _row(text):
    return (text, 1.0, 0.9, 0.5, 0.0)


@pytest.fixture
def fake_pipeline(monkeypatch):
    import main

    # Router для «какие книги…» уходит в ветку is_global → rrf + small-to-big.
    monkeypatch.setattr(
        "astro_app.modes.graph.search_rrf_context",
        lambda hipporag, q, **kw: [
            _row("Максимова Ирина Валериевна Психологический фэншуй"),
            _row("Астрофэншуй квартиры издание второе"),
        ],
    )
    monkeypatch.setattr(
        "astro_app.modes.graph.search_small_to_big_ru",
        lambda hipporag, q, **kw: [_row("Эзотерический фэншуй переиздание")],
    )

    def fake_get(url, timeout=3):
        return FakeResponse({"data": []})

    def fake_post(url, json=None, timeout=300):
        assert json and json["messages"], "промпт должен содержать источники"
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Максимова написала три книги: Психологический, "
                            "Астрофэншуй квартиры, Эзотерический фэншуй."
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr("httpx.post", fake_post)
    return main


def test_graphrag_agent_returns_answer(fake_pipeline):
    results = fake_pipeline.search_graphrag_agent(object(), "какие книги написала Максимова?")

    assert results, "режим 19 вернул пустоту вместо ответа LLM"
    text = results[0][0]
    assert "Максимова" in text
    assert text.strip()


def test_graphrag_agent_falls_back_to_search_results(monkeypatch):
    """Если LLM недоступен — отдаём найденные результаты, а не None."""
    import main

    monkeypatch.setattr(
        "astro_app.modes.graph.search_rrf_context",
        lambda hipporag, q, **kw: [_row("фрагмент корпуса")],
    )
    monkeypatch.setattr(
        "astro_app.modes.graph.search_small_to_big_ru",
        lambda hipporag, q, **kw: [_row("другой фрагмент")],
    )
    monkeypatch.setattr("httpx.get", lambda url, timeout=3: FakeResponse({"data": []}))
    monkeypatch.setattr(
        "httpx.post", lambda url, json=None, timeout=300: FakeResponse({"choices": []})
    )

    results = main.search_graphrag_agent(object(), "какие книги написала Максимова?")

    assert results, "без ответа LLM режим не должен возвращать пустоту"
    assert results[0][0]
