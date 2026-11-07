"""Тесты веб-UI H2-1: режимы, заземлённый ответ, обратная связь и статика."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import HTTPException

from astro_app.modes import ModeSpec
from astro_app.web import server as web_server
from astro_app.web import service
from astro_app.web.feedback import FeedbackStore
from astro_app.web.models import AskRequest, FeedbackRequest


class FakeStore:
    """Минимальное хранилище чанков с метаданными."""

    def __init__(self, texts, pages=None, sources=None):
        self.texts = list(texts)
        self.hash_ids = [f"h{index}" for index in range(len(self.texts))]
        self.embeddings = []
        pages = pages or [index + 1 for index in range(len(self.texts))]
        sources = sources or ["book.pdf" for _ in self.texts]
        self._rows = {
            f"h{index}": {
                "metadata": [
                    {
                        "source_file": sources[index],
                        "doc_title": f"Книга {index + 1}",
                        "page_num": pages[index],
                        "section": "Раздел",
                    }
                ]
            }
            for index in range(len(self.texts))
        }

    def get_row(self, hash_id):
        return self._rows[hash_id]


class FakeHippoRAG:
    def __init__(self, store):
        self.chunk_embedding_store = store


def _spec(fn, extra=False):
    return ModeSpec(label="test", menu="test", fn=fn, extra=extra)


def _row(text, score=1.0):
    return (text, score, 0.0, 0.0, 0.0)


@pytest.fixture
def fake_hipporag():
    return FakeHippoRAG(
        FakeStore(
            [
                "Число гуа 8 связано с землёй.",
                "Зона земли находится в центре.",
            ],
            pages=[12, 13],
        )
    )


@pytest.fixture
def grounded_chat():
    def chat(_messages):
        return "ОТВЕТ: Число гуа 8 связано с землёй [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет"

    return chat


def test_mode_payload_is_default_plus_two_experts():
    modes = service.mode_payload()

    assert [mode.id for mode in modes] == ["5", "2", "1"]
    assert modes[0].expert is False
    assert sum(1 for mode in modes if not mode.expert) == 1
    assert modes[0].id == service.DEFAULT_UI_MODE


def test_answer_question_returns_used_citations_and_fragments(
    monkeypatch, fake_hipporag, grounded_chat
):
    monkeypatch.setitem(
        service.MODES,
        "5",
        _spec(
            lambda _hipporag, _query, top_k=5: [
                _row("Число гуа 8 связано с землёй."),
                _row("Зона земли находится в центре."),
            ]
        ),
    )

    response = service.answer_question(
        fake_hipporag,
        "С чем связано число гуа 8?",
        mode_id="5",
        chat=grounded_chat,
    )

    assert response.answer.startswith("Число гуа 8")
    assert response.mode == "5"
    assert response.mode_label == "Быстрый"
    assert response.expert is False
    assert response.abstained is False
    assert response.confidence == 1.0
    assert [citation.id for citation in response.citations] == [1]
    assert response.citations[0].doc_title == "Книга 1"
    assert response.citations[0].page_num == 12
    assert len(response.fragments) == 2
    assert response.retrieved_docs[0] == "Число гуа 8 связано с землёй."


def test_answer_question_only_returns_cited_fragment(monkeypatch, fake_hipporag):
    monkeypatch.setitem(
        service.MODES,
        "5",
        _spec(
            lambda _hipporag, _query, top_k=5: [
                _row("Число гуа 8 связано с землёй."),
                _row("Зона земли находится в центре."),
            ]
        ),
    )

    def chat(_messages):
        return "ОТВЕТ: Второй фрагмент подтверждает ответ [2].\nИСТОЧНИКИ: 2\nОТКАЗ: нет"

    response = service.answer_question(fake_hipporag, "Вопрос", chat=chat)

    assert [citation.id for citation in response.citations] == [2]
    assert response.citations[0].page_num == 13
    assert len(response.fragments) == 2


def test_answer_question_without_fragments_is_honest_abstention(monkeypatch, fake_hipporag):
    monkeypatch.setitem(service.MODES, "5", _spec(lambda _hipporag, _query, top_k=5: []))

    response = service.answer_question(
        fake_hipporag, "Вопрос", chat=lambda _m: "не должен вызваться"
    )

    assert response.abstained is True
    assert response.confidence == 0.0
    assert response.citations == []
    assert "no_fragments" in response.grounding_errors


def test_answer_question_marks_empty_llm_response(monkeypatch, fake_hipporag):
    monkeypatch.setitem(
        service.MODES,
        "5",
        _spec(lambda _hipporag, _query, top_k=5: [_row("Фрагмент.")]),
    )

    response = service.answer_question(fake_hipporag, "Вопрос", chat=lambda _m: "")

    assert response.abstained is True
    assert "llm_unavailable_or_empty" in response.grounding_errors


def test_extra_mode_exposes_reformulated_query(monkeypatch, fake_hipporag):
    monkeypatch.setitem(
        service.MODES,
        "1",
        _spec(
            lambda _hipporag, _query, top_k=5: (
                [_row("Фрагмент про число гуа.")],
                "гуа 8 земля",
            ),
            extra=True,
        ),
    )

    response = service.answer_question(
        fake_hipporag,
        "С чем связано 8?",
        mode_id="1",
        chat=lambda _m: "ОТВЕТ: Фрагмент [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет",
    )

    assert response.reformulated_query == "гуа 8 земля"
    assert response.mode_label == "Максимальный охват"
    assert response.expert is True


def test_six_tuple_mode_uses_source_and_page_range(monkeypatch, fake_hipporag):
    monkeypatch.setitem(
        service.MODES,
        "2",
        _spec(
            lambda _hipporag, _query, top_k=5: [
                ("Полный текст страницы", 1.0, "book.pdf", 3, 4, 0.2)
            ]
        ),
    )

    response = service.answer_question(
        fake_hipporag,
        "Вопрос",
        mode_id="2",
        chat=lambda _m: "ОТВЕТ: Ответ [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет",
    )

    assert response.fragments[0].page_num == 3
    assert response.fragments[0].section == "стр. 3-4"
    assert response.fragments[0].doc_title == "book"


def test_duplicate_results_are_deduplicated(monkeypatch, fake_hipporag):
    row = _row("Один и тот же фрагмент.")
    monkeypatch.setitem(service.MODES, "5", _spec(lambda _hipporag, _query, top_k=5: [row, row]))

    response = service.answer_question(
        fake_hipporag,
        "Вопрос",
        chat=lambda _m: "ОТВЕТ: Ответ [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет",
    )

    assert len(response.fragments) == 1


def test_unknown_mode_is_rejected(fake_hipporag):
    with pytest.raises(ValueError, match="Неизвестный режим"):
        service.answer_question(fake_hipporag, "Вопрос", mode_id="99")


def test_feedback_store_appends_utf8_jsonl(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.jsonl")

    record = store.append(
        query="Что такое гуа 8?",
        answer="Число гуа 8 связано с землёй [1].",
        mode="5",
        rating="up",
        comment="Понятно",
    )

    assert record["schema_version"] == 1
    assert record["id"]
    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["comment"] == "Понятно"
    assert saved["rating"] == "up"
    assert store.read_all() == [saved]


def test_ui_config_exposes_three_modes():
    config = asyncio.run(web_server.ui_config())

    assert config.auth_required in (True, False)
    assert config.default_mode == "5"
    assert [mode.id for mode in config.modes] == ["5", "2", "1"]
    assert config.feedback_enabled is True


def test_ask_route_maps_unknown_mode_to_400(monkeypatch):
    monkeypatch.setattr(web_server.api_server, "hipporag", object())

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("Неизвестный режим: 99")

    monkeypatch.setattr(web_server.service, "answer_question", raise_value_error)

    with pytest.raises(HTTPException) as error:
        asyncio.run(web_server.ask(AskRequest(question="Вопрос", mode="99")))

    assert error.value.status_code == 400


class _FakeAnswer:
    """Минимальный ответ сервиса для проверки маршрута без пайплайна."""

    retrieval_time = 0.0
    answer_time = 0.0
    abstained = False
    citations = [1]
    grounding_errors: list[str] = []
    mode = "5"
    fragments = [object()]


def test_concurrent_ask_requests_are_serialized(monkeypatch):
    """Пайплайн общий: два одновременных /api/ask не должны идти параллельно."""
    monkeypatch.setattr(web_server.api_server, "hipporag", object())
    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    def fake_answer_question(*_args, **_kwargs):
        with guard:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.1)
        finally:
            with guard:
                state["active"] -= 1
        return _FakeAnswer()

    monkeypatch.setattr(web_server.service, "answer_question", fake_answer_question)

    async def run_both():
        return await asyncio.gather(
            web_server.ask(AskRequest(question="Вопрос 1", mode="5")),
            web_server.ask(AskRequest(question="Вопрос 2", mode="1")),
        )

    answers = asyncio.run(run_both())

    assert state["max_active"] == 1
    assert len(answers) == 2


def test_feedback_route_writes_record(tmp_path, monkeypatch):
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    monkeypatch.setattr(web_server, "feedback_store", store)

    response = asyncio.run(
        web_server.feedback(
            FeedbackRequest(
                rating="down",
                query="Вопрос",
                answer="Ответ [1].",
                mode="5",
                comment="Мало источников",
            )
        )
    )

    assert response.status == "ok"
    records = store.read_all()
    assert records[0]["rating"] == "down"
    assert records[0]["comment"] == "Мало источников"


def test_static_assets_exist_and_contain_contract():
    index = (web_server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (web_server.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    styles = (web_server.STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="ask-form"' in index
    assert 'id="mode-picker"' in index
    assert 'id="history-list"' in index
    assert 'id="key-dialog"' in index
    assert "/ui/app.js" in index
    assert "/api/ask" in script
    assert "/api/feedback" in script
    assert "localStorage" in script
    assert "mode-card" in styles
