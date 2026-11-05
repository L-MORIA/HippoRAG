"""Контракт H1-6: page-level режим 5 — путь по умолчанию (UI и CLI).

После подключения page-level пути к продукту должно быть правдой:

* ``MODES["5"].fn`` — диспетчер ``bm25_hybrid_ru_mode``: по умолчанию
  page-level (``PageHit``), при ``ASTRO_RETRIEVER_PAGE_LEVEL=0`` — прежний
  чанковый путь;
* чанковая функция ``search_bm25_hybrid_ru`` не меняется: её используют
  режимы 19/21 и eval-адаптеры;
* консольный дисплей и веб-сервис понимают ``PageHit`` (источник и страница
  в выдаче).

Сброс кэша page-индекса в lifecycle проверяется в ``test_index_lifecycle.py``
(фикстура ``index_env`` живёт там).
"""

from __future__ import annotations

import pytest

from astro_app.display import show_results
from astro_app.modes import DEFAULT_MODE, MODES, bm25_hybrid
from astro_app.page_retrieval import PageHit, PageRecord, page_level_enabled
from astro_app.web import service as web_service

PAGE = PageRecord(
    source_file="Астрофэншуй квартиры.pdf",
    doc_title="Астрофэншуй Квартиры",
    page_num=7,
    section="",
    text="Овну соответствуют прихожая и плита, управитель — Марс.",
    chunk_indices=(0,),
)


def _hit() -> PageHit:
    return PageHit(page=PAGE, text=PAGE.text, score=0.42, bm25_score=1.0, dense_score=0.5)


# ── диспетчер режима 5 ───────────────────────────────────────────────────────


def test_mode5_fn_is_page_level_dispatcher():
    assert MODES[DEFAULT_MODE].fn is bm25_hybrid.bm25_hybrid_ru_mode


def test_dispatcher_uses_pages_by_default(monkeypatch):
    calls = []
    sentinel = [_hit()]

    def fake_pages(_hipporag, _query, top_k=None, **kwargs):
        calls.append({"top_k": top_k, "kwargs": kwargs})
        return sentinel

    monkeypatch.setattr(bm25_hybrid, "search_pages_hybrid", fake_pages)

    result = bm25_hybrid.bm25_hybrid_ru_mode(object(), "овен зоны", top_k=5)

    assert result is sentinel
    assert calls == [{"top_k": 5, "kwargs": {}}]


def test_dispatcher_env_off_uses_chunk_path(monkeypatch):
    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", "0")
    calls = []
    sentinel = [("чанк", 1.0, 0.0, 0.0, 0.0)]

    def fake_chunks(_hipporag, _query, top_k=None, candidate_pool=None):
        calls.append({"top_k": top_k, "candidate_pool": candidate_pool})
        return sentinel

    monkeypatch.setattr(bm25_hybrid, "search_bm25_hybrid_ru", fake_chunks)

    result = bm25_hybrid.bm25_hybrid_ru_mode(object(), "овен зоны", top_k=5)

    assert result is sentinel
    assert calls == [{"top_k": 5, "candidate_pool": 200}]


def test_page_level_enabled_env_matrix(monkeypatch):
    monkeypatch.delenv("ASTRO_RETRIEVER_PAGE_LEVEL", raising=False)
    assert page_level_enabled() is True
    for raw in ("0", "false", "no", "off"):
        monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", raw)
        assert page_level_enabled() is False
    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", "1")
    assert page_level_enabled() is True


# ── отображение PageHit ──────────────────────────────────────────────────────


def test_show_results_renders_page_hit(capsys):
    show_results("овен зоны", [_hit()], "BM25+CE-ru")

    out = capsys.readouterr().out
    assert "Астрофэншуй Квартиры" in out
    assert "стр.7" in out
    assert "score: 0.4200" in out
    assert "Марс" in out


# ── веб-сервис принимает PageHit ─────────────────────────────────────────────


def test_web_service_normalizes_page_hit():
    fragment = web_service._fragment_from_result(object(), _hit())

    assert fragment is not None
    assert fragment.doc_title == "Астрофэншуй Квартиры"
    assert fragment.page_num == 7
    assert fragment.source_file == "Астрофэншуй квартиры.pdf"
    assert fragment.score == pytest.approx(0.42)
