"""Тесты ретривер-адаптера режима 5 для eval_grounding (H2-6, H1-6)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import grounding_astro_retriever as gar  # noqa: E402

from astro_app.page_retrieval import PageHit, PageRecord  # noqa: E402


class FakeStore:
    """Минимальное хранилище чанков: текст -> метаданные."""

    def __init__(self, mapping):
        self.text_to_hash_id = {text: f"chunk-{index}" for index, text in enumerate(mapping)}
        self._meta = {
            f"chunk-{index}": [{"doc_title": meta[0], "source_file": meta[1], "page_num": meta[2]}]
            for index, meta in enumerate(mapping.values())
        }

    def get_all_metadata_entries(self, hash_id):
        return self._meta.get(hash_id, [])


def _fake_module(**functions):
    module = types.ModuleType("astro_app.modes.bm25_hybrid")
    for name, function in functions.items():
        setattr(module, name, function)
    return module


def _patch_chunk_retriever(monkeypatch, rows, store=None, captured=None):
    """Чанковый путь: page-level явно выключен, как в baseline H2-6."""
    store = store or FakeStore({})
    monkeypatch.setattr(gar, "_engine", lambda: SimpleNamespace(chunk_embedding_store=store))
    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", "0")

    def fake_search(engine, query, top_k=10):
        if captured is not None:
            captured["query"] = query
            captured["top_k"] = top_k
        return rows

    def page_search(*args, **kwargs):  # pragma: no cover - не должен вызываться
        raise AssertionError("page-level не должен вызываться при ASTRO_RETRIEVER_PAGE_LEVEL=0")

    monkeypatch.setitem(
        sys.modules,
        "astro_app.modes.bm25_hybrid",
        _fake_module(
            search_bm25_hybrid_ru=fake_search,
            search_bm25_hybrid_ru_pages=page_search,
        ),
    )


def _page(source_file: str, doc_title: str, page_num: int, text: str) -> PageHit:
    record = PageRecord(
        source_file=source_file,
        doc_title=doc_title,
        page_num=page_num,
        section="",
        text=text,
        chunk_indices=(0,),
    )
    return PageHit(page=record, text=text, score=1.0)


def _patch_page_retriever(monkeypatch, hits, store=None, captured=None):
    """Page-level путь по умолчанию (H1-6)."""
    store = store or FakeStore({})
    monkeypatch.setattr(gar, "_engine", lambda: SimpleNamespace(chunk_embedding_store=store))
    monkeypatch.delenv("ASTRO_RETRIEVER_PAGE_LEVEL", raising=False)

    def fake_pages(engine, query, top_k=5, **kwargs):
        if captured is not None:
            captured["query"] = query
            captured["top_k"] = top_k
            captured.update(kwargs)
        return hits

    def chunk_search(*args, **kwargs):  # pragma: no cover - не должен вызываться
        raise AssertionError("чанковый путь не должен вызываться при page-level по умолчанию")

    monkeypatch.setitem(
        sys.modules,
        "astro_app.modes.bm25_hybrid",
        _fake_module(
            search_bm25_hybrid_ru=chunk_search,
            search_bm25_hybrid_ru_pages=fake_pages,
        ),
    )


def test_fragment_uses_first_metadata_entry_and_page():
    store = FakeStore({"текст": ("Астрофэншуй Квартиры", "Астрофэншуй квартиры.pdf", 7)})

    fragment = gar.fragment_from_chunk(store, "текст")

    assert fragment["text"] == "текст"
    assert fragment["doc_title"] == "Астрофэншуй Квартиры"
    assert fragment["source_file"] == "Астрофэншуй квартиры.pdf"
    assert fragment["page_num"] == 7


def test_fragment_without_known_text_has_no_source():
    store = FakeStore({})

    fragment = gar.fragment_from_chunk(store, "неизвестный чанк")

    assert fragment["doc_title"] == ""
    assert fragment["page_num"] == 0


def test_fragment_from_page_keeps_page_origin():
    hit = _page("Психологический фэншуй.pdf", "Психологический Фэншуй", 2, "полный текст")

    fragment = gar.fragment_from_page(hit)

    assert fragment == {
        "text": "полный текст",
        "doc_title": "Психологический Фэншуй",
        "page_num": 2,
        "section": "",
        "source_file": "Психологический фэншуй.pdf",
    }


def test_retrieve_chunk_mode_maps_ranked_rows_to_fragments(monkeypatch):
    captured = {}
    store = FakeStore(
        {
            "первый": ("Психологический Фэншуй", "Психологический фэншуй.pdf", 2),
            "второй": ("Эзотерический Фэншуй", "Эзотерический фэншуй.pdf", 149),
        }
    )
    _patch_chunk_retriever(monkeypatch, [("первый", 0.9), ("второй", 0.5)], store, captured)

    fragments = gar.retrieve("кто автор книги", top_k=3)

    assert [fragment["page_num"] for fragment in fragments] == [2, 149]
    assert captured == {"query": "кто автор книги", "top_k": 3}


def test_retrieve_page_mode_is_default_and_passes_settings(monkeypatch):
    captured = {}
    hits = [
        _page("Психологический фэншуй.pdf", "Психологический Фэншуй", 2, "стр. 2"),
        _page("Астрофэншуй квартиры.pdf", "Астрофэншуй Квартиры", 7, "стр. 7"),
    ]
    _patch_page_retriever(monkeypatch, hits, captured=captured)
    monkeypatch.setenv("ASTRO_RETRIEVER_RERANK_POOL", "12")
    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_MAX_CHARS", "900")

    fragments = gar.retrieve("кто автор книги", top_k=2)

    assert [fragment["page_num"] for fragment in fragments] == [2, 7]
    assert [fragment["text"] for fragment in fragments] == ["стр. 2", "стр. 7"]
    assert captured == {
        "query": "кто автор книги",
        "top_k": 2,
        "rerank_pool": 12,
        "max_chars": 900,
    }


def test_retrieve_top_k_defaults_to_env(monkeypatch):
    captured = {}
    _patch_chunk_retriever(monkeypatch, [("чанк", 1.0)], captured=captured)
    monkeypatch.setenv("ASTRO_RETRIEVER_TOP_K", "8")

    gar.retrieve("вопрос")

    assert captured["top_k"] == 8


def test_retrieve_returns_empty_list_for_no_hits(monkeypatch):
    _patch_page_retriever(monkeypatch, [])

    assert gar.retrieve("вопрос") == []


def test_page_level_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", "0")
    assert gar.page_level_enabled() is False

    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", "false")
    assert gar.page_level_enabled() is False

    monkeypatch.setenv("ASTRO_RETRIEVER_PAGE_LEVEL", "1")
    assert gar.page_level_enabled() is True


def test_top_k_env_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("ASTRO_RETRIEVER_TOP_K", "0")

    with pytest.raises(ValueError):
        gar.retriever_top_k()


def test_top_k_default_is_five(monkeypatch):
    monkeypatch.delenv("ASTRO_RETRIEVER_TOP_K", raising=False)

    assert gar.retriever_top_k() == 5
