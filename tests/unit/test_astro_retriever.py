"""Тесты ретривер-адаптера режима 5 для eval_grounding (H2-6)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import grounding_astro_retriever as gar  # noqa: E402


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


def _patch_retriever(monkeypatch, rows, store=None, captured=None):
    store = store or FakeStore({})
    monkeypatch.setattr(gar, "_engine", lambda: SimpleNamespace(chunk_embedding_store=store))

    def fake_search(engine, query, top_k=10):
        if captured is not None:
            captured["query"] = query
            captured["top_k"] = top_k
        return rows

    fake_module = types.ModuleType("astro_app.modes.bm25_hybrid")
    fake_module.search_bm25_hybrid_ru = fake_search
    monkeypatch.setitem(sys.modules, "astro_app.modes.bm25_hybrid", fake_module)


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


def test_retrieve_maps_ranked_rows_to_fragments(monkeypatch):
    captured = {}
    store = FakeStore(
        {
            "первый": ("Психологический Фэншуй", "Психологический фэншуй.pdf", 2),
            "второй": ("Эзотерический Фэншуй", "Эзотерический фэншуй.pdf", 149),
        }
    )
    _patch_retriever(monkeypatch, [("первый", 0.9), ("второй", 0.5)], store, captured)

    fragments = gar.retrieve("кто автор книги", top_k=3)

    assert [fragment["page_num"] for fragment in fragments] == [2, 149]
    assert captured == {"query": "кто автор книги", "top_k": 3}


def test_retrieve_top_k_defaults_to_env(monkeypatch):
    captured = {}
    _patch_retriever(monkeypatch, [("чанк", 1.0)], captured=captured)
    monkeypatch.setenv("ASTRO_RETRIEVER_TOP_K", "8")

    gar.retrieve("вопрос")

    assert captured["top_k"] == 8


def test_retrieve_returns_empty_list_for_no_hits(monkeypatch):
    _patch_retriever(monkeypatch, [])

    assert gar.retrieve("вопрос") == []


def test_top_k_env_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("ASTRO_RETRIEVER_TOP_K", "0")

    with pytest.raises(ValueError):
        gar.retriever_top_k()


def test_top_k_default_is_five(monkeypatch):
    monkeypatch.delenv("ASTRO_RETRIEVER_TOP_K", raising=False)

    assert gar.retriever_top_k() == 5
