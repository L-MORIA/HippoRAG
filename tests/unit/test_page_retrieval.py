"""Тесты page-level гибридного ретрива (H1-6)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from astro_app import page_retrieval as pr


class FakeStore:
    """Хранилище чанков с метаданными в формате EmbeddingStore."""

    def __init__(self, chunks, embeddings=None):
        self.texts = [chunk["text"] for chunk in chunks]
        self.hash_ids = [f"chunk-{index}" for index in range(len(chunks))]
        self.metadatas = [chunk.get("metadata", []) for chunk in chunks]
        self.embeddings = embeddings if embeddings is not None else []

    def get_all_metadata_entries(self, hash_id):
        index = self.hash_ids.index(hash_id)
        return list(self.metadatas[index])


class FakeRowStore:
    """Хранилище без get_all_metadata_entries: метаданные в get_row."""

    def __init__(self, rows):
        self.texts = [row["text"] for row in rows]
        self.hash_ids = [f"row-{index}" for index in range(len(rows))]
        self.embeddings = []
        self._rows = {
            f"row-{index}": {"metadata": row["metadata"]} for index, row in enumerate(rows)
        }

    def get_row(self, hash_id):
        return self._rows[hash_id]


class FakeEmbeddingModel:
    def __init__(self, vector):
        self.vector = np.asarray([vector], dtype=float)

    def encode(self, texts):
        return self.vector


class FakeHippoRAG:
    def __init__(self, store, vector):
        self.chunk_embedding_store = store
        self.embedding_model = FakeEmbeddingModel(vector)


@pytest.fixture(autouse=True)
def _reset_cache():
    pr.reset_page_index_cache()
    yield
    pr.reset_page_index_cache()


def _meta(source_file="doc.pdf", doc_title="Doc", page_num=1, section=""):
    return [
        {
            "source_file": source_file,
            "doc_title": doc_title,
            "page_num": page_num,
            "section": section,
        }
    ]


def _chunk(text, page_num, source_file="doc.pdf", doc_title="Doc"):
    return {"text": text, "metadata": _meta(source_file, doc_title, page_num)}


def _identity_rerank(query, candidates):
    return candidates


def test_page_index_groups_chunks_by_source_and_page():
    store = FakeStore(
        [
            _chunk("первый фрагмент", 1),
            _chunk("второй фрагмент", 1),
            _chunk("другая страница", 2),
        ]
    )

    index = pr._build_page_index(store)

    assert [page.page_num for page in index.pages] == [1, 2]
    assert index.pages[0].text == "первый фрагмент\n\nвторой фрагмент"
    assert index.pages[0].chunk_indices == (0, 1)
    assert index.pages[1].chunk_indices == (2,)
    assert index.bm25 is not None


def test_page_index_falls_back_to_get_row_json_metadata():
    store = FakeRowStore(
        [
            {"text": "текст", "metadata": json.dumps({"source_file": "x.pdf", "page_num": 3})},
        ]
    )

    index = pr._build_page_index(store)

    assert len(index.pages) == 1
    assert index.pages[0].source_file == "x.pdf"
    assert index.pages[0].doc_title == "x.pdf"
    assert index.pages[0].page_num == 3


def test_page_index_skips_chunks_without_page():
    store = FakeStore(
        [
            {"text": "без метаданных", "metadata": []},
            {"text": "страница ноль", "metadata": _meta(page_num=0)},
            _chunk("нормальная страница", 1),
        ]
    )

    index = pr._build_page_index(store)

    assert [page.text for page in index.pages] == ["нормальная страница"]


def test_page_index_cache_is_reused_for_same_store():
    store = FakeStore([_chunk("текст", 1)])

    first = pr._build_page_index(store)
    second = pr._build_page_index(store)

    assert first is second


def test_ranks_are_deterministic_for_ties():
    assert pr._ranks([1.0, 1.0, 2.0]) == [1, 2, 0]
    assert pr._ranks([3.0, 2.0, 1.0]) == [0, 1, 2]


def test_truncate_page_centers_on_best_chunk():
    best = "нужный фрагмент"
    text = "начало " * 50 + best + " конец" * 50

    truncated = pr._truncate_page(text, best, max_chars=80)

    assert len(truncated) <= 80
    assert best in truncated


def test_truncate_page_keeps_short_text():
    text = "короткая страница"

    assert pr._truncate_page(text, text, max_chars=100) == text


def test_search_pages_hybrid_deduplicates_pages_and_takes_dense_max(monkeypatch):
    monkeypatch.setattr(pr, "_rerank_with_crossencoder_ru", _identity_rerank)
    store = FakeStore(
        [
            _chunk("марс первый", 1),
            _chunk("марс второй", 1),
            _chunk("венера", 2),
        ],
        embeddings=[[0.1, 0.0], [0.9, 0.0], [0.0, 1.0]],
    )
    hipporag = FakeHippoRAG(store, [1.0, 0.0])

    hits = pr.search_pages_hybrid(hipporag, "марс", top_k=5)

    assert [hit.page.page_num for hit in hits] == [1, 2]
    assert hits[0].dense_score == pytest.approx(0.9)
    assert hits[0].best_chunk_index == 1


def test_search_pages_hybrid_works_without_dense_channel(monkeypatch):
    monkeypatch.setattr(pr, "_rerank_with_crossencoder_ru", _identity_rerank)
    store = FakeStore([_chunk("марс", 1), _chunk("венера", 2)])
    hipporag = FakeHippoRAG(store, [1.0, 0.0])

    hits = pr.search_pages_hybrid(hipporag, "марс", top_k=5)

    assert hits
    assert hits[0].page.page_num == 1
    assert hits[0].dense_score == 0.0


def test_search_pages_hybrid_fills_top_k_beyond_rerank_pool(monkeypatch):
    captured = {}

    def fake_rerank(query, candidates):
        captured["pool"] = len(candidates)
        return candidates

    monkeypatch.setattr(pr, "_rerank_with_crossencoder_ru", fake_rerank)
    store = FakeStore(
        [
            _chunk("марс один", 1),
            _chunk("марс два", 2),
            _chunk("марс три", 3),
        ]
    )
    hipporag = FakeHippoRAG(store, [1.0, 0.0])

    hits = pr.search_pages_hybrid(hipporag, "марс", top_k=2, rerank_pool=1)

    assert captured["pool"] == 1
    assert len(hits) == 2
    assert hits[0].page.page_num == 1
    assert hits[1].page.page_num in {2, 3}


def test_search_pages_hybrid_returns_empty_for_empty_store():
    store = FakeStore([])
    hipporag = FakeHippoRAG(store, [1.0, 0.0])

    assert pr.search_pages_hybrid(hipporag, "марс", top_k=5) == []


def test_search_pages_hybrid_respects_top_k(monkeypatch):
    monkeypatch.setattr(pr, "_rerank_with_crossencoder_ru", _identity_rerank)
    store = FakeStore([_chunk(f"марс {index}", index) for index in range(1, 6)])
    hipporag = FakeHippoRAG(store, [1.0, 0.0])

    hits = pr.search_pages_hybrid(hipporag, "марс", top_k=2)

    assert len(hits) == 2
