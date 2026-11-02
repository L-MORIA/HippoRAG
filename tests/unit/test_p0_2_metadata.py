"""Регрессия P0-2: _get_chunk_meta обязан понимать формат метаданных S-12.

Фикс S-12 изменил метаданные чанка с dict на список dict'ов (по одному на
каждый источник документа). Потребитель в main.py не был обновлён, из-за
чего 8 из 23 режимов меню возвращали пустоту молча.

Эти тесты закрывают обе стороны:
- новый формат (список) — должен давать source_file и page_num;
- старый формат (dict) и JSON-строка — тоже должны работать.
"""

import json

import numpy as np
import pytest

from hipporag.embedding_store import EmbeddingStore


class FakeEmbeddingModel:
    def batch_encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeStore:
    """Минимальный аналог EmbeddingStore с заданным форматом метаданных."""

    def __init__(self, metadata):
        self.hash_ids = ["h0"]
        self._metadata = metadata

    def get_row(self, hash_id):
        return {"hash_id": hash_id, "content": "text", "metadata": self._metadata}


@pytest.fixture
def get_chunk_meta():
    import main

    return main._get_chunk_meta


@pytest.mark.parametrize(
    "metadata",
    [
        [{"source_file": "A.pdf", "doc_title": "A", "page_num": 7, "section": ""}],
        {"source_file": "A.pdf", "doc_title": "A", "page_num": 7, "section": ""},
        '[{"source_file": "A.pdf", "doc_title": "A", "page_num": 7, "section": ""}]',
        '[{"source_file": "A.pdf", "doc_title": "A", "page_num": 7, "section": ""}]'.replace(
            "'", '"'
        ),
    ],
    ids=["list_S12", "dict_legacy", "json_string", "json_string_real"],
)
def test_get_chunk_meta_reads_all_formats(get_chunk_meta, metadata):
    """Метаданные читаются независимо от формата: список / dict / JSON-строка."""
    store = FakeStore(metadata)
    assert get_chunk_meta(store, 0) == ("A.pdf", 7)


def test_get_chunk_meta_takes_primary_source_of_list(get_chunk_meta):
    """Из списка источников берётся первичный (первый) — как в get_metadata()."""
    store = FakeStore(
        [
            {"source_file": "A.pdf", "page_num": 1},
            {"source_file": "B.pdf", "page_num": 9},
        ]
    )
    assert get_chunk_meta(store, 0) == ("A.pdf", 1)


def test_get_chunk_meta_survives_empty_and_broken(get_chunk_meta):
    """Пустые/битые метаданные не роняют функцию — возвращается пустой источник."""
    assert get_chunk_meta(FakeStore([]), 0) == ("", 0)
    assert get_chunk_meta(FakeStore({}), 0) == ("", 0)
    assert get_chunk_meta(FakeStore("не json"), 0) == ("", 0)
    assert get_chunk_meta(FakeStore(None), 0) == ("", 0)


def test_split_query_terms_strips_punctuation():
    """P2-10: пунктуация снимается — иначе «Овна?» не совпадает с «овна».

    Без этого BM25 и keyword-boost теряли все совпадения на любом запросе
    со знаком вопроса: токен получался «овна?» и не встречался в корпусе.
    """
    import main

    terms, _ = main.split_query_terms("какая зона подходит для Овна?")
    assert "овна" in terms
    assert "овна?" not in terms

    terms, _ = main.split_query_terms("что такое Багуа?")
    assert "багуа" in terms

    terms, _ = main.split_query_terms("какие книги написала Максимова И.В.?")
    assert "максимова" in terms
    assert not any("?" in t for t in terms)
    assert not any("." in t for t in terms)


def test_ce_scores_are_normalized_before_blending():
    """P1-4: CE-скор нормируется перед смешиванием с embedding-скором."""
    import inspect

    import numpy as np

    import main
    from hipporag.utils.misc_utils import min_max_normalize

    for fn in (main._rerank_with_crossencoder, main._rerank_with_crossencoder_ru):
        src = inspect.getsource(fn)
        assert "min_max_normalize" in src, f"{fn.__name__} не нормирует CE-скор"

    x = np.array([8.3, 8.4, 8.3, 8.9])
    n = min_max_normalize(x)
    assert n.min() == 0.0 and n.max() == 1.0
    assert list(min_max_normalize(np.array([5.0, 5.0]))) == [1.0, 1.0]


def test_meta_reformulation_detector():
    """Мета-ответы LLM («Запрос не содержит…») не должны попадать в эмбеддинг."""
    import main

    assert main._is_meta_reformulation(
        "Запрос не содержит астрологических или фэншуй-терминов для улучшения."
    )
    assert main._is_meta_reformulation("Невозможно улучшить запрос")
    assert main._is_meta_reformulation("The query does not contain astrology terms")
    assert not main._is_meta_reformulation("Максимова И.В. книги фэншуй")
    assert not main._is_meta_reformulation("плита прихожая Овен")


def test_qa_derive_short_fallback():
    """P2-8: короткий ответ выводится, даже если нет маркера 'ОТВЕТ:'."""
    import main

    text = "=== РЕЗУЛЬТАТ ПОИСКА ===\nЦелевая зона: плита\nНайден 1 вариант"
    assert main._qa_derive_short(text) == "Целевая зона: плита"
    assert main._qa_derive_short("") == ""
    assert main._qa_derive_short("=== ТОЛЬКО ЗАГОЛОВКИ ===") == ""


def test_embedding_store_metadata_roundtrip_is_list(tmp_path):
    """Хранилище отдаёт список источников — именно его и читает _get_chunk_meta."""
    store = EmbeddingStore(
        FakeEmbeddingModel(), str(tmp_path / "chunks"), batch_size=8, namespace="chunk"
    )
    text = "Текст чанка"
    store.insert_strings([text], [{"source_file": "A.pdf", "doc_title": "A", "page_num": 3}])
    hash_id = store.text_to_hash_id[text]
    row = store.get_row(hash_id)
    assert isinstance(row["metadata"], list)
    assert row["metadata"][0]["source_file"] == "A.pdf"
    assert row["metadata"][0]["page_num"] == 3
    # json-сериализация не ломается
    assert json.loads(json.dumps(row["metadata"]))[0]["source_file"] == "A.pdf"
