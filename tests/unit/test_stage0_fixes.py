"""Приёмо-сдаточные тесты Этапа 0 сводного аудита (S-08, S-12, S-20)."""

import numpy as np
import pytest

from hipporag.embedding_store import EmbeddingStore
from hipporag.pdf_ingestion.chunker import chunk_text_with_metadata


class FakeEmbeddingModel:
    def batch_encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(
        FakeEmbeddingModel(), str(tmp_path / "chunks"), batch_size=8, namespace="chunk"
    )


def test_same_text_different_documents_keep_metadata(store):
    """S-12: идентичный текст из двух PDF — оба источника сохраняются И попадают в цитату."""
    text = "Одинаковый колонтитул страницы"
    store.insert_strings([text], [{"source_file": "A.pdf", "doc_title": "A", "page_num": 1}])
    store.insert_strings([text], [{"source_file": "B.pdf", "doc_title": "B", "page_num": 7}])

    hash_id = store.text_to_hash_id[text]
    entries = store.get_all_metadata_entries(hash_id)
    assert [e["source_file"] for e in entries] == ["A.pdf", "B.pdf"]

    # D-1/D-8: цитата обязана показывать ВСЕ источники фрагмента, а не только первый —
    # иначе исходный дефект «цитата указывает не на тот документ» сохраняется
    from hipporag.utils.misc_utils import build_citation_from_metadata

    citation = build_citation_from_metadata(entries, text)
    assert set(citation.sources) == {"A.pdf", "B.pdf"}
    assert citation.source_file == "A.pdf"  # первичный (первый известный) источник


def test_same_text_same_metadata_not_duplicated(store):
    """Повторная вставка тех же метаданных не размножает записи."""
    text = "Одинаковый колонтитул страницы"
    meta = {"source_file": "A.pdf", "doc_title": "A", "page_num": 1}
    store.insert_strings([text], [meta])
    store.insert_strings([text], [dict(meta)])

    hash_id = store.text_to_hash_id[text]
    assert len(store.get_all_metadata_entries(hash_id)) == 1


def test_metadata_without_source_does_not_create_entry(store):
    """Пустые метаданные повторной вставки ничего не меняют."""
    text = "Текст без источника"
    store.insert_strings([text], [{"source_file": "A.pdf"}])
    store.insert_strings([text], [{}])

    hash_id = store.text_to_hash_id[text]
    assert len(store.get_all_metadata_entries(hash_id)) == 1


def test_metadata_survives_parquet_roundtrip(tmp_path):
    """Метаданные-список переживают сохранение/загрузку parquet; старый формат тоже читается."""
    db_dir = str(tmp_path / "chunks")
    store = EmbeddingStore(FakeEmbeddingModel(), db_dir, batch_size=8, namespace="chunk")
    text = "Одинаковый колонтитул страницы"
    store.insert_strings([text], [{"source_file": "A.pdf", "page_num": 1}])
    store.insert_strings([text], [{"source_file": "B.pdf", "page_num": 7}])

    reloaded = EmbeddingStore(FakeEmbeddingModel(), db_dir, batch_size=8, namespace="chunk")
    hash_id = reloaded.text_to_hash_id[text]
    assert [e["source_file"] for e in reloaded.get_all_metadata_entries(hash_id)] == [
        "A.pdf",
        "B.pdf",
    ]
    assert isinstance(reloaded.get_metadata(hash_id), dict)


def test_chunker_does_not_invent_page_numbers():
    """S-20: чанки одной страницы наследуют её номер без выдуманного смещения."""
    text = ("Очень длинное предложение про фэншуй и астрологию жилища. " * 300).strip()
    chunks = chunk_text_with_metadata(
        text=text, source_file="t.pdf", doc_title="T", page_num=3, section=None, config=None
    )
    assert len(chunks) > 1
    assert all(c.page_num == 3 for c in chunks)


def test_clean_database_refuses_path_traversal(tmp_path):
    """S-08: удаление вне базовой директории отклоняется."""
    from clean_database import DatabaseCleaner

    base = tmp_path / "outputs"
    base.mkdir()
    (base / "astrology").mkdir()
    cleaner = DatabaseCleaner(str(base))

    with pytest.raises(ValueError):
        cleaner._validate_inside_base(base / "..")
    with pytest.raises(ValueError):
        cleaner._validate_inside_base(base)

    resolved = cleaner._validate_inside_base(base / "astrology")
    assert resolved == (base / "astrology").resolve()
