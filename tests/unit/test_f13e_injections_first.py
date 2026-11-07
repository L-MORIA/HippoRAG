"""F-13d: инъекции (таблица гуа, уточнение автора) идут первыми в выдаче."""

from __future__ import annotations

import pytest

from astro_app.web import service


class FakeStore:
    def __init__(self, rows, texts):
        self._rows = rows
        self.texts = texts
        self.hash_ids = [f"h{i}" for i in range(len(texts))]

    def get_row(self, hash_id):
        return self._rows[hash_id]


class FakeHippoRAG:
    def __init__(self, store):
        self.chunk_embedding_store = store


@pytest.fixture
def fake_hipporag():
    texts = [f"текст про пятёрки {i}" for i in range(2)]
    rows = {
        f"h{i}": {
            "metadata": [
                {
                    "source_file": "Эзотерический фэншуй.pdf",
                    "doc_title": "Эзотерический Фэншуй",
                    "page_num": page,
                    "section": "",
                }
            ]
        }
        for i, page in enumerate((38, 120))
    }
    return FakeHippoRAG(FakeStore(rows, texts))


@pytest.fixture
def chunk_mode(monkeypatch):
    def fn(_hipporag, _query, top_k=5):
        return [(f"текст про пятёрки {i}", 1.0 - i * 0.1, 0.0, 0.0, 0.0) for i in range(2)]

    monkeypatch.setitem(
        service.MODES, "5", service.MODES["5"].__class__(label="т", menu="т", fn=fn)
    )
    return fn


def _texts(result):
    return [f.text[:40] for f in result.fragments]


def test_injections_precede_book_fragments(fake_hipporag, chunk_mode, monkeypatch):
    result = service.retrieve(
        fake_hipporag,
        "Какое у меня число гуа если я родилась 15.07.1972 женщина",
        mode_id="5",
        top_k=5,
    )
    texts = _texts(result)
    assert texts[0].startswith("Для даты рождения 15.07.1972")
    assert texts[1].startswith("Разделение направлений")
    assert result.fragments[0].source_file == "book_table"
    assert len(result.fragments) == 4  # 2 инъекции + 2 фрагмента книги


def test_no_injections_keeps_book_order(fake_hipporag, chunk_mode):
    result = service.retrieve(fake_hipporag, "зоны овна", mode_id="5", top_k=5)
    assert len(result.fragments) == 2
    assert result.fragments[0].source_file == "Эзотерический фэншуй.pdf"
