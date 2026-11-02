"""P1-6: Self-RAG обязан подмешивать dense-поиск в кандидаты.

Раньше кандидатов отбирал чистый BM25: на вопрос «какие книги написала
Максимова И.В.?» он поднимал прозу («написала я книгу»), а титульная страница
с названием книги не попадала в источники LLM — и LLM честно отвечал
«в источниках нет названий». Тест проверяет, что чанк, найденный только
dense-каналом, доходит до текста источников.
"""

import numpy as np
import pytest

TEXTS = [
    "пример: написала я книги о том, как обустроить дом",
    "психологический фэншуй как прокачать плохой фэншуй",
    "астрофэншуй квартиры введение общие принципы",
    "ремонт и отделка практические советы",
    "знаки зодиака и зоны квартиры овен плита",
    "багуа и секторы квартиры",
]
# Чанк 1 (титульная страница) не содержит слов вопроса: его видит только dense.
EMBEDDINGS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
QUERY_VECTOR = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)


class FakeStore:
    def __init__(self, texts, embeddings):
        self.texts = texts
        self.embeddings = embeddings
        self.hash_ids = [f"h{i}" for i in range(len(texts))]
        # Страницы далеки друг от друга: контекстное окно не должно склеивать чанки.
        self.pages = [10 * (i + 1) for i in range(len(texts))]

    def get_row(self, hash_id):
        idx = int(hash_id[1:])
        return {
            "hash_id": hash_id,
            "content": self.texts[idx],
            "metadata": [
                {
                    "source_file": "book.pdf",
                    "doc_title": "book",
                    "page_num": self.pages[idx],
                    "section": "",
                }
            ],
        }


class FakeEmbeddingModel:
    def encode(self, texts):
        return QUERY_VECTOR


class FakeHippoRAG:
    def __init__(self, store):
        self.chunk_embedding_store = store
        self.embedding_model = FakeEmbeddingModel()


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_llm(monkeypatch):
    """LLM всегда «отвечает», чтобы увидеть полный текст источников."""

    def fake_get(url, timeout=3):
        return FakeResponse({"data": []})

    def fake_post(url, json=None, timeout=300):
        assert json and json["messages"], "промпт должен содержать источники"
        return FakeResponse(
            {"choices": [{"message": {"content": "Книга: Психологический фэншуй."}}]}
        )

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr("httpx.post", fake_post)


def test_self_rag_reaches_dense_only_chunk(fake_llm):
    import main

    store = FakeStore(TEXTS, EMBEDDINGS)
    hipporag = FakeHippoRAG(store)

    results = main.search_self_rag(
        hipporag, "какие книги написала Максимова И.В.?", top_k=30, candidate_pool=2
    )

    assert results, "Self-RAG не вернул ни одного результата"
    text = results[0][0]
    assert "психологический фэншуй" in text.lower(), (
        "титульная страница не дошла до источников: кандидаты снова отобраны " "только по BM25"
    )


def test_self_rag_works_without_embeddings(monkeypatch, fake_llm):
    """Без dense-канала метод остаётся работоспособным (BM25-only)."""
    import main

    store = FakeStore(TEXTS, np.zeros((0, 3), dtype=np.float32))
    hipporag = FakeHippoRAG(store)

    results = main.search_self_rag(hipporag, "багуа и секторы квартиры", top_k=5)

    assert results
    assert results[0][0]
