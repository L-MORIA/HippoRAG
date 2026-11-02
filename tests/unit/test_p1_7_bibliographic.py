"""P1-7: библиографические вопросы уходят с sentence-level на chunk-level.

Название книги — не предложение, поэтому sentence-level поиск структурно
не может его найти. Для вопросов «какие книги / кто автор / какие издания»
метод 20 должен переключаться на chunk-level поиск (P1-7).
"""

import pytest


@pytest.mark.parametrize(
    "query",
    [
        "какие книги написала Максимова И.В.?",
        "какую книгу написал автор?",
        "названия книг этого автора",
        "кто автор этой книги?",
        "что написал Пушкин",
        "какие произведения есть у автора",
        "список книг по фэншуй",
        "библиография Максимовой",
        "какая книга про астрофэншуй",
    ],
)
def test_bibliographic_queries_detected(query):
    import main

    assert main._is_bibliographic_query(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "что такое Багуа?",
        "какая зона подходит для Овна?",
        "какая планета отвечает за плиту?",
        "что важно для Рака в квартире?",
    ],
)
def test_content_queries_not_bibliographic(query):
    import main

    assert main._is_bibliographic_query(query) is False


def test_sentence_level_delegates_bibliographic_query(monkeypatch):
    import main

    sentinel = [("chunk-level результат", 1.0, "book.pdf", 7, 0, 0.0)]
    calls = {}

    def fake_chunk_level(hipporag, query, top_k=10, candidate_pool=200):
        calls["query"] = query
        calls["top_k"] = top_k
        return sentinel

    monkeypatch.setattr(main, "search_bm25_hybrid_ru", fake_chunk_level)

    # hipporag намеренно пустой: при раннем возврате он не должен понадобиться.
    result = main.search_sentence_level(object(), "какие книги написала Максимова?", top_k=7)

    assert result == sentinel
    assert calls == {"query": "какие книги написала Максимова?", "top_k": 7}
