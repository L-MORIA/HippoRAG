"""Тесты контракта заземления в API `/search` (H2-6, шаг 6).

Сервер не поднимается: `hipporag.rag_qa` подменяется стабом, а обработчик
вызывается напрямую. Проверяем то, что видит клиент: какие цитаты уходят,
что означает `confidence` и как выглядит отказ.
"""

import asyncio

from hipporag.qa.grounding import ABSTAIN_TEXT
from hipporag.search_api import server
from hipporag.search_api.models import SearchRequest
from hipporag.utils.custom_types import Citation, QuerySolution


class _StubRag:
    def __init__(self, solution):
        self.solution = solution

    def rag_qa(self, queries):  # noqa: ARG002
        return [self.solution], [], [{}]


def _call_search(monkeypatch, solution):
    monkeypatch.setattr(server, "hipporag", _StubRag(solution))
    return asyncio.run(server.search(SearchRequest(question="От чего работает прибор?")))


def test_search_returns_only_used_citations(monkeypatch):
    used = Citation(source_file="a.pdf", doc_title="Книга A", page_num=1)
    unused = Citation(source_file="b.pdf", doc_title="Книга B", page_num=2)
    solution = QuerySolution(
        question="От чего работает прибор?",
        answer="Прибор работает от батареи AA [1].",
        docs=["фрагмент один", "фрагмент два"],
        citations=[used, unused],
        used_citations=[used],
        abstained=False,
        grounding_errors=[],
    )
    response = _call_search(monkeypatch, solution)

    assert len(response.data.citations) == 1
    assert response.data.citations[0].doc_title == "Книга A"
    assert response.data.abstained is False
    assert response.data.grounding_errors == []
    assert response.data.confidence == 1.0
    # Диагностика retrieval сохраняется полностью.
    assert response.data.retrieved_docs == ["фрагмент один", "фрагмент два"]


def test_search_abstention_has_no_citations_and_zero_confidence(monkeypatch):
    solution = QuerySolution(
        question="Сколько стоит прибор?",
        answer=ABSTAIN_TEXT,
        docs=["Прибор работает от батареи AA."],
        citations=[Citation(source_file="a.pdf", doc_title="Книга A", page_num=1)],
        used_citations=[],
        abstained=True,
        grounding_errors=["no_inline_citations"],
    )
    response = _call_search(monkeypatch, solution)

    assert response.data.answer == ABSTAIN_TEXT
    assert response.data.abstained is True
    assert response.data.citations == []
    assert response.data.grounding_errors == ["no_inline_citations"]
    assert response.data.confidence == 0.0


def test_search_grounding_errors_lower_confidence(monkeypatch):
    used = Citation(source_file="a.pdf", doc_title="Книга A", page_num=1)
    solution = QuerySolution(
        question="От чего работает прибор?",
        answer="Прибор работает от батареи AA [1]. Ещё одно утверждение без ссылки.",
        docs=["Прибор работает от батареи AA."],
        citations=[used],
        used_citations=[used],
        abstained=False,
        grounding_errors=["uncited_sentences:1"],
    )
    response = _call_search(monkeypatch, solution)

    assert response.data.citations[0].doc_title == "Книга A"
    assert response.data.confidence == 0.5
