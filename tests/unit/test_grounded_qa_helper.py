"""Тесты общего helper'а генеративных режимов CLI (H2-6, шаг 7).

Без сети и LM Studio: `chat` подменяется функцией, которая возвращает
заранее заданный ответ. Проверяем, что все режимы получают один контракт:
нумерованные фрагменты, проверка ссылок и общий отказ.
"""

from astro_app.grounded_qa import (
    Fragment,
    answer_grounded,
    build_messages,
    used_sources,
)
from hipporag.qa.grounding import ABSTAIN_TEXT


def _fragments():
    return [
        Fragment(text="Прибор «Альфа» работает от батареи AA.", doc_title="Книга A", page_num=1),
        Fragment(text="Корпус прибора изготовлен из пластика.", doc_title="Книга B", page_num=2),
    ]


def test_build_messages_numbers_fragments_and_uses_system_contract():
    messages = build_messages("От чего работает прибор?", _fragments())

    assert messages[0]["role"] == "system"
    assert "ОТВЕТ:" in messages[0]["content"]
    assert "[1] Источник: Книга A, стр. 1" in messages[1]["content"]
    assert "[2] Источник: Книга B, стр. 2" in messages[1]["content"]
    assert messages[1]["content"].rstrip().endswith("Вопрос: От чего работает прибор?")


def test_answer_grounded_accepts_valid_citation():
    def chat(messages):  # noqa: ARG001
        return "ОТВЕТ: Прибор работает от батареи AA [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет"

    grounded = answer_grounded("От чего работает прибор?", _fragments(), chat=chat)

    assert grounded.text == "Прибор работает от батареи AA [1]."
    assert grounded.used_citation_ids == [1]
    assert grounded.abstained is False
    assert grounded.errors == []


def test_answer_grounded_abstains_without_citations():
    def chat(messages):  # noqa: ARG001
        return "ОТВЕТ: Прибор работает от батареи AA.\nИСТОЧНИКИ:\nОТКАЗ: нет"

    grounded = answer_grounded("От чего работает прибор?", _fragments(), chat=chat)

    assert grounded.text == ABSTAIN_TEXT
    assert grounded.abstained is True
    assert "no_inline_citations" in grounded.errors


def test_answer_grounded_does_not_call_chat_without_fragments():
    calls = []

    def chat(messages):
        calls.append(messages)
        return "ОТВЕТ: что-то [1]"

    grounded = answer_grounded("Вопрос?", [], chat=chat)

    assert calls == []
    assert grounded.abstained is True
    assert grounded.text == ABSTAIN_TEXT


def test_used_sources_lists_only_cited_fragments():
    text = used_sources(_fragments(), [2])

    assert "[2] Книга B, стр. 2" in text
    assert "Книга A" not in text
