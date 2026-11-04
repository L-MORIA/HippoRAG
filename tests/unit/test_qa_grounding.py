"""Тесты контракта заземления Q&A (H2-6, шаг 1).

Проверяем чистый парсер без LLM и индекса: ссылки, отказ, code fences,
старый формат и защиту от «ответа без доказательства».
"""

import pytest

from hipporag.qa.grounding import (
    ABSTAIN_TEXT,
    format_fragments,
    parse_grounded_response,
)
from hipporag.utils.custom_types import Citation


def test_valid_answer_extracts_citation():
    raw = "ОТВЕТ: Прибор работает от батареи AA [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.text == "Прибор работает от батареи AA [1]."
    assert result.used_citation_ids == [1]
    assert result.abstained is False
    assert result.errors == []
    assert result.is_grounded is True


def test_refusal_is_recognized():
    raw = f"ОТВЕТ: {ABSTAIN_TEXT}\nИСТОЧНИКИ:\nОТКАЗ: да"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.text == ABSTAIN_TEXT
    assert result.used_citation_ids == []
    assert result.abstained is True
    assert result.errors == []
    assert result.is_grounded is True


def test_empty_raw_abstains():
    result = parse_grounded_response("", fragment_count=2)

    assert result.text == ABSTAIN_TEXT
    assert result.abstained is True


def test_sources_without_inline_markers_abstains():
    raw = "ОТВЕТ: Прибор работает от батареи AA.\nИСТОЧНИКИ: 1\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.abstained is True
    assert result.text == ABSTAIN_TEXT
    assert "no_inline_citations" in result.errors


def test_answer_without_any_citations_abstains():
    raw = "ОТВЕТ: Прибор работает от батареи AA.\nИСТОЧНИКИ:\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.abstained is True
    assert "no_inline_citations" in result.errors


def test_out_of_range_citation_abstains():
    raw = "ОТВЕТ: Прибор работает от батареи AA [7].\nИСТОЧНИКИ: 7\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.abstained is True
    assert result.text == ABSTAIN_TEXT
    assert "invalid_citations:[7]" in result.errors
    assert "no_valid_citations" in result.errors


def test_multiple_citation_forms():
    raw = "ОТВЕТ: Прибор работает от батареи AA [1][2].\nИСТОЧНИКИ: 1,2\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.used_citation_ids == [1, 2]
    assert result.errors == []


def test_comma_separated_citation_group():
    raw = "ОТВЕТ: Прибор работает от батареи AA [1, 2].\nИСТОЧНИКИ: 1,2\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.used_citation_ids == [1, 2]


def test_source_mismatch_is_reported():
    raw = "ОТВЕТ: Прибор работает от батареи AA [1].\nИСТОЧНИКИ: 2\nОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.used_citation_ids == [1]
    assert any(error.startswith("source_mismatch") for error in result.errors)
    assert result.is_grounded is False


def test_uncited_sentence_is_reported():
    raw = (
        "ОТВЕТ: Прибор работает от батареи AA [1]. "
        "Корпус прибора «Альфа» изготовлен из прочного пластика.\n"
        "ИСТОЧНИКИ: 1\nОТКАЗ: нет"
    )
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.used_citation_ids == [1]
    assert any(error.startswith("uncited_sentences") for error in result.errors)


def test_code_fence_is_stripped():
    raw = (
        "```text\n"
        "ОТВЕТ: Прибор работает от батареи AA [1].\n"
        "ИСТОЧНИКИ: 1\n"
        "ОТКАЗ: нет\n"
        "```"
    )
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.text == "Прибор работает от батареи AA [1]."
    assert result.used_citation_ids == [1]


def test_legacy_answer_format_non_strict():
    raw = "Answer: 42"
    result = parse_grounded_response(raw, fragment_count=2, strict=False)

    assert result.text == "42"
    assert result.abstained is False
    assert result.used_citation_ids == []


def test_legacy_answer_format_strict_abstains():
    raw = "Answer: 42"
    result = parse_grounded_response(raw, fragment_count=2, strict=True)

    assert result.abstained is True
    assert "no_inline_citations" in result.errors


def test_source_label_does_not_count_as_citation():
    raw = "ОТВЕТ: Прибор работает от батареи AA [Источник 1].\n" "ИСТОЧНИКИ: 1\n" "ОТКАЗ: нет"
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.abstained is True
    assert "no_inline_citations" in result.errors


def test_format_fragments_numbers_and_labels_sources():
    citations = [
        Citation(source_file="a.pdf", doc_title="Книга A", page_num=1),
        Citation(source_file="b.pdf", doc_title="Книга B", page_num=2, section="Раздел"),
    ]
    text = format_fragments(["первый фрагмент", "второй фрагмент"], citations)

    assert "[1] Источник: Книга A, стр. 1" in text
    assert "[2] Источник: Книга B, стр. 2, раздел: Раздел" in text
    assert "первый фрагмент" in text
    assert "второй фрагмент" in text


@pytest.mark.parametrize("raw", ["ОТВЕТ:", "ОТВЕТ: \nИСТОЧНИКИ: 1\nОТКАЗ: нет"])
def test_empty_answer_abstains(raw):
    result = parse_grounded_response(raw, fragment_count=2)

    assert result.text == ABSTAIN_TEXT
    assert result.abstained is True
