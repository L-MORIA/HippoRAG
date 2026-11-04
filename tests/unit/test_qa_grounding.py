"""Тесты контракта заземления Q&A (H2-6).

Проверяем чистый парсер без LLM и индекса: ссылки, отказ, code fences,
старый формат и защиту от «ответа без доказательства». Интеграционная часть
ниже собирает ``HippoRAG`` без ``__init__`` (без моделей и индекса), чтобы
проверить, что ``qa()`` не зовёт LLM без контекста и сохраняет только
использованные цитаты.
"""

import pytest

from hipporag.qa.grounding import (
    ABSTAIN_TEXT,
    format_fragments,
    parse_grounded_response,
)
from hipporag.utils.custom_types import Citation, QuerySolution


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


def test_query_solution_has_grounding_defaults():
    solution = QuerySolution(question="Что такое?")

    assert solution.used_citations == []
    assert solution.abstained is False
    assert solution.grounding_errors == []
    assert solution.to_dict()["abstained"] is False
    assert solution.to_dict()["grounding_errors"] == []


def test_config_has_strict_grounding_flag():
    from hipporag.utils.config_utils import BaseConfig

    assert BaseConfig().qa_strict_grounding is True


class _StubLLM:
    """LLM-заглушка: отдаёт заранее заданные ответы и считает вызовы."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def infer(self, messages, **kwargs):  # noqa: ARG002
        self.messages.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response, {"prompt_tokens": 1}, False


def _make_hipporag_stub():
    """HippoRAG без __init__: только то, что нужно ``qa()``."""
    from hipporag.HippoRAG import HippoRAG
    from hipporag.prompts.prompt_template_manager import PromptTemplateManager
    from hipporag.utils.config_utils import BaseConfig

    rag = HippoRAG.__new__(HippoRAG)
    rag.global_config = BaseConfig()
    rag.global_config.dataset = "astrology"
    rag.global_config.qa_top_k = 5
    rag.global_config.qa_strict_grounding = True
    rag.prompt_template_manager = PromptTemplateManager()
    return rag


def test_hipporag_qa_skips_llm_on_empty_context():
    rag = _make_hipporag_stub()
    rag.llm_model = _StubLLM([])

    solutions = [QuerySolution(question="Что такое?", docs=[])]
    out_solutions, messages, metadata = rag.qa(solutions)

    assert rag.llm_model.calls == 0
    assert out_solutions[0].answer == ABSTAIN_TEXT
    assert out_solutions[0].abstained is True
    assert out_solutions[0].used_citations == []
    assert len(messages) == len(metadata) == 1


def test_hipporag_qa_keeps_only_used_citations():
    rag = _make_hipporag_stub()
    first = Citation(source_file="a.pdf", doc_title="Книга A", page_num=1)
    second = Citation(source_file="b.pdf", doc_title="Книга B", page_num=2)
    rag.llm_model = _StubLLM(
        ["ОТВЕТ: Прибор работает от батареи AA [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет"]
    )

    solution = QuerySolution(
        question="От чего работает прибор?",
        docs=["фрагмент один", "фрагмент два"],
        citations=[first, second],
    )
    out_solutions, messages, _ = rag.qa([solution])

    assert rag.llm_model.calls == 1
    assert out_solutions[0].used_citations == [first]
    assert out_solutions[0].abstained is False
    assert out_solutions[0].grounding_errors == []
    assert out_solutions[0].answer == "Прибор работает от батареи AA [1]."
    assert "[1] Источник: Книга A, стр. 1" in rag.llm_model.messages[0][-1]["content"]


def test_hipporag_qa_abstains_when_model_ignores_citations():
    rag = _make_hipporag_stub()
    rag.llm_model = _StubLLM(["ОТВЕТ: Прибор работает от батареи AA.\nИСТОЧНИКИ:\nОТКАЗ: нет"])

    solution = QuerySolution(
        question="От чего работает прибор?",
        docs=["Прибор «Альфа» работает от батареи AA."],
        citations=[Citation(source_file="a.pdf", doc_title="Книга A", page_num=1)],
    )
    out_solutions, _, _ = rag.qa([solution])

    assert out_solutions[0].answer == ABSTAIN_TEXT
    assert out_solutions[0].abstained is True
    assert "no_inline_citations" in out_solutions[0].grounding_errors
    assert out_solutions[0].used_citations == []


def test_hipporag_qa_mixed_batch_keeps_alignment():
    rag = _make_hipporag_stub()
    rag.llm_model = _StubLLM(
        ["ОТВЕТ: Прибор работает от батареи AA [1].\nИСТОЧНИКИ: 1\nОТКАЗ: нет"]
    )

    empty = QuerySolution(question="Первый?", docs=[])
    answerable = QuerySolution(
        question="Второй?",
        docs=["Прибор «Альфа» работает от батареи AA."],
        citations=[Citation(source_file="a.pdf", doc_title="Книга A", page_num=1)],
    )
    out_solutions, messages, metadata = rag.qa([empty, answerable])

    assert rag.llm_model.calls == 1
    assert len(messages) == len(metadata) == 2
    assert out_solutions[0].answer == ABSTAIN_TEXT
    assert out_solutions[1].abstained is False
    assert out_solutions[1].used_citations == [answerable.citations[0]]


def test_astrology_template_renders_grounded_contract():
    from hipporag.prompts.prompt_template_manager import PromptTemplateManager

    manager = PromptTemplateManager()
    rendered = manager.render(name="rag_qa_astrology", prompt_user="Вопрос: тест")

    assert isinstance(rendered, list)
    system = rendered[0]["content"]
    assert ABSTAIN_TEXT in system
    assert "[1]" in system
    assert "гуа 8" not in system

    # Положительный пример показывает ссылку, отрицательный — отказ.
    assert "[1]" in rendered[2]["content"]
    assert "ОТКАЗ: да" in rendered[4]["content"]
    assert rendered[-1] == {"role": "user", "content": "Вопрос: тест"}
