"""Контракт H1-2: рекомендация режимов зафиксирована в коде.

После H1-2 должно быть правдой:

* `DEFAULT_MODE` существует в реестре и это измеренный режим «2» (Граф) —
  единственный измеренный режим, укладывающийся в бюджет p95 ≤ 10 с;
* ровно два рекомендованных режима — «1» и «2», и только у них `star`;
* у каждого из 23 режимов заполнено поле `when` («когда уместен»);
* рекомендуемые и экспертные режимы разбивают реестр без потерь и пересечений;
* режим по умолчанию не зависит от LLM и не является answer-режимом — базовый
  путь должен работать без LM Studio;
* меню печатает обе секции, пометку «(по умолчанию)» и строки «когда: …».
"""

from astro_app.evaluation import ANSWER_ONLY_MODES, LLM_DEPENDENT_MODES
from astro_app.menu import format_modes_menu
from astro_app.modes import DEFAULT_MODE, MODES, RECOMMENDED_MODES, expert_modes, recommended_modes


def test_default_mode_is_measured_graph_mode():
    """Default зафиксирован как режим 2 «Граф» (решение по данным H1-1)."""
    assert DEFAULT_MODE == "2"
    assert DEFAULT_MODE in MODES
    assert MODES[DEFAULT_MODE].label == "Граф"


def test_recommended_modes_are_one_and_two():
    """Ровно два рекомендованных режима: 1 (качество) и 2 (скорость)."""
    assert RECOMMENDED_MODES == ("1", "2")
    assert set(RECOMMENDED_MODES) <= set(MODES)


def test_only_recommended_modes_are_starred():
    """Звёздочка ⭐ только у рекомендованных режимов."""
    starred = {num for num, spec in MODES.items() if spec.star}
    assert starred == set(RECOMMENDED_MODES)


def test_every_mode_documents_when_to_use():
    """У каждого режима заполнено «когда уместен»."""
    missing = [num for num, spec in MODES.items() if not spec.when.strip()]
    assert missing == []
    assert len(MODES) == 23


def test_recommended_and_expert_partition_modes():
    """Секции меню разбивают реестр без потерь и пересечений."""
    rec = recommended_modes()
    exp = expert_modes()
    assert set(rec) & set(exp) == set()
    assert set(rec) | set(exp) == set(MODES)
    assert len(rec) == len(RECOMMENDED_MODES)


def test_default_mode_needs_no_llm():
    """Default не зависит от LLM: базовый путь работает без LM Studio."""
    assert DEFAULT_MODE not in LLM_DEPENDENT_MODES


def test_default_mode_is_not_answer_only():
    """Default возвращает выдачу фрагментов, а не сгенерированный ответ."""
    assert DEFAULT_MODE not in ANSWER_ONLY_MODES


def test_menu_has_both_sections():
    """Меню разделено на рекомендуемые и экспертные режимы."""
    text = format_modes_menu()
    assert "Рекомендуемые:" in text
    assert "Экспертные:" in text


def test_menu_marks_default_mode():
    """Режим по умолчанию помечен в меню."""
    text = format_modes_menu()
    default_line = next(
        line for line in text.splitlines() if line.strip().startswith(f"{DEFAULT_MODE} —")
    )
    assert "(по умолчанию)" in default_line


def test_menu_lists_all_modes_with_when_hint():
    """Все 23 режима в меню, у каждого есть строка «когда: …»."""
    text = format_modes_menu()
    for num in MODES:
        assert f"{num} — " in text
    assert text.count("когда: ") == len(MODES)
