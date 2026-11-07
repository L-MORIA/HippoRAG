"""Числовой мост «гуа 5 → пятёрка» (уточнение автора книг, 2026-11-07).

В книге направления пятого числа сформулированы словом «пятёрка» без
«гуа 5», поэтому словарные синонимы их не находили. Мост добавляет слово
«пятёрка» в расширение запроса только при явном «гуа» и цифре 5.
"""

from astro_app.query import _expand_query, split_query_terms


def _expand(question: str) -> set:
    terms, q_lower = split_query_terms(question)
    return set(_expand_query(q_lower, terms))


def test_bridge_fires_for_gua_5():
    expanded = _expand("Какое направление Шен-ци у числа гуа 5?")
    assert {"пятёрка", "пятёрки", "пятёрок"} <= expanded


def test_bridge_does_not_fire_for_other_gua_numbers():
    for question in (
        "Какое направление Шен-ци у числа гуа 1?",
        "Что такое Шен-ци у числа гуа 8?",
    ):
        assert not ({"пятёрка", "пятёрки", "пятёрок"} & _expand(question))


def test_bridge_does_not_fire_without_gua():
    assert not ({"пятёрка", "пятёрки", "пятёрок"} & _expand("Зона числа 5 в доме?"))
    assert not ({"пятёрка", "пятёрки", "пятёрок"} & _expand("Какие зоны у Овна?"))
