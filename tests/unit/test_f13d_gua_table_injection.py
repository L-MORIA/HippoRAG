"""F-13d: инъекция строки таблицы «год → число гуа» (таблица в PDF картинкой)."""

from astro_app.web.service import _gua_table_fragment


def test_date_query_gets_covering_row():
    fragment = _gua_table_fragment("Какое у меня число гуа если я родилась 15.07.1972 женщина")
    assert fragment is not None
    assert "1972-02-15" in fragment.text and "1973-02-02" in fragment.text
    assert "женщин — 5" in fragment.text
    assert fragment.source_file == "book_table"


def test_year_only_query_gets_mid_year_row():
    fragment = _gua_table_fragment("число гуа 1980")
    assert fragment is not None
    assert "1980-02-16" in fragment.text


def test_non_gua_query_gets_nothing():
    assert _gua_table_fragment("зоны овна в квартире") is None


def test_query_without_year_gets_nothing():
    assert _gua_table_fragment("что такое шен-ци") is None
