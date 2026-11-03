"""Контракт бюджета латентности (H1-3).

Политика (план, H1-3):

* режим по умолчанию — p95 ≤ 10 с;
* LLM-режимы — p95 ≤ 30 с;
* всё, что медленнее 30 с, — «экспериментальный» режим (выключен по умолчанию).

Тест работает без индекса, моделей и сети: считает вердикты по числам.
"""

import json

from astro_app import performance as perf
from astro_app.menu import EXPERIMENTAL_MARK, format_modes_menu
from astro_app.modes import DEFAULT_MODE


def _summary(mode_id, label, p50, p95, llm_calls=0.0):
    return {
        "mode": {"id": mode_id, "label": label},
        "n_questions": 40,
        "latency_ms": {"p50": p50, "p95": p95, "mean": p50},
        "retrieval_ms": {"p50": p50, "p95": p95},
        "llm_calls_mean": llm_calls,
    }


def test_default_budget_is_ten_seconds():
    assert perf.DEFAULT_P95_BUDGET_MS == 10_000
    assert perf.LLM_P95_BUDGET_MS == 30_000
    assert perf.EXPERIMENTAL_P95_MS == 30_000


def test_roles_follow_default_and_llm_modes():
    assert perf.mode_role(DEFAULT_MODE) == perf.ROLE_DEFAULT
    assert perf.mode_role("1") == perf.ROLE_LLM  # ходит в LLM
    assert perf.mode_role("5") == perf.ROLE_EXPERT  # без LLM и не default


def test_budget_for_role():
    assert perf.budget_for_role(perf.ROLE_DEFAULT) == 10_000
    assert perf.budget_for_role(perf.ROLE_LLM) == 30_000
    assert perf.budget_for_role(perf.ROLE_EXPERT) == 30_000


def test_classify_latency():
    assert perf.classify_latency(5_000, perf.ROLE_DEFAULT) == perf.STATUS_OK
    assert perf.classify_latency(10_000, perf.ROLE_DEFAULT) == perf.STATUS_OK
    assert perf.classify_latency(10_001, perf.ROLE_DEFAULT) == perf.STATUS_OVER_BUDGET
    assert perf.classify_latency(12_800, perf.ROLE_LLM) == perf.STATUS_OK
    assert perf.classify_latency(31_000, perf.ROLE_LLM) == perf.STATUS_EXPERIMENTAL
    assert perf.classify_latency(None, perf.ROLE_DEFAULT) == perf.STATUS_UNKNOWN


def test_evaluate_mode_marks_default_violation():
    """Режим 1 (p95 12,8 с) нарушает бюджет режима по умолчанию."""
    verdict = perf.evaluate_mode(_summary("1", "Multi-vector", 6_200, 12_800))
    assert verdict["role"] == perf.ROLE_LLM
    assert verdict["status"] == perf.STATUS_OK  # для LLM-режима 30 с — уложился

    verdict = perf.evaluate_mode(_summary("2", "Граф", 545, 11_000))
    assert verdict["role"] == perf.ROLE_DEFAULT
    assert verdict["status"] == perf.STATUS_OVER_BUDGET
    assert verdict["margin_ms"] == -1_000


def test_evaluate_budgets_splits_violations_and_experimental():
    summaries = [
        _summary("2", "Граф", 545, 5_000),  # default, ок
        _summary("1", "Multi-vector", 6_200, 12_800),  # llm, ок
        _summary("5", "BM25+CE-ru", 2_000, 12_000),  # expert, ок (30 с)
        _summary("12", "Self-RAG", 23_800, 37_900),  # > 30 с → экспериментальный
    ]
    result = perf.evaluate_budgets(summaries)
    assert result["ok"] is True
    assert result["violations"] == []
    assert result["experimental"] == ["12"]

    summaries[0] = _summary("2", "Граф", 545, 12_000)
    result = perf.evaluate_budgets(summaries)
    assert result["ok"] is False
    assert result["violations"] == ["2"]


def test_unknown_p95_is_not_a_violation():
    result = perf.evaluate_budgets([_summary("2", "Граф", 545, None)])
    assert result["ok"] is True
    assert result["unknown"] == ["2"]


def test_format_budget_table_has_all_modes():
    verdicts = perf.evaluate_budgets(
        [_summary("2", "Граф", 545, 5_000), _summary("12", "Self-RAG", 23_800, 37_900)]
    )["modes"]
    table = perf.format_budget_table(verdicts)
    assert "Граф" in table
    assert "Self-RAG" in table
    assert "experimental" in table
    # заголовок + разделитель + по строке на режим
    assert table.count("\n") == 1 + len(verdicts)


def test_repo_baseline_snapshot_is_consistent():
    """Снимок замеров в репозитории согласован с политикой."""
    baseline = perf.load_baseline()
    assert baseline, "eval/dataset/latency_baseline.json должен существовать"
    assert baseline["budgets_ms"]["default"] == perf.DEFAULT_P95_BUDGET_MS

    verdicts = perf.baseline_verdicts(baseline)
    by_mode = {v["mode"]: v for v in verdicts}
    assert by_mode["12"]["status"] == perf.STATUS_EXPERIMENTAL
    assert by_mode["19"]["status"] == perf.STATUS_EXPERIMENTAL
    assert by_mode["21"]["status"] == perf.STATUS_OK
    # p95 режима по умолчанию не измерен — это «unknown», а не «ok».
    assert by_mode["2"]["status"] == perf.STATUS_UNKNOWN


def test_menu_marks_experimental_modes():
    """Режимы > 30 с помечаются в меню как экспериментальные."""
    text = format_modes_menu()
    assert EXPERIMENTAL_MARK.strip() in text
    experimental_lines = [line for line in text.splitlines() if EXPERIMENTAL_MARK in line]
    numbers = {line.strip().split(" — ")[0] for line in experimental_lines}
    assert numbers == {"12", "19"}


def test_load_baseline_tolerates_missing_file():
    assert perf.load_baseline(perf.BASELINE_PATH.parent / "нет-такого-файла.json") == {}
    assert perf.experimental_modes({}) == set()


def test_baseline_json_is_valid_json():
    """Файл снимка — валидный JSON с ожидаемыми ключами."""
    raw = perf.BASELINE_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert set(data) >= {"modes", "budgets_ms", "provenance"}
    assert "1" in data["modes"]
