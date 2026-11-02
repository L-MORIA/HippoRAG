"""Тесты display.show_results на реальных формах данных снапшота (шаг 7/§7 плана).

Формы взяты из outputs/refactor_baseline/pre/snapshot.json:
- Self-RAG (режим 12): 6-кортежи [str, float, str, int, int, int] — до фикса
  show_results падал с ValueError (P: режим 12).
- Small→Big: та же форма 6-кортежей.
- Обычные режимы: 5-кортежи [str, float, float, float, float].
"""

from astro_app.display import show_results


def run(query, results, mode_name, extra=None):
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        show_results(query, results, mode_name, extra)
    return buf.getvalue()


def test_self_rag_six_tuple_renders_with_source_and_pages():
    """Режим 12: 6-кортеж Self-RAG отображается со источником и страницами.

    До фикса: ValueError (too many values to unpack, expected 5, got 6).
    """
    results = [
        ["Текст про багуа и секторы квартиры.", 0.85, "Психологический фэншуй.pdf", 3, 7, 0.5],
        ["Ещё фрагмент про знак Овна.", 0.72, "Астрофэншуй квартиры.pdf", 11, 14, 0.3],
    ]
    out = run("багуа и секторы", results, "Self-RAG")
    assert "Self-RAG" in out
    assert "Психологический фэншуй.pdf" in out
    assert "стр.3-7" in out
    assert "стр.11-14" in out
    assert "багуа и секторы квартиры" in out.lower()


def test_small_to_big_six_tuple_renders_unchanged():
    """Small→Big (та же форма) продолжает отображаться как раньше."""
    results = [
        ["Документ про спальню", 0.87, "Психологический фэншуй.pdf", 3, 7, 0.5],
    ]
    out = run("спальня для тельца", results, "Small→Big-en")
    assert "стр.3-7" in out
    assert "Документ про спальню" in out


def test_regular_five_tuple_renders_unchanged():
    """Обычные режимы (5-кортежи) — вывод не изменился."""
    results = [["Документ про прихожую Овна", 0.95, 0.8, 0.7, 0.1]]
    out = run("прихожая для овна", results, "Multi-vector")
    assert "dpr=0.800" in out
    assert "multi=0.700" in out
    assert "Документ про прихожую Овна" in out


def test_full_material_branch_renders_file_marker():
    """Ветка «Весь знак» с маркером файла — без регрессии.

    Реальная форма режима 9 из снапшота: 6-кортеж (unpack в show_results
    происходит ДО ветки is_full_material).
    """
    results = [["Файл: outputs/reports/овен.docx\nПолный текст знака овен", 0, "", 0, 0, 0]]
    out = run("весь знак овен", results, "Весь знак")
    assert ">>> Файл: outputs/reports/овен.docx <<<" in out


def test_empty_results_renders_placeholder():
    out = run("любой запрос", [], "Multi-vector")
    assert "Ничего не найдено" in out
