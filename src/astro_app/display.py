"""Отображение результатов поиска в консоли (перенос из main.py, шаг 2)."""

from .textutils import fix_chunk_start, show_text_safe


def show_results(query, results, mode_name, reformulated=None):
    """Display search results with sentence-boundary detection."""
    q_lower = query.lower()
    print(f"\n{'=' * 80}")
    print(f"[{mode_name}] Результаты: '{query}'")
    if reformulated and reformulated != query:
        label = "гипотетический документ" if mode_name == "HyDE" else "переформулировка"
        print(f"  ({label}: '{reformulated[:120]}')")
    print(f"{'=' * 80}")
    if not results:
        print("Ничего не найдено.")
        return

    is_full_material = mode_name in ("Весь знак", "Резюме", "Self-RAG")

    for i, r in enumerate(results, 1):
        is_selfrag = mode_name == "Self-RAG"
        is_small_to_big = len(r) == 6 and isinstance(r[2], str) and not is_selfrag
        if is_small_to_big:
            doc, score, source, min_p, max_p, boost = r
            extra = f" ({str(source)[:40]} стр.{min_p}-{max_p})"
        else:
            doc, score, dpr, multi, boost = r
            extra = f" (dpr={dpr:.3f}, multi={multi:.3f})"

        has_term = q_lower in doc.lower()
        marker = " ✅" if has_term else ""

        if is_full_material:
            # Для полного материала: показываем резюме + путь к файлу
            display = doc
            # Если есть строка "Файл: " — выделяем её
            if "Файл:" in doc:
                for ln in doc.split("\n"):
                    if ln.startswith("Файл:") or "outputs/reports" in ln:
                        display = display.replace(ln, f"\n>>> {ln} <<<")
        else:
            clean_doc, was_truncated = fix_chunk_start(doc)
            display = show_text_safe(clean_doc)
            if was_truncated:
                display = "[...обрыв] " + display

        print(f"\n--- {i}. (score: {score:.4f}, kw: +{boost:.2f}){extra}{marker} ---")
        print(display)
        print()
