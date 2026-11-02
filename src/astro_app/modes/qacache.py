"""Режим 23: QA Cache — перенос из main.py (шаг 6).

Импортирует режим 22 (.sign_zones) и подсистему кэша (.qa_cache) — без цикла:
sign_zones тянет только .qa_cache, а этот модуль — оба.
"""

from ..qa_cache import _qa_derive_short, _qa_load, _qa_lookup, _qa_store
from .sign_zones import search_semantic_filter


def search_qa_cache(hipporag, query, top_k=10):
    """
    Метод 23: QA Cache — поиск по базе готовых ответов.
    Если вопрос уже задавался — возвращает кэшированный ответ.
    Если нет — запускает метод 22, сохраняет результат, возвращает.
    """
    import time as _time

    t0 = _time.time()
    print("  Метод 23: QA Cache", flush=True)
    cache = _qa_load()
    cached, score = _qa_lookup(query, cache)
    if cached:
        print(f"  Найден в кэше (совпадение: {score:.0%})", flush=True)
        short = cached.get("short_answer", "")
        full = cached.get("answer", "")
        # Записи, созданные до P2-8, могли остаться с пустым short_answer —
        # восстанавливаем короткий ответ из полного (P2-8).
        if not short:
            short = _qa_derive_short(full)
        result = (
            f"=== ОТВЕТ ИЗ КЭША ===\n"
            f"Запрос: {query}\n"
            f"Совпадение: {score:.0%}\n\n"
            f"ОТВЕТ: {short or 'не указан'}\n\n"
            f"--- Полный ответ ---\n"
            f"{full}\n\n"
            f"Сохранён: {cached['timestamp']} | Метод: {cached['method']} | Модель: {cached['model']}"
        )
        print(f"  Готово за {_time.time() - t0:.2f}с", flush=True)
        return [(result, 999.0, 0, 0, 0)]
    # Не найдено → запускаем метод 22, сохраняем
    print("  Не найден в кэше, запускаю метод 22...", flush=True)
    res22 = search_semantic_filter(hipporag, query, top_k=50)
    if res22 and len(res22) > 0:
        txt = res22[0][0] if isinstance(res22[0], tuple) else str(res22[0])
        # Честный отказ не кэшируем: после переиндексации/правок он может
        # перестать быть верным, а кэш обязан быть версионным (P1-9).
        if "Ответ не найден" in txt:
            print(f"  Готово за {_time.time() - t0:.2f}с (отказ не кэшируется)", flush=True)
            return res22
        # Извлекаем короткий ответ
        short = ""
        for line in txt.split("\n"):
            if "ОТВЕТ:" in line:
                short = line.split("ОТВЕТ:")[1].strip()
                break
        sources = []
        for line in txt.split("\n"):
            if line.startswith("[Источник"):
                sources.append(line.strip())
        _qa_store(query, txt, short, "QACache+Pattern", 23, sources, 999.0)
        print(f"  Готово за {_time.time() - t0:.2f}с", flush=True)
        return res22
    return [("Не удалось получить ответ.", 0, 0, 0, 0)]
