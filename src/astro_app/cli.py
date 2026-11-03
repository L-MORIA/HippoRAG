"""CLI-приложение: меню режимов, диспетчер, QA-действия (шаг 7 миграции).

main() воспроизводит поведение main.py (модель-чейсер, меню, автосохранение в
QA-кэш, честный отказ для специализированных режимов), но диспетчер строится
из реестра MODES вместо if/elif.
"""

from __future__ import annotations

import sys

from . import config
from .display import show_results
from .modes import MODES
from .qa_cache import _qa_export_training, _qa_load, _qa_lookup, _qa_store

MODEL_OPTIONS = {
    "1": ("google/gemma-4-e4b", "Gemma-4 ~9B активных (текущая)"),
    "2": ("google/gemma-4-26b-a4b", "Gemma-4 26B MoE ~4B активных, больше знаний"),
    "3": (
        "qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled@iq4_xs",
        "Qwen3.6 35B MoE Claude-distilled, быстрый",
    ),
    "4": ("nvidia/nemotron-3-nano-4b", "Nvidia Nemotron 4B, лёгкая и быстрая"),
}


def _qa_run_test(hipporag, query):
    """Запустить все методы и сравнить с кэшем."""
    from .modes import self_rag, sentence_dat, sign_zones, small_to_big

    test_methods = [
        ("6", "Small→Big-ru", lambda: small_to_big.search_small_to_big_ru(hipporag, query)),
        ("12", "Self-RAG", lambda: self_rag.search_self_rag(hipporag, query)),
        ("20", "Sentence", lambda: sentence_dat.search_sentence_level(hipporag, query)),
        ("21", "DAT+LLM", lambda: sentence_dat.search_dat_sentence_llm(hipporag, query)),
        ("22", "Pattern", lambda: sign_zones.search_semantic_filter(hipporag, query)),
    ]
    cache = _qa_load()
    cached, _ = _qa_lookup(query, cache)
    print(f"\n{'=' * 60}")
    print("ТЕСТИРОВАНИЕ ВСЕХ МЕТОДОВ")
    print(f"Запрос: {query}")
    if cached:
        print(f"Ожидаемый ответ: {cached.get('short_answer', '')[:100]}")
    print(f"{'=' * 60}\n")
    print(f"{'Метод':<15} {'Ответ (первые 80)':<80} {'Совпадение':<10}")
    print("-" * 105)
    for _mid, mname, mfunc in test_methods:
        try:
            res = mfunc()
            txt = ""
            if res:
                r = res[0]
                txt = r[0] if isinstance(r, tuple) and len(r) > 0 else str(r)
            short = txt[:80].replace("\n", " ").strip()
            match = "✓" if cached and cached.get("short_answer", "") in txt else "?"
            print(f"{mname:<15} {short:<80} {match:<10}")
        except Exception as e:
            print(f"{mname:<15} {'ERROR: ' + str(e)[:60]:<80} {'':<10}")


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("hipo-rag — интерактивный поиск по астрологическим PDF.")
        print("Запуск без аргументов: hipo-rag")
        print("Модель и режим выбираются в меню; настройки — в .env.")
        return

    print("=" * 80)
    print("HIPPORAG - Поиск по базе знаний (астрофэншуй)")
    print("=" * 80)

    print("\nВыберите модель для ответов:")
    for k, v in MODEL_OPTIONS.items():
        cur = " ← текущая" if v[0] == config.LLM_NAME else ""
        print(f"  {k} — {v[1]}{cur}")
    model_choice = input("Выбор [1-4], Enter для текущей: ").strip()
    chosen_model = MODEL_OPTIONS.get(model_choice, (None, None))[0] or config.LLM_NAME
    if chosen_model != config.LLM_NAME:
        print(f"  → Переключаю API на: {chosen_model}")
    else:
        print(f"  → Модель: {chosen_model}")
    # Динамически: qa_cache штампует записи текущей моделью (как main.LLM_NAME раньше)
    config.LLM_NAME = chosen_model

    print("\nЗагрузка индекса...")
    hipporag = config.load_hipporag(llm_name=chosen_model)
    print("Готово!\n")

    while True:
        print("-" * 80)
        print("Режимы поиска:")
        for num, spec in MODES.items():
            print(f"  {num} — {spec.menu}{' ⭐' if spec.star else ''}")
            if spec.special:
                print(f"       {spec.special}")
        print("  t — Тестирование всех методов по кэшу")
        print("  e — Экспорт кэша в JSONL (для обучения)")
        print("  q — Выход")
        mode = input("Выбор [1-23/q/t/e]: ").strip().lower()
        if mode == "q":
            break
        if mode == "t":
            q = input("Запрос для теста (Enter = из кэша): ").strip()
            if not q:
                cache = _qa_load()
                if cache:
                    q = cache[-1]["query"]
                    print(f"  Беру последний из кэша: {q}")
                else:
                    print("  Кэш пуст.")
                    continue
            _qa_run_test(hipporag, q)
            continue
        if mode == "e":
            _qa_export_training()
            continue
        if mode not in MODES:
            continue

        spec = MODES[mode]
        query = input("Запрос: ").strip()
        if not query:
            continue

        reformulated = None
        hyde_text = None
        result = spec.fn(hipporag, query)
        if spec.extra:
            results, extra = result
            if mode == "1":
                reformulated = extra
            else:
                hyde_text = extra
        else:
            results = result

        # Автосохранение в QA Cache
        if results and mode != "23":
            try:
                r0 = results[0]
                txt = r0[0] if isinstance(r0, tuple) and len(r0) > 0 else str(r0)
                # Честный отказ не кэшируем: после правок/переиндексации он
                # может перестать быть верным (P1-9).
                if "Ответ не найден" in txt:
                    raise ValueError("отказ не кэшируем")
                short = ""
                for line in txt.split("\n"):
                    if "ОТВЕТ:" in line:
                        short = line.split("ОТВЕТ:")[1].strip()
                        break
                sources = [ln.strip() for ln in txt.split("\n") if "Источник" in ln[:20]]
                _qa_store(query, txt, short, spec.label, int(mode), sources[:5])
            except Exception:
                pass

        show_results(query, results, spec.label, reformulated if reformulated else hyde_text)

    print("\nДо свидания!")
