"""QA Cache: база вопросов-ответов (кэш + экспорт для обучения).

Перенос из main.py (шаг 3 миграции MAIN_SPLIT_PLAN.md). Функции, зависящие от
режимов поиска (_qa_run_test, search_qa_cache), переезжают на шаге 6 — вместе
с режимом 22, который они вызывают.

QA_CACHE_PATH и QA_CACHE_VERSION — атрибуты модуля: тесты подменяют путь
монкипатчем, чтобы писать во временную копию, а не в пользовательский кэш.
"""

import json
import re
import time
from pathlib import Path

from . import config

QA_CACHE_PATH = Path("outputs/qa_cache/qa_cache.json")

# Версия формата записи и логики ответа (P1-9). Кэш переживал правки кода:
# записи, сохранённые до фиксов, отдавались как «кэшированный ответ», то есть
# неверный ответ под видом кэша. Увеличивайте версию при изменении формата
# ответа/промптов — старые записи перестанут читаться и будут пересчитаны.
QA_CACHE_VERSION = 2


def _qa_is_current(entry):
    """Запись актуальна: совпадает версия формата и модель."""
    if not isinstance(entry, dict):
        return False
    if entry.get("schema_version") != QA_CACHE_VERSION:
        return False
    model = entry.get("model")
    if model and model != config.LLM_NAME:
        return False
    return True


def _qa_load():
    if QA_CACHE_PATH.exists():
        try:
            with open(str(QA_CACHE_PATH), "r", encoding="utf-8") as f:
                cache = json.load(f)
            if isinstance(cache, list):
                return [entry for entry in cache if _qa_is_current(entry)]
        except Exception:
            pass
    return []


def _qa_save(cache):
    QA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(QA_CACHE_PATH), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _qa_norm(q):
    return re.sub(r"[^\w\s]", "", q.lower()).strip()


def _qa_lookup(query, cache):
    """Поиск похожего вопроса в кэше. Возвращает (entry, score) или None."""
    qn = _qa_norm(query)
    q_words = set(qn.split())
    best = None
    best_score = 0
    for e in cache:
        en = e.get("query_norm", "")
        e_words = set(en.split())
        if not q_words or not e_words:
            continue
        overlap = len(q_words & e_words) / max(len(q_words), len(e_words))
        if overlap > best_score:
            best_score = overlap
            best = e
    if best_score >= 0.7:
        return best, best_score
    return None, 0


def _qa_derive_short(text: str) -> str:
    """Короткий ответ из текста результата, если нет маркера 'ОТВЕТ:'.

    Раньше для режимов без маркера (например 22 — SemFilter) в кэш попадал
    пустой short_answer, и режим 23 отдавал «ОТВЕТ: не указан» (P2-8).
    """
    if not text:
        return ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("===") and line.endswith("==="):
            continue
        if line.startswith("["):  # строки вида "[Источник 1 — file.pdf стр.1-2]"
            continue
        return line[:200]
    return ""


def _qa_store(query, answer, short_answer, method_name, method_id, sources, score=999.0):
    """Сохранить результат в кэш."""
    cache = _qa_load()
    qn = _qa_norm(query)
    # Не дублировать точные совпадения
    for e in cache:
        if e.get("query_norm") == qn and e.get("method_id") == method_id:
            return
    if not short_answer:
        short_answer = _qa_derive_short(answer)
    entry = {
        "id": f"qa_{len(cache) + 1}",
        "schema_version": QA_CACHE_VERSION,
        "query": query,
        "query_norm": qn,
        "answer": answer if answer else "",
        "short_answer": short_answer[:200] if short_answer else "",
        "method": method_name,
        "method_id": method_id,
        "model": config.LLM_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": sources[:10] if sources else [],
        "score": score,
    }
    cache.append(entry)
    # _qa_load уже отбросил записи старой версии/модели, поэтому сохранение
    # заодно вычищает устаревшие записи с диска (P1-9).
    _qa_save(cache)
    print(f"  Сохранено в QA Cache (#{len(cache)})", flush=True)


def _qa_export_training(output_path="outputs/qa_cache/training_data.jsonl"):
    """Экспорт кэша в JSONL для fine-tuning."""
    cache = _qa_load()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out), "w", encoding="utf-8") as f:
        for e in cache:
            line = json.dumps(
                {
                    "instruction": e["query"],
                    "output": e["short_answer"] or e["answer"][:500],
                    "sources": e["sources"],
                    "method": e["method"],
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")
    print(f"  Экспортировано {len(cache)} записей в {out}", flush=True)
