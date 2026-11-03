"""Реестр всех 23 режимов поиска (шаг 7 миграции MAIN_SPLIT_PLAN.md).

Единственный источник истины для меню, диспетчера и _qa_run_test:
добавление режима = одна запись здесь, а не правка в 4 местах.
Контракт вызова: fn(hipporag, query) — дефолты каждого режима сохраняются
(Возражение 3 рецензии: top_k не навязывается).

Рекомендация режимов (H1-2, закрыто 2026-11-03 по данным H1-1 —
`eval/dataset/questions.jsonl`, 40 размеченных вопросов):

* `DEFAULT_MODE` = **«2» Граф** — режим по умолчанию;
* `RECOMMENDED_MODES` = **«1» и «2»** — две рекомендованные звёздочки;
* остальные 21 режим — «экспертные», у каждого в `when` записано, когда он
  уместен.

Основание выбора (приёмка H1-2: «default не хуже лучшего режима на 80 %
eval-запросов; p95 ≤ 10 с»):

* режим 1 «Multi-vector» — лучший измеренный recall@5 (97,5 %), но p95 12,8 с,
  то есть **вне бюджета** H1-3 (≤ 10 с), и требует LLM;
* режим 2 «Граф» — recall@5 95,0 % (хуже всего на 1 вопрос из 40 = 2,5 п.п.
  при допустимых 20 %), p50 545 мс, LLM в пути выдачи не нужен ⇒ единственный
  измеренный режим, укладывающийся в бюджет;
* режим 5 «BM25+CE-ru» на датасете H1-1 **не измерялся** («8/8 адекватности,
  2 с» — это замер `docs/ROADMAP.md` на 8 контрольных вопросах), поэтому
  назначать его default до замера нельзя. Замер: `scripts/benchmark_budget.py
  --modes 1,2,5,21`; после него смена default = одна строка здесь.
"""

from dataclasses import dataclass
from typing import Callable

from . import (
    bm25_hybrid,
    dense,
    filtered,
    graph,
    hyde,
    llm_filter,
    qacache,
    self_rag,
    sentence_dat,
    sign_zones,
    small_to_big,
)

#: Режим по умолчанию (H1-2).
DEFAULT_MODE = "2"

#: Рекомендованные режимы — «звёздочки» в меню (H1-2).
RECOMMENDED_MODES: tuple[str, ...] = ("1", "2")


@dataclass(frozen=True)
class ModeSpec:
    label: str  # короткое имя (логи, QA-кэш)
    menu: str  # строка меню (длинное описание)
    fn: Callable  # fn(hipporag, query) -> results | (results, extra)
    extra: bool = False  # True → возвращает (results, reformulated/hyde_text)
    star: bool = False  # ⭐ рекомендованный
    special: str = ""  # вторая строка меню (предупреждение о специализированности)
    when: str = ""  # когда уместен режим (H1-2)


MODES: dict[str, ModeSpec] = {
    "1": ModeSpec(
        label="Multi-vector",
        menu="Multi-vector + LLM переформулировка",
        fn=dense.search_multi_vector,
        extra=True,
        star=True,
        when="максимальное качество на H1-1 (recall@5 97,5 %); нужен LLM, p95 12,8 с",
    ),
    "2": ModeSpec(
        label="Граф",
        menu="Графовый (entity linking + PPR)",
        fn=graph.search_hipporag_graph,
        star=True,
        when="по умолчанию: recall@5 95,0 %, p50 545 мс, LLM не нужен",
    ),
    "3": ModeSpec(
        label="BM25+CE-en",
        menu="BM25 + Embedding + CE (English)",
        fn=bm25_hybrid.search_bm25_hybrid,
        when="английский корпус: быстрый лексический + векторный гибрид",
    ),
    "4": ModeSpec(
        label="Small→Big-en",
        menu="Small-to-big + CE (English)",
        fn=small_to_big.search_small_to_big,
        when="английский корпус, иерархия «маленький чанк → большой»",
    ),
    "5": ModeSpec(
        label="BM25+CE-ru",
        menu="BM25 + RU-CE + Query expansion",
        fn=bm25_hybrid.search_bm25_hybrid_ru,
        when="русские запросы без LLM; на H1-1 не измерялся — кандидат на default",
    ),
    "6": ModeSpec(
        label="Small→Big-ru",
        menu="Small-to-big + RU-CE (русский)",
        fn=small_to_big.search_small_to_big_ru,
        when="русский корпус, иерархия «маленький чанк → большой»",
    ),
    "7": ModeSpec(
        label="HyDE",
        menu="HyDE + RU-CE (гипотетический документ)",
        fn=hyde.search_hyde,
        extra=True,
        when="сложные, плохо лексикальные запросы; требует LLM",
    ),
    "8": ModeSpec(
        label="Фильтр",
        menu="Фильтр по разделу + RU-CE",
        fn=filtered.search_filtered,
        when="запрос относится к конкретному разделу корпуса",
    ),
    "9": ModeSpec(
        label="Весь знак",
        menu="Весь материал по знаку зодиака (полностью)",
        fn=sign_zones.extract_sign_full,
        when="нужен весь материал по одному знаку (ответ-режим, без LLM)",
    ),
    "10": ModeSpec(
        label="Резюме",
        menu="Весь материал + резюме LLM",
        fn=sign_zones.extract_sign_summary,
        when="сводка по знаку силами LLM",
    ),
    "11": ModeSpec(
        label="Обраб.",
        menu="Весь материал + LLM обработка (стиль автора)",
        fn=sign_zones.extract_sign_polished,
        when="переработка материала по знаку в стиле автора (LLM)",
    ),
    "12": ModeSpec(
        label="Self-RAG",
        menu="Self-RAG (поиск + ответ LLM)",
        fn=self_rag.search_self_rag,
        when="ответ с самопроверкой; p95 37,9 с — экспериментальный режим",
    ),
    "13": ModeSpec(
        label="RRF+Контекст",
        menu="RRF + контекстное окно + RU-CE",
        fn=bm25_hybrid.search_rrf_context,
        when="гибрид RRF с расширенным контекстным окном",
    ),
    "14": ModeSpec(
        label="E5-rerank",
        menu="E5 sentence-rerank",
        fn=dense.search_e5_rerank,
        when="точный реранк выдачи моделью E5",
    ),
    "15": ModeSpec(
        label="RRF+sign",
        menu="RRF + контекст + sign dominance",
        fn=dense.search_mxbai,
        when="RRF с приоритетом упоминаний знака",
    ),
    "16": ModeSpec(
        label="ColBERT",
        menu="ColBERT-v2 late interaction",
        fn=dense.search_colbert,
        when="близкие по смыслу тексты, где важна поздняя точность",
    ),
    "17": ModeSpec(
        label="LLM-фильтр",
        menu="Любой метод + LLM фильтрация",
        fn=llm_filter.search_with_llm_filter,
        when="нужна строгая отборка выдачи LLM поверх любого метода",
    ),
    "18": ModeSpec(
        label="Qwen+FRIDA",
        menu="Qwen3 + FRIDA + BM25 + LLM",
        fn=llm_filter.search_qwen_frida,
        when="сложные многошаговые запросы с декомпозицией (LLM)",
    ),
    "19": ModeSpec(
        label="GraphRAG",
        menu="GraphRAG Agent (маршрутизация + самокоррекция)",
        fn=graph.search_graphrag_agent,
        when="агентная маршрутизация; p95 33,0 с — экспериментальный режим",
    ),
    "20": ModeSpec(
        label="Sentence",
        menu="Sentence-level chunking (BM25 по предложениям)",
        fn=sentence_dat.search_sentence_level,
        when="названия книг и авторов (это не предложение)",
        special="(названия книг/автор — это не предложение, см. режимы 2 и 5)",
    ),
    "21": ModeSpec(
        label="DAT+LLM",
        menu="DAT Sentence-level + LLM (динамический α)",
        fn=sentence_dat.search_dat_sentence_llm,
        when="нужен готовый связный ответ, а не выдача фрагментов",
    ),
    "22": ModeSpec(
        label="SemFilter",
        menu="Pattern Search: знак ↔ зона (зона или знак из запроса)",
        fn=sign_zones.search_semantic_filter,
        when="запрос про конкретный знак или зону",
        special="⚠ специализированный: нужен знак или зона в запросе, иначе честный отказ",
    ),
    "23": ModeSpec(
        label="QACache",
        menu="QA Cache (кэш ответов; без кэша — режим 22)",
        fn=qacache.search_qa_cache,
        when="повторяющиеся вопросы — отдача из кэша ответов",
        special="⚠ тот же метод 22 — тоже нужен знак или зона",
    ),
}


def recommended_modes() -> dict[str, ModeSpec]:
    """Рекомендованные режимы («звёздочки») в порядке реестра."""
    return {num: spec for num, spec in MODES.items() if num in RECOMMENDED_MODES}


def expert_modes() -> dict[str, ModeSpec]:
    """Экспертные режимы (всё, что не рекомендовано) в порядке реестра."""
    return {num: spec for num, spec in MODES.items() if num not in RECOMMENDED_MODES}
