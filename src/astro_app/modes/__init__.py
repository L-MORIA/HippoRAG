"""Реестр всех 23 режимов поиска (шаг 7 миграции MAIN_SPLIT_PLAN.md).

Единственный источник истины для меню, диспетчера и _qa_run_test:
добавление режима = одна запись здесь, а не правка в 4 местах.
Контракт вызова: fn(hipporag, query) — дефолты каждого режима сохраняются
(Возражение 3 рецензии: top_k не навязывается).
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


@dataclass(frozen=True)
class ModeSpec:
    label: str  # короткое имя (логи, QA-кэш)
    menu: str  # строка меню (длинное описание)
    fn: Callable  # fn(hipporag, query) -> results | (results, extra)
    extra: bool = False  # True → возвращает (results, reformulated/hyde_text)
    star: bool = False  # ⭐ рекомендованный
    special: str = ""  # вторая строка меню (предупреждение о специализированности)


MODES: dict[str, ModeSpec] = {
    "1": ModeSpec(
        label="Multi-vector",
        menu="Multi-vector + LLM переформулировка",
        fn=dense.search_multi_vector,
        extra=True,
        star=True,
    ),
    "2": ModeSpec(
        label="Граф", menu="Графовый (entity linking + PPR)", fn=graph.search_hipporag_graph
    ),
    "3": ModeSpec(
        label="BM25+CE-en",
        menu="BM25 + Embedding + CE (English)",
        fn=bm25_hybrid.search_bm25_hybrid,
    ),
    "4": ModeSpec(
        label="Small→Big-en",
        menu="Small-to-big + CE (English)",
        fn=small_to_big.search_small_to_big,
    ),
    "5": ModeSpec(
        label="BM25+CE-ru",
        menu="BM25 + RU-CE + Query expansion",
        fn=bm25_hybrid.search_bm25_hybrid_ru,
    ),
    "6": ModeSpec(
        label="Small→Big-ru",
        menu="Small-to-big + RU-CE (русский)",
        fn=small_to_big.search_small_to_big_ru,
    ),
    "7": ModeSpec(
        label="HyDE",
        menu="HyDE + RU-CE (гипотетический документ)",
        fn=hyde.search_hyde,
        extra=True,
        star=True,
    ),
    "8": ModeSpec(label="Фильтр", menu="Фильтр по разделу + RU-CE", fn=filtered.search_filtered),
    "9": ModeSpec(
        label="Весь знак",
        menu="Весь материал по знаку зодиака (полностью)",
        fn=sign_zones.extract_sign_full,
    ),
    "10": ModeSpec(
        label="Резюме", menu="Весь материал + резюме LLM", fn=sign_zones.extract_sign_summary
    ),
    "11": ModeSpec(
        label="Обраб.",
        menu="Весь материал + LLM обработка (стиль автора)",
        fn=sign_zones.extract_sign_polished,
    ),
    "12": ModeSpec(
        label="Self-RAG",
        menu="Self-RAG (поиск + ответ LLM)",
        fn=self_rag.search_self_rag,
        star=True,
    ),
    "13": ModeSpec(
        label="RRF+Контекст",
        menu="RRF + контекстное окно + RU-CE",
        fn=bm25_hybrid.search_rrf_context,
    ),
    "14": ModeSpec(label="E5-rerank", menu="E5 sentence-rerank", fn=dense.search_e5_rerank),
    "15": ModeSpec(label="RRF+sign", menu="RRF + контекст + sign dominance", fn=dense.search_mxbai),
    "16": ModeSpec(label="ColBERT", menu="ColBERT-v2 late interaction", fn=dense.search_colbert),
    "17": ModeSpec(
        label="LLM-фильтр",
        menu="Любой метод + LLM фильтрация",
        fn=llm_filter.search_with_llm_filter,
    ),
    "18": ModeSpec(
        label="Qwen+FRIDA", menu="Qwen3 + FRIDA + BM25 + LLM", fn=llm_filter.search_qwen_frida
    ),
    "19": ModeSpec(
        label="GraphRAG",
        menu="GraphRAG Agent (маршрутизация + самокоррекция)",
        fn=graph.search_graphrag_agent,
        star=True,
    ),
    "20": ModeSpec(
        label="Sentence",
        menu="Sentence-level chunking (BM25 по предложениям)",
        fn=sentence_dat.search_sentence_level,
        star=True,
        special="(названия книг/автор — это не предложение, см. режимы 2 и 5)",
    ),
    "21": ModeSpec(
        label="DAT+LLM",
        menu="DAT Sentence-level + LLM (динамический α)",
        fn=sentence_dat.search_dat_sentence_llm,
        star=True,
    ),
    "22": ModeSpec(
        label="SemFilter",
        menu="Pattern Search: знак ↔ зона (зона или знак из запроса)",
        fn=sign_zones.search_semantic_filter,
        special="⚠ специализированный: нужен знак или зона в запросе, иначе честный отказ",
    ),
    "23": ModeSpec(
        label="QACache",
        menu="QA Cache (кэш ответов; без кэша — режим 22)",
        fn=qacache.search_qa_cache,
        special="⚠ тот же метод 22 — тоже нужен знак или зона",
    ),
}
