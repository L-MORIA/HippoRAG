"""Тонкая точка входа Astrology RAG (шаг 7 миграции MAIN_SPLIT_PLAN.md).

Вся логика живёт в пакете astro_app; этот файл — фасад:
1. CLI-запуск (hipporag.bat, menu.py, test_menu.bat): python main.py
2. Реэкспорт публичного API для обратной совместимости — тесты (tests/unit/*)
   и scripts/evaluate_search.py импортируют функции через `import main`.
Имена с подчёркиванием реэкспортируются сознательно: их используют тесты.
"""

from multiprocessing import freeze_support

from astro_app.cli import MODEL_OPTIONS, main

# — конфигурация —
from astro_app.config import (
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_NAME,
    SAVE_DIR,
    SENTENCE_START,
    STOP_WORDS,
)

# — инфраструктура —
from astro_app.display import show_results

# — режимы (23) —
from astro_app.modes.bm25_hybrid import (
    search_bm25_hybrid,
    search_bm25_hybrid_ru,
    search_rrf_context,
)
from astro_app.modes.dense import (
    search_colbert,
    search_e5_rerank,
    search_multi_vector,
    search_mxbai,
)
from astro_app.modes.filtered import search_filtered
from astro_app.modes.graph import search_graphrag_agent, search_hipporag_graph
from astro_app.modes.hyde import search_hyde
from astro_app.modes.llm_filter import (
    _filter_by_section,
    _get_frida,
    _get_qwen,
    _llm_filter_results,
    search_qwen_frida,
    search_with_llm_filter,
)
from astro_app.modes.qacache import search_qa_cache
from astro_app.modes.self_rag import search_self_rag
from astro_app.modes.sentence_dat import (
    BIBLIO_QUERY_PATTERNS,
    _dat_combine_results,
    _dat_compute_alpha,
    _dat_evaluate,
    _is_bibliographic_query,
    search_dat_sentence_llm,
    search_sentence_level,
)
from astro_app.modes.sign_zones import (
    _find_sign_zones,
    _find_zone_signs,
    _sign_zone_answer,
    _sign_zone_class,
    _zones_in_text,
    extract_sign_full,
    extract_sign_polished,
    extract_sign_summary,
    search_semantic_filter,
)
from astro_app.modes.small_to_big import search_small_to_big, search_small_to_big_ru
from astro_app.primitives import _build_bm25, _embedding_scores, _get_chunk_meta
from astro_app.qa_cache import (
    QA_CACHE_PATH,
    QA_CACHE_VERSION,
    _qa_derive_short,
    _qa_export_training,
    _qa_is_current,
    _qa_load,
    _qa_lookup,
    _qa_norm,
    _qa_save,
    _qa_store,
)
from astro_app.query import (
    _ALL_SIGN_FORMS_RE,
    _META_REFORMULATION_MARKERS,
    _SIGN_RE,
    ASTRO_TERMS,
    FORM_TO_SIGN,
    ZODIAC_FORMS,
    ZODIAC_SIGNS,
    _expand_query,
    _is_meta_reformulation,
    _word_re,
    reformulate_query,
    split_query_terms,
)
from astro_app.rerankers import (
    _colbert_encode,
    _colbert_score,
    _get_colbert,
    _get_crossencoder,
    _get_crossencoder_ru,
    _get_e5,
    _rerank_with_crossencoder,
    _rerank_with_crossencoder_ru,
)
from astro_app.signs import (
    _build_sign_index,
    _chunk_sign_score,
    _count_sign,
    _keyword_boost,
    _sign_in_query,
)
from astro_app.synonyms import load_synonyms
from astro_app.textutils import _tokenize, fix_chunk_start, show_text_safe

# Словарь синонимов: данные пакета; main.py оставляет прежнее имя переменной
SYNONYM_MAP = load_synonyms()

__all__ = [
    "main",
    "MODEL_OPTIONS",
    "SAVE_DIR",
    "LLM_NAME",
    "LLM_BASE_URL",
    "EMBEDDING_MODEL",
    "STOP_WORDS",
    "SENTENCE_START",
    "SYNONYM_MAP",
    "show_results",
    "show_text_safe",
    "fix_chunk_start",
    "_tokenize",
    "_build_bm25",
    "_embedding_scores",
    "_get_chunk_meta",
    "QA_CACHE_PATH",
    "QA_CACHE_VERSION",
    "_qa_derive_short",
    "_qa_export_training",
    "_qa_is_current",
    "_qa_load",
    "_qa_lookup",
    "_qa_norm",
    "_qa_save",
    "_qa_store",
    "_expand_query",
    "_is_meta_reformulation",
    "_word_re",
    "reformulate_query",
    "split_query_terms",
    "ASTRO_TERMS",
    "FORM_TO_SIGN",
    "ZODIAC_FORMS",
    "ZODIAC_SIGNS",
    "_ALL_SIGN_FORMS_RE",
    "_META_REFORMULATION_MARKERS",
    "_SIGN_RE",
    "_colbert_encode",
    "_colbert_score",
    "_get_colbert",
    "_get_crossencoder",
    "_get_crossencoder_ru",
    "_get_e5",
    "_rerank_with_crossencoder",
    "_rerank_with_crossencoder_ru",
    "_build_sign_index",
    "_chunk_sign_score",
    "_count_sign",
    "_keyword_boost",
    "_sign_in_query",
    "_build_bm25",
    "search_bm25_hybrid",
    "search_bm25_hybrid_ru",
    "search_colbert",
    "search_e5_rerank",
    "search_mxbai",
    "search_multi_vector",
    "search_rrf_context",
    "search_graphrag_agent",
    "search_hipporag_graph",
    "search_filtered",
    "search_small_to_big",
    "search_small_to_big_ru",
    "search_hyde",
    "search_self_rag",
    "search_sentence_level",
    "search_dat_sentence_llm",
    "search_with_llm_filter",
    "search_qwen_frida",
    "search_semantic_filter",
    "search_qa_cache",
    "extract_sign_full",
    "extract_sign_polished",
    "extract_sign_summary",
    "_filter_by_section",
    "_get_frida",
    "_get_qwen",
    "_llm_filter_results",
    "_is_bibliographic_query",
    "BIBLIO_QUERY_PATTERNS",
    "_dat_combine_results",
    "_dat_compute_alpha",
    "_dat_evaluate",
    "_find_sign_zones",
    "_find_zone_signs",
    "_sign_zone_answer",
    "_sign_zone_class",
    "_zones_in_text",
]


if __name__ == "__main__":
    freeze_support()
    main()
