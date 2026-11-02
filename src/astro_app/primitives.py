"""Примитивы поиска: BM25-корпус, dense-скоры, метаданные чанка.

Перенос из main.py (шаг 4 миграции MAIN_SPLIT_PLAN.md) — тела извлечены
байт-в-байт (823-846, 1143-1164); _tokenize импортируется из .textutils.
"""

import json

import numpy as np
from rank_bm25 import BM25Okapi

from .textutils import _tokenize


def _build_bm25(chunk_store):
    # Кэш привязан к конкретному хранилищу: раньше он был глобальным и при
    # смене индекса (или в тестах) молча возвращал BM25 от другого корпуса.
    cache_key = (id(chunk_store), len(chunk_store.texts))
    if getattr(_build_bm25, "cache_key", None) != cache_key:
        corpus = [_tokenize(t) for t in chunk_store.texts]
        _build_bm25.cache = BM25Okapi(corpus)
        _build_bm25.cache_key = cache_key
    return _build_bm25.cache


def _embedding_scores(hipporag, chunk_store, query):
    """Косинусные скоры запроса по всем чанкам (dense-канал).

    Возвращает None, если эмбеддинги недоступны — вызывающий код тогда
    остаётся на BM25. Нужно для Self-RAG (P1-6): чистый BM25 не находит
    титульные страницы, потому что там нет слов вопроса.
    """
    embeddings = getattr(chunk_store, "embeddings", None)
    if embeddings is None or len(embeddings) == 0:
        return None
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.asarray(embeddings)
    return np.atleast_1d(np.asarray(np.dot(all_emb, q_emb.T)).squeeze())


def _get_chunk_meta(chunk_store, idx):
    """Метаданные чанка: (source_file, page_num).

    Формат метаданных менялся (S-12): раньше это был dict, теперь — список
    dict'ов, по одному на каждый источник документа. Поддерживаются оба
    формата (и JSON-строка), иначе m.get() бросает AttributeError, исключение
    глушится и функция возвращает ("", 0) для КАЖДОГО чанка — из-за чего все
    вызовы с "if not source: continue" отбрасывали кандидатов и 8 режимов
    меню возвращали пустоту (P0-2).
    """
    try:
        row = chunk_store.get_row(chunk_store.hash_ids[idx])
        m = row.get("metadata", {}) if isinstance(row, dict) else {}
        if isinstance(m, str):
            m = json.loads(m)
        if isinstance(m, list):
            m = m[0] if m else {}  # список источников → первичный источник
        if not isinstance(m, dict):
            return "", 0
        return str(m.get("source_file", "")), int(m.get("page_num", 0))
    except Exception:
        return "", 0
