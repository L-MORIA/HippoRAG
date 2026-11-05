"""Page-level retrieval для H1-6: страница — единица выдачи.

Зачем отдельный слой. Режим 5 возвращает чанки, а H1-6 меряет page recall@5:
несколько чанков одной страницы занимают слоты top-5, и страница, которая
в корпусе есть, может не попасть в пять «слотов». Здесь чанки группируются
по ``(source_file, page_num)``, BM25 считается по полному тексту страницы,
dense-канал — максимум по чанкам страницы, каналы объединяются RRF, а
top-``rerank_pool`` переранжируется русским cross-encoder.

Модуль не меняет выдачу CLI-режима 5: он используется адаптером
``scripts/grounding_astro_retriever.py`` для замера H1-6 и Q&A-заземления.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .query import _expand_query, split_query_terms
from .rerankers import _rerank_with_crossencoder_ru
from .signs import _keyword_boost
from .textutils import _tokenize

logger = logging.getLogger(__name__)

#: Константа RRF; 60 — стандартное значение из литературы.
RRF_K = 60
#: Сколько страниц отдавать в cross-encoder.
DEFAULT_RERANK_POOL = 50
#: Максимальная длина текста страницы во фрагменте (0 — не обрезать).
DEFAULT_MAX_CHARS = 4000
#: Сколько символов страницы видит cross-encoder (он сам режет по 512).
RERANK_MAX_CHARS = 512
#: Насколько keyword-boost влияет на RRF до переранка.
BOOST_WEIGHT = 0.1


@dataclass(frozen=True)
class PageRecord:
    """Страница корпуса, собранная из одного или нескольких чанков."""

    source_file: str
    doc_title: str
    page_num: int
    section: str
    text: str
    chunk_indices: Tuple[int, ...]


@dataclass
class PageHit:
    """Результат page-level поиска: страница, её текст и скоры каналов."""

    page: PageRecord
    text: str
    score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0
    best_chunk_index: int = -1


@dataclass
class PageIndex:
    """Кэш страничного индекса: страницы и BM25 по их текстам."""

    pages: List[PageRecord]
    bm25: Optional[BM25Okapi]


_CACHE_KEY: Optional[Tuple[int, int, int]] = None
_CACHE_INDEX: Optional[PageIndex] = None


def reset_page_index_cache() -> None:
    """Сбросить кэш (тесты, смена индекса в одном процессе)."""
    global _CACHE_KEY, _CACHE_INDEX
    _CACHE_KEY = None
    _CACHE_INDEX = None


def page_level_enabled() -> bool:
    """H1-6: page-level выдача режима 5 включена по умолчанию.

    Единица выдачи продукта — страница, а не чанк (page recall@5 87,0 % на
    M7 против потолка чанков). Чанковый вариант возвращается при
    ``ASTRO_RETRIEVER_PAGE_LEVEL=0`` (или false/no/off). Та же переменная и
    то же умолчание, что в адаптере ``scripts/grounding_astro_retriever.py``.
    """
    raw = os.getenv("ASTRO_RETRIEVER_PAGE_LEVEL", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _metadata_entries(store, idx: int) -> List[Dict]:
    """Метаданные чанка списком; поддерживает оба формата хранилища."""
    hash_id = store.hash_ids[idx]
    getter = getattr(store, "get_all_metadata_entries", None)
    if callable(getter):
        entries = getter(hash_id)
        if entries:
            return list(entries)

    get_row = getattr(store, "get_row", None)
    if not callable(get_row):
        return []
    row = get_row(hash_id)
    meta = row.get("metadata", {}) if isinstance(row, dict) else {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return []
    if isinstance(meta, dict):
        return [meta]
    if isinstance(meta, list):
        return [entry for entry in meta if isinstance(entry, dict)]
    return []


def _build_page_index(store) -> PageIndex:
    """Сгруппировать чанки в страницы и построить BM25 по текстам страниц."""
    global _CACHE_KEY, _CACHE_INDEX

    texts = getattr(store, "texts", None) or []
    key = (id(store), id(texts), len(texts))
    if _CACHE_KEY == key and _CACHE_INDEX is not None:
        return _CACHE_INDEX

    grouped: Dict[Tuple[str, int], Dict] = {}
    order: List[Tuple[str, int]] = []

    for idx, chunk_text in enumerate(texts):
        entries = _metadata_entries(store, idx)
        primary = entries[0] if entries else {}
        source_file = str(primary.get("source_file") or "")
        doc_title = str(primary.get("doc_title") or source_file or "")
        page_num = int(primary.get("page_num") or 0)
        if not (source_file or doc_title) or page_num < 1:
            continue

        page_key = (source_file or doc_title, page_num)
        if page_key not in grouped:
            grouped[page_key] = {
                "source_file": source_file or doc_title,
                "doc_title": doc_title or source_file,
                "page_num": page_num,
                "section": str(primary.get("section") or ""),
                "chunks": [],
            }
            order.append(page_key)
        grouped[page_key]["chunks"].append((idx, chunk_text))

    pages: List[PageRecord] = []
    for page_key in order:
        item = grouped[page_key]
        chunks = sorted(item["chunks"], key=lambda pair: pair[0])
        page_text = "\n\n".join(chunk_text for _, chunk_text in chunks).strip()
        if not page_text:
            continue
        pages.append(
            PageRecord(
                source_file=item["source_file"],
                doc_title=item["doc_title"],
                page_num=item["page_num"],
                section=item["section"],
                text=page_text,
                chunk_indices=tuple(idx for idx, _ in chunks),
            )
        )

    corpus = [_tokenize(page.text) for page in pages]
    bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None
    index = PageIndex(pages=pages, bm25=bm25)
    _CACHE_KEY = key
    _CACHE_INDEX = index
    return index


def _ranks(scores: Sequence[float]) -> List[int]:
    """Ранги 0..n-1: больше скор — меньше ранг; ничьи детерминированы индексом."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _chunk_dense_scores(hipporag, store, query: str) -> Optional[np.ndarray]:
    """Dense-скоры чанков; None, если эмбеддинги недоступны."""
    embeddings = getattr(store, "embeddings", None)
    if embeddings is None or len(embeddings) == 0:
        return None
    try:
        query_embedding = hipporag.embedding_model.encode([query])
        all_embeddings = np.asarray(embeddings)
        scores = np.dot(all_embeddings, query_embedding.T)
        return np.atleast_1d(np.asarray(scores).squeeze()).astype(float)
    except Exception as exc:  # pragma: no cover - зависит от модели/индекса
        logger.warning("page retrieval: dense-канал недоступен: %s", exc)
        return None


def _truncate_page(text: str, best_chunk_text: str, max_chars: int) -> str:
    """Обрезать длинную страницу вокруг лучшего чанка, а не с начала."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if best_chunk_text:
        position = text.find(best_chunk_text)
        if position >= 0:
            start = max(0, position - max(0, (max_chars - len(best_chunk_text)) // 2))
            end = min(len(text), start + max_chars)
            start = max(0, end - max_chars)
            return text[start:end]
    return text[:max_chars]


def _best_chunk_index(
    page: PageRecord,
    chunk_dense: Optional[np.ndarray],
    chunk_count: int,
) -> int:
    """Индекс лучшего чанка страницы: dense-максимум или первый чанк."""
    if chunk_dense is not None and len(chunk_dense) >= chunk_count:
        return max(page.chunk_indices, key=lambda idx: (float(chunk_dense[idx]), -idx))
    return page.chunk_indices[0]


def search_pages_hybrid(
    hipporag,
    query: str,
    top_k: int = 5,
    rerank_pool: int = DEFAULT_RERANK_POOL,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[PageHit]:
    """Найти top_k страниц гибридом BM25 (по странице) + dense (max по чанкам).

    Args:
        hipporag: объект HippoRAG с ``chunk_embedding_store`` и ``embedding_model``.
        query: поисковый запрос.
        top_k: сколько страниц вернуть.
        rerank_pool: сколько страниц отдать в RU cross-encoder.
        max_chars: ограничение текста страницы во фрагменте (0 — не обрезать).

    Returns:
        Список ``PageHit``, по одному на страницу, в порядке релевантности.
    """
    store = hipporag.chunk_embedding_store
    texts = getattr(store, "texts", None) or []
    if not texts:
        return []

    index = _build_page_index(store)
    if not index.pages:
        return []

    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [word for word in query.lower().split() if len(word) > 2]
    expanded_terms = _expand_query(q_lower, terms)
    # Синонимы могут быть фразами — разбиваем их на токены, иначе BM25
    # сравнивает фразу целиком и не находит ни одного документа.
    expanded_tokens: List[str] = []
    for term in expanded_terms:
        expanded_tokens.extend(_tokenize(term))
    query_tokens = list(set(_tokenize(query) + expanded_tokens))

    bm25_scores = (
        [float(score) for score in index.bm25.get_scores(query_tokens)]
        if index.bm25 is not None
        else [0.0] * len(index.pages)
    )

    chunk_dense = _chunk_dense_scores(hipporag, store, query)
    if chunk_dense is not None and len(chunk_dense) >= len(texts):
        dense_scores: Optional[List[float]] = [
            max(float(chunk_dense[idx]) for idx in page.chunk_indices) for page in index.pages
        ]
    else:
        dense_scores = None

    bm25_ranks = _ranks(bm25_scores)
    dense_ranks = _ranks(dense_scores) if dense_scores is not None else None
    boosts = [_keyword_boost(page.text.lower(), q_lower, terms) for page in index.pages]

    scored: List[Tuple[PageRecord, float, float, float, float]] = []
    for position, page in enumerate(index.pages):
        rrf = 1.0 / (RRF_K + bm25_ranks[position])
        if dense_ranks is not None:
            rrf += 1.0 / (RRF_K + dense_ranks[position])
        base = rrf * (1.0 + BOOST_WEIGHT * boosts[position])
        scored.append(
            (
                page,
                base,
                bm25_scores[position],
                dense_scores[position] if dense_scores is not None else 0.0,
                boosts[position],
            )
        )
    scored.sort(key=lambda item: item[1], reverse=True)

    candidates = scored[: max(1, rerank_pool)]
    rerank_input: List[Tuple[str, float, float, float, float]] = []
    by_text: Dict[str, Tuple[PageRecord, float, float, int]] = {}
    for page, base, bm25_score, dense_score, _ in candidates:
        best_idx = _best_chunk_index(page, chunk_dense, len(texts))
        # Реранкер режет вход по 512 символам; отдаём ему окно вокруг лучшего
        # чанка, иначе на длинной странице он видит только её начало.
        rerank_text = _truncate_page(page.text, texts[best_idx], RERANK_MAX_CHARS)
        if rerank_text in by_text:
            continue
        rerank_input.append((rerank_text, base, dense_score, bm25_score, 0.0))
        by_text[rerank_text] = (page, bm25_score, dense_score, best_idx)

    reranked = _rerank_with_crossencoder_ru(query, rerank_input)

    hits: List[PageHit] = []
    seen_pages = set()
    for row in reranked:
        found = by_text.get(row[0])
        if found is None:
            continue
        page, bm25_score, dense_score, best_idx = found
        if page in seen_pages:
            continue
        seen_pages.add(page)
        hits.append(
            PageHit(
                page=page,
                text=_truncate_page(page.text, texts[best_idx], max_chars),
                score=float(row[1]),
                bm25_score=bm25_score,
                dense_score=dense_score,
                best_chunk_index=best_idx,
            )
        )
        if len(hits) >= top_k:
            break

    # Переранкер теоретически может вернуть не все строки (например, при
    # сбое) — добираем кандидатов в порядке RRF, чтобы top_k был стабильным.
    if len(hits) < top_k:
        for page, _, bm25_score, dense_score, _ in scored:
            if page in seen_pages:
                continue
            seen_pages.add(page)
            best_idx = _best_chunk_index(page, chunk_dense, len(texts))
            hits.append(
                PageHit(
                    page=page,
                    text=_truncate_page(page.text, texts[best_idx], max_chars),
                    score=0.0,
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                    best_chunk_index=best_idx,
                )
            )
            if len(hits) >= top_k:
                break

    return hits[:top_k]
