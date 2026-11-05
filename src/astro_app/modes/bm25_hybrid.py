"""Режимы bm25_hybrid — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import numpy as np

from ..page_retrieval import page_level_enabled, search_pages_hybrid
from ..primitives import _build_bm25, _get_chunk_meta
from ..query import _expand_query, split_query_terms
from ..rerankers import _rerank_with_crossencoder, _rerank_with_crossencoder_ru
from ..signs import _chunk_sign_score, _keyword_boost
from ..textutils import _tokenize


def search_bm25_hybrid(hipporag, query, top_k=10, candidate_pool=200):
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    q_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        final = emb * (1.0 + boost)
        scored.append((text, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_50 = scored[:50]
    rest = scored[50:]
    reranked = _rerank_with_crossencoder(query, top_50)
    return (reranked + rest)[:top_k]


def search_bm25_hybrid_ru(hipporag, query, top_k=10, candidate_pool=200):
    """BM25 + RU-CE + Query expansion (оригинальная версия)."""
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        emb = float(emb_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        final = emb * (1.0 + boost)
        scored.append((text, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_50 = scored[:50]
    rest = scored[50:]
    reranked = _rerank_with_crossencoder_ru(query, top_50)
    return (reranked + rest)[:top_k]


def search_bm25_hybrid_ru_pages(hipporag, query, top_k=5, **kwargs):
    """Page-level вариант режима 5 для H1-6 и Q&A-заземления.

    Возвращает ``PageHit`` (по одному на страницу), а не чанки: это поднимает
    page recall@5, потому что несколько чанков одной страницы больше не
    занимают несколько слотов выдачи.
    """
    return search_pages_hybrid(hipporag, query, top_k=top_k, **kwargs)


def bm25_hybrid_ru_mode(hipporag, query, top_k=10, candidate_pool=200):
    """Функция режима 5 в реестре: page-level H1-6 по умолчанию.

    По умолчанию возвращает ``PageHit`` (страница — единица выдачи продукта,
    page recall@5 87,0 % на M7); параметры page-пути — дефолты
    ``search_pages_hybrid`` (pool 50, max 4000 — как в замере M7).
    ``ASTRO_RETRIEVER_PAGE_LEVEL=0`` возвращает прежний чанковый путь
    ``search_bm25_hybrid_ru`` (кортежи текст/скор/dpr/bm25/boost) — он же
    используется режимами 19/21 и eval-адаптерами, поэтому сам не меняется.
    """
    if page_level_enabled():
        return search_pages_hybrid(hipporag, query, top_k=top_k)
    return search_bm25_hybrid_ru(hipporag, query, top_k=top_k, candidate_pool=candidate_pool)


def search_rrf_context(hipporag, query, top_k=10, candidate_pool=150, context_window=1):
    """
    Гибрид RRF (BM25 + Embedding) + контекстное окно + RU-CE.
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]

    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_raw = np.dot(all_emb, q_emb.T).squeeze()

    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_raw = bm25.get_scores(q_tokens)

    # Только top-candidates от BM25
    candidate_idx = set(np.argsort(bm25_raw)[::-1][:candidate_pool])

    # RRF среди кандидатов + sign dominance + keyword boost
    bm25_rank = {idx: pos for pos, idx in enumerate(np.argsort(bm25_raw)[::-1])}
    emb_rank = {idx: pos for pos, idx in enumerate(np.argsort(emb_raw)[::-1])}
    k = 60

    scored = []
    for idx in candidate_idx:
        rrf = 1 / (k + bm25_rank.get(idx, 1e6)) + 1 / (k + emb_rank.get(idx, 1e6))
        tl = chunk_store.texts[idx].lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        scored.append((idx, rrf, float(emb_raw[idx]), float(bm25_raw[idx]), boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # CE rerank top 30
    top_30 = [(chunk_store.texts[idx], s, e, b, boost) for idx, s, e, b, boost in scored[:30]]
    try:
        reranked = _rerank_with_crossencoder_ru(query, top_30)
        text_to_idx = {chunk_store.texts[i]: i for i in range(n_chunks)}
        reranked_idx = [text_to_idx[r[0]] for r in reranked if r[0] in text_to_idx]
    except Exception:
        reranked_idx = [scored[i][0] for i in range(min(30, len(scored)))]
    reranked_idx.extend([scored[i][0] for i in range(min(30, len(scored)), len(scored))])

    # Контекстное окно (только ±1 страница)
    used_ranges = set()
    merged = []
    for idx in reranked_idx:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-context_window, context_window + 1):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= context_window:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        min_p, max_p = min(pages), max(pages)
        rkey = f"{source}:{min_p}-{max_p}"
        if rkey in used_ranges:
            continue
        used_ranges.add(rkey)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        full_text = "\n\n".join(parts)
        tl_full = full_text.lower()
        boost = _keyword_boost(tl_full, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = float(emb_raw[idx]) * (1.0 + boost)
        merged.append((full_text, final, source, min_p, max_p, boost))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]
