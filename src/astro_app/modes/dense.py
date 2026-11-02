"""Режимы dense — перенос из main.py (шаг 6, байт-в-байт)."""

import numpy as np

from ..primitives import _build_bm25, _get_chunk_meta
from ..query import _expand_query, reformulate_query, split_query_terms
from ..rerankers import _get_e5
from ..signs import _chunk_sign_score, _keyword_boost
from ..textutils import _tokenize


def search_multi_vector(hipporag, query, top_k=10, candidate_pool=200):
    """
    Multi-vector поиск:
    1. Переформулировка запроса через LLM
    2. Полный запрос → эмбеддинг + similarity
    3. Каждый терм запроса → эмбеддинг + similarity (max-pooling)
    4. Большой keyword boost (1.0)
    5. Комбинированный скор
    """
    # Шаг 0: Переформулировка
    search_query = reformulate_query(hipporag, query)

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        print("No chunk embeddings found.")
        return []

    all_emb = np.array(chunk_store.embeddings)
    n_chunks = len(all_emb)

    # Шаг 1: эмбеддинг полного запроса
    full_emb = hipporag.embedding_model.encode([search_query])
    full_scores = np.dot(all_emb, full_emb.T).squeeze()

    # Шаг 2: эмбеддинг каждого терма
    terms, q_lower = split_query_terms(search_query)
    if not terms:
        terms = [w for w in search_query.lower().split() if len(w) > 2]

    multi_scores = np.zeros(n_chunks)
    if terms:
        term_embs = hipporag.embedding_model.encode(terms)
        for te in term_embs:
            ts = np.dot(all_emb, te.reshape(-1, 1)).squeeze()
            multi_scores = np.maximum(multi_scores, ts)  # max-pooling

    # Шаг 3: сортировка по full_scores для candidate pool
    sorted_idx = np.argsort(full_scores)[::-1]
    pool = min(candidate_pool, n_chunks)

    scored = []
    for i in range(pool):
        idx = sorted_idx[i]
        text = chunk_store.texts[idx]
        tl = text.lower()

        dpr = float(full_scores[idx])
        multi = float(multi_scores[idx]) if terms else 0.0

        # Keyword boost
        boost = 0.0
        if q_lower in tl:
            boost += 1.0
        if terms:
            matches = sum(1 for t in terms if t in tl)
            boost += 0.3 * min(matches, 5) / max(len(terms), 1)

        # Комбинируем: full-query (40%) + multi-vector (60%) + keyword boost
        combo = dpr * 0.4 + multi * 0.6
        final = combo * (1.0 + boost)
        scored.append((text, final, dpr, multi, boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k], search_query


def search_mxbai(hipporag, query, top_k=10, candidate_pool=200):
    """
    Поиск через BM25 + существующие эмбеддинги (RRF) + sign dominance.
    mxbai-embed и mxbai-rerank — английские модели, не работают для русского домена.
    Метод использует проверенную RRF-комбинацию (как метод 13).
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)

    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()

    # RRF
    bm25_rank = np.argsort(bm25_scores)[::-1]
    emb_rank = np.argsort(emb_scores)[::-1]
    bm25_pos = {idx: pos for pos, idx in enumerate(bm25_rank)}
    emb_pos = {idx: pos for pos, idx in enumerate(emb_rank)}
    k = 60
    rrf_scores = [
        1 / (k + bm25_pos.get(i, 1e6)) + 1 / (k + emb_pos.get(i, 1e6)) for i in range(n_chunks)
    ]
    candidate_idx = np.argsort(rrf_scores)[::-1][:candidate_pool]
    if len(candidate_idx) == 0:
        return []

    print("  ✅ RRF (BM25 + эмбеддинги) + sign dominance + контекст", flush=True)

    scored = []
    for idx in candidate_idx:
        text = chunk_store.texts[idx]
        tl = text.lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        emb = float(emb_scores[idx])
        rrf = float(rrf_scores[idx])
        final = emb * 0.6 + rrf * 0.4 * (1.0 + boost)
        scored.append((idx, final, emb, rrf, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Контекстное окно ±1
    used_ranges = set()
    merged = []
    for idx, final, emb_sim, rrf, boost in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-1, 2):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
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
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = emb_sim * (1.0 + boost_final)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def search_e5_rerank(hipporag, query, top_k=10, candidate_pool=200):
    """
    Поиск: существующие эмбеддинги (1369 чанков) → E5 sentence-rerank top 50.
    E5 эмбеддит только query + предложения топ-50 кандидатов (не все чанки).
    Качество не падает — BM25 + существующие эмбеддинги отбирают лучших кандидатов,
    E5 только доуточняет ранжирование по предложениям.
    """
    import re as _re

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)

    # Кандидаты: комбинация BM25 + существующие эмбеддинги (все 1369 чанков)
    q_emb = hipporag.embedding_model.encode([query])
    all_emb = np.array(chunk_store.embeddings)
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()
    bm25_rank = np.argsort(bm25_scores)[::-1]
    emb_rank = np.argsort(emb_scores)[::-1]
    bm25_pos = {idx: pos for pos, idx in enumerate(bm25_rank)}
    emb_pos = {idx: pos for pos, idx in enumerate(emb_rank)}
    k = 60
    rrf_scores = [
        1 / (k + bm25_pos.get(i, 1e6)) + 1 / (k + emb_pos.get(i, 1e6)) for i in range(n_chunks)
    ]
    candidate_idx = np.argsort(rrf_scores)[::-1][:candidate_pool]

    if len(candidate_idx) == 0:
        return []

    # E5: только для sentence-reranking топ-50
    e5 = _get_e5()
    q_sentences = [s.strip() for s in _re.split(r"[.!?]+", query) if len(s.strip()) > 3]
    if not q_sentences:
        q_sentences = [query]
    q_sent_embs = e5.encode([f"query: {s}" for s in q_sentences])

    top50 = candidate_idx[:50]
    rest = candidate_idx[50:]

    scored = []
    for idx in top50:
        text = chunk_store.texts[idx]
        tl = text.lower()

        # Sentence-level через E5
        chunk_sents = [s.strip() for s in _re.split(r"[.!?]+", text) if len(s.strip()) > 10]
        sent_score = 0
        if chunk_sents:
            chunk_sents = chunk_sents[:10]
            cs_embs = e5.encode([f"passage: {s}" for s in chunk_sents])
            sim = np.dot(cs_embs, q_sent_embs.T)
            sent_score = float(np.max(sim))

        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score

        final = (
            float(emb_scores[idx]) * 0.4
            + sent_score * 0.4
            + float(emb_scores[idx]) * 0.2 * max(boost, 0)
        )
        scored.append((idx, final, float(emb_scores[idx]), sent_score, boost))

    for idx in rest:
        text = chunk_store.texts[idx]
        tl = text.lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = float(emb_scores[idx]) * (1.0 + boost)
        scored.append((idx, final, float(emb_scores[idx]), 0, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Контекстное окно ±1
    used_ranges = set()
    merged = []
    for idx, final, emb_sim, sent_score, boost in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-1, 2):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
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
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = emb_sim * (1.0 + boost_final) + sent_score * 0.3
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]
