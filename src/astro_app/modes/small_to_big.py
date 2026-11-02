"""Режимы small_to_big — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import numpy as np

from ..primitives import _build_bm25, _get_chunk_meta
from ..query import ZODIAC_FORMS, _expand_query, split_query_terms
from ..rerankers import _get_crossencoder, _get_crossencoder_ru
from ..signs import _chunk_sign_score, _keyword_boost, _sign_in_query
from ..textutils import _tokenize


def search_small_to_big(hipporag, query, top_k=10, candidate_pool=100, context_window=2):
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
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = emb * (1.0 + boost)
        scored.append((idx, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Cross-encoder rerank top 50
    try:
        ce = _get_crossencoder()
        top_50 = scored[:50]
        pairs = [(query, chunk_store.texts[idx][:512]) for idx, *_ in top_50]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        for i, (idx, final, emb, bm25_s, boost) in enumerate(top_50):
            xc = float(xc_scores[i])
            final = emb * 0.8 * (1.0 + boost) + xc * 0.2
            scored[i] = (idx, final, emb, bm25_s, boost, xc)
    except Exception:
        pass

    scored.sort(key=lambda x: x[1], reverse=True)
    used_ranges = set()
    merged = []
    context_window = 1

    for idx, final, emb, bm25_s, boost, *_ in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-context_window * 2, context_window * 2 + 1):
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
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = final * (1.0 + boost_final * 0.5)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    # Обрезаем начало до заголовка нужного знака зодиака
    sign_in_result = _sign_in_query(q_lower)
    if sign_in_result:
        import re as _re

        trimmed = []
        for text, score, source, min_p, max_p, boost in merged:
            for form in ZODIAC_FORMS.get(sign_in_result, [sign_in_result]):
                pattern = rf"(?:^|[.!\n])\s*{_re.escape(form.upper())}(?:\s|[:,.!])"
                m = _re.search(pattern, text, _re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    if cut_pos > 0:
                        text = text[cut_pos:]
                    text = text.lstrip(" .\n\t,;:!?")
                    break
            trimmed.append((text, score, source, min_p, max_p, boost))
        merged = trimmed

    # Дедупликация: убираем дубликаты по содержимому
    seen_texts = set()
    deduped = []
    for text, score, source, min_p, max_p, boost in merged:
        # Сравниваем по первым 200 символам
        key = text[:200].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((text, score, source, min_p, max_p, boost))
    merged = deduped

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def search_small_to_big_ru(hipporag, query, top_k=10, candidate_pool=300, context_window=1):
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
    # Query expansion для BM25
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
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final = emb * (1.0 + boost)
        scored.append((idx, final, emb, float(bm25_scores[idx]), boost))
    scored.sort(key=lambda x: x[1], reverse=True)

    try:
        ce = _get_crossencoder_ru()
        top_30 = scored[:30]
        pairs = [(query, chunk_store.texts[idx][:512]) for idx, *_ in top_30]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        for i, (idx, final, emb, bm25_s, boost) in enumerate(top_30):
            xc = float(xc_scores[i])
            final = emb * 0.8 * (1.0 + boost) + xc * 0.2
            scored[i] = (idx, final, emb, bm25_s, boost, xc)
    except Exception:
        pass

    scored.sort(key=lambda x: x[1], reverse=True)
    used_ranges = set()
    merged = []

    for idx, final, emb, bm25_s, boost, *_ in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-context_window * 2, context_window * 2 + 1):
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
        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = final * (1.0 + boost_final * 0.5)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    sign_in_result = _sign_in_query(q_lower)
    if sign_in_result:
        import re as _re

        trimmed = []
        for text, score, source, min_p, max_p, boost in merged:
            for form in ZODIAC_FORMS.get(sign_in_result, [sign_in_result]):
                pattern = rf"(?:^|[.!\n])\s*{_re.escape(form.upper())}(?:\s|[:,.!])"
                m = _re.search(pattern, text, _re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    if cut_pos > 0:
                        text = text[cut_pos:]
                    text = text.lstrip(" .\n\t,;:!?")
                    break
            trimmed.append((text, score, source, min_p, max_p, boost))
        merged = trimmed

    seen_texts = set()
    deduped = []
    for text, score, source, min_p, max_p, boost in merged:
        key = text[:200].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((text, score, source, min_p, max_p, boost))
    merged = deduped

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]
