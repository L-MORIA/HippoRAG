"""Режимы filtered — перенос из main.py (шаг 6, байт-в-байт)."""

import numpy as np

from ..primitives import _build_bm25, _get_chunk_meta
from ..query import ZODIAC_FORMS, _expand_query, split_query_terms
from ..rerankers import _get_crossencoder_ru
from ..signs import _chunk_sign_score, _keyword_boost, _sign_in_query
from ..textutils import _tokenize
from .llm_filter import _filter_by_section


def search_filtered(hipporag, query, top_k=10, candidate_pool=200):
    """
    Small-to-big + RU-CE с фильтрацией по разделу знака зодиака.
    Если запрос про Тельца — показываем ТОЛЬКО чанки из раздела Тельца.
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
    emb_scores = np.dot(all_emb, q_emb.T).squeeze()

    expanded_terms = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + expanded_terms))
    bm25_scores = bm25.get_scores(q_tokens)

    # BM25 candidates → фильтрация по разделу
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
    candidate_idx = _filter_by_section(candidate_idx, chunk_store, q_lower)

    if len(candidate_idx) == 0:
        return []

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

    # RU Cross-encoder rerank top 30
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

    # Small-to-big merging
    used_ranges = set()
    merged = []
    for idx, final, emb, bm25_s, boost, *_ in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for offset in range(-2, 3):
            ni = idx + offset
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 2:
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

        # Обрезка до заголовка знака
        sign_in_result = _sign_in_query(q_lower)
        if sign_in_result:
            import re as _re

            for form in ZODIAC_FORMS.get(sign_in_result, [sign_in_result]):
                pattern = rf"(?:^|[.!\n])\s*{_re.escape(form.upper())}(?:\s|[:,.!])"
                m = _re.search(pattern, full_text, _re.IGNORECASE)
                if m:
                    cut_pos = m.start()
                    if cut_pos > 0:
                        full_text = full_text[cut_pos:]
                    full_text = full_text.lstrip(" .\n\t,;:!?")
                    break

        boost_final = _keyword_boost(full_text.lower(), q_lower, terms)
        final_score = final * (1.0 + boost_final * 0.5)
        merged.append((full_text, final_score, source, min_p, max_p, boost_final))
        if len(merged) >= top_k:
            break

    # Дедупликация
    seen_texts = set()
    deduped = []
    for text, score, source, min_p, max_p, boost in merged:
        key = text[:200].lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        deduped.append((text, score, source, min_p, max_p, boost))

    deduped.sort(key=lambda x: x[1], reverse=True)
    return deduped[:top_k]
