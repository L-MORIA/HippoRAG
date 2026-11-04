"""Режимы self_rag — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import numpy as np

from ..grounded_qa import Fragment, answer_grounded, used_sources
from ..primitives import _build_bm25, _embedding_scores, _get_chunk_meta
from ..query import _expand_query, split_query_terms
from ..signs import _chunk_sign_score, _keyword_boost
from ..textutils import _tokenize


def search_self_rag(hipporag, query, top_k=30, candidate_pool=200):
    """
    Self-RAG: BM25 + dense (RRF) → LLM формулирует ответ.

    Раньше кандидатов отбирал чистый BM25 (P1-6): на вопрос «какие книги
    написала Максимова И.В.?» он поднимал прозу корпуса («например, написала
    я книгу») выше титульных страниц, потому что в них нет слов вопроса.
    Теперь кандидаты — объединение BM25-топа и dense-топа, а ранжирование
    считается по RRF: при отсутствии эмбеддингов поведение остаётся прежним.
    """
    _, q_lower = split_query_terms(query)
    terms, _ = split_query_terms(query)

    chunk_store = hipporag.chunk_embedding_store
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    bm25_scores = np.asarray(bm25.get_scores(q_tokens))
    emb_scores = _embedding_scores(hipporag, chunk_store, query)

    # Шаг 1: кандидаты = BM25-топ ∪ dense-топ, ранжирование RRF
    if emb_scores is None or len(emb_scores) != n_chunks:
        candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]
        ranked = [(int(idx), float(bm25_scores[idx])) for idx in candidate_idx]
    else:
        bm25_pos = {int(idx): pos for pos, idx in enumerate(np.argsort(bm25_scores)[::-1])}
        emb_pos = {int(idx): pos for pos, idx in enumerate(np.argsort(emb_scores)[::-1])}
        pool = set(np.argsort(bm25_scores)[::-1][:candidate_pool].tolist())
        pool |= set(np.argsort(emb_scores)[::-1][:candidate_pool].tolist())
        k = 60
        ranked = [
            (idx, 1.0 / (k + bm25_pos.get(idx, 1e6)) + 1.0 / (k + emb_pos.get(idx, 1e6)))
            for idx in pool
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)

    scored = []
    for idx, base in ranked:
        tl = chunk_store.texts[idx].lower()
        boost = _keyword_boost(tl, q_lower, terms) + _chunk_sign_score(chunk_store, idx, q_lower)
        final = base * (1.0 + boost)
        scored.append((chunk_store.texts[idx], final, boost, idx))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Контекстное окно: добавляем соседние чанки
    used_ranges = set()
    merged = []
    for text, base_score, boost, idx in scored:
        source, page = _get_chunk_meta(chunk_store, idx)
        if not source:
            continue
        window = set([idx])
        for off in range(-1, 2):
            ni = idx + off
            if ni < 0 or ni >= n_chunks:
                continue
            ns, np_ = _get_chunk_meta(chunk_store, ni)
            if ns == source and abs(np_ - page) <= 1:
                window.add(ni)
        pages = set()
        for wi in window:
            _, wp = _get_chunk_meta(chunk_store, wi)
            pages.add(wp)
        mp, xp = min(pages), max(pages)
        rk = f"{source}:{mp}-{xp}"
        if rk in used_ranges:
            continue
        used_ranges.add(rk)
        parts = [chunk_store.texts[wi] for wi in sorted(window)]
        merged.append(("\n\n".join(parts), source, mp, xp))
        if len(merged) >= top_k:
            break

    # Шаг 2: единый заземлённый ответ по нумерованным фрагментам
    fragments = [
        Fragment(
            text=text[:1200],
            doc_title=source,
            page_num=mp,
            section=f"стр. {mp}-{xp}",
        )
        for text, source, mp, xp in merged
    ]
    grounded = answer_grounded(query, fragments)

    sources_text = used_sources(fragments, grounded.used_citation_ids)
    retrieved_text = "\n\n".join(
        f"[{i}] {fragment.doc_title}, стр. {fragment.page_num}\n{fragment.text}"
        for i, fragment in enumerate(fragments, 1)
    )
    result_text = (
        f"=== ОТВЕТ ===\n\n{grounded.text}\n\n"
        f"=== ИСТОЧНИКИ ===\n{sources_text}\n\n"
        f"=== НАЙДЕННЫЕ ФРАГМЕНТЫ ===\n{retrieved_text[:3000]}"
    )
    return [(result_text, 999.0, 0, 0, 0)]
