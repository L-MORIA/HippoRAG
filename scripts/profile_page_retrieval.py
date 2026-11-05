#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Профиль стадий page-level поиска (H1-6): где идёт время.

Служебный скрипт; в package data не входит.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from astro_app.modes.bm25_hybrid import search_bm25_hybrid_ru_pages  # noqa: E402
from astro_app.page_retrieval import (  # noqa: E402
    _build_page_index,
    _chunk_dense_scores,
    _ranks,
)
from astro_app.query import _expand_query, split_query_terms  # noqa: E402
from astro_app.rerankers import _rerank_with_crossencoder_ru  # noqa: E402
from astro_app.signs import _keyword_boost  # noqa: E402
from astro_app.textutils import _tokenize  # noqa: E402

QUERY = "Какое направление Шен-ци у числа гуа 1?"
RRF_K = 60


def timed(label, fn):
    started = time.perf_counter()
    result = fn()
    print(f"  {label}: {(time.perf_counter() - started) * 1000:.0f} мс")
    return result


def main() -> int:
    from astro_app.config import load_hipporag

    engine = timed("load_hipporag", lambda: load_hipporag())
    store = engine.chunk_embedding_store
    texts = store.texts

    for attempt in (1, 2):
        print(f"--- прогон {attempt} ---")
        index = timed("_build_page_index", lambda: _build_page_index(store))
        print(f"    страниц: {len(index.pages)}")

        terms, q_lower = split_query_terms(QUERY)
        expanded = _expand_query(q_lower, terms)
        tokens = list(set(_tokenize(QUERY) + [t for e in expanded for t in _tokenize(e)]))
        timed("bm25.get_scores", lambda: index.bm25.get_scores(tokens))
        dense = timed("_chunk_dense_scores", lambda: _chunk_dense_scores(engine, store, QUERY))
        timed("boost по страницам", lambda: [_keyword_boost(p.text.lower(), q_lower, terms) for p in index.pages])

        bm25_scores = [float(s) for s in index.bm25.get_scores(tokens)]
        bm25_ranks = _ranks(bm25_scores)
        dense_ranks = _ranks(list(dense) if dense is not None else None)
        scored = []
        for pos, page in enumerate(index.pages):
            rrf = 1.0 / (RRF_K + bm25_ranks[pos]) + 1.0 / (RRF_K + (dense_ranks[pos] if dense_ranks else 500))
            scored.append((rrf, pos))
        scored.sort(reverse=True)
        candidates = [index.pages[pos] for _, pos in scored[:50]]
        rerank_input = []
        for page in candidates:
            chunk_text = texts[page.chunk_indices[0]]
            window = page.text[:512]
            rerank_input.append((window, 0.0, 0.0, 0.0, 0.0))
        reranked = timed(
            f"_rerank_with_crossencoder_ru ({len(rerank_input)} страниц)",
            lambda: _rerank_with_crossencoder_ru(QUERY, rerank_input),
        )

    timed("полный search_pages_hybrid (warm)", lambda: search_bm25_hybrid_ru_pages(engine, QUERY, top_k=5))
    timed("полный search_pages_hybrid (warm, 2)", lambda: search_bm25_hybrid_ru_pages(engine, QUERY, top_k=5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
