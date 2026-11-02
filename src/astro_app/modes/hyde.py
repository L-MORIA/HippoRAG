"""Режимы hyde — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import numpy as np

from .. import config
from ..config import LLM_BASE_URL
from ..primitives import _build_bm25
from ..query import split_query_terms
from ..rerankers import _get_crossencoder_ru
from ..signs import _chunk_sign_score, _keyword_boost
from ..textutils import _tokenize


def search_hyde(hipporag, query, top_k=10, candidate_pool=200):
    """
    HyDE (Hypothetical Document Embedding).
    1. LLM генерирует гипотетический документ (стиль книги Астрофэншуй)
    2. Эмбеддинг запроса + эмбеддинг гипотетического документа
    3. BM25 отбирает кандидатов
    4. Multi-vector scoring + sign dominance
    5. Русский Cross-encoder reranking
    """
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL

    # Шаг 1: генерация гипотетического документа
    hyde_text = None
    try:
        r_health = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r_health.status_code == 200:
            hyde_prompt = {
                "model": config.LLM_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты — автор книги «Астрофэншуй квартиры». Напиши отрывок из книги, "
                            "который отвечает на вопрос читателя. Используй стиль книги: "
                            "астрологические термины, фэншуй, знаки зодиака, планеты, зоны квартиры. "
                            "Напиши 3-5 предложений как в настоящей книге."
                        ),
                    },
                    {"role": "user", "content": f"Вопрос читателя: {query}"},
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
            }
            r = httpx.post(f"{api_url}/v1/chat/completions", json=hyde_prompt, timeout=300)
            if r.status_code == 200:
                data = r.json()
                msg = data["choices"][0]["message"]
                hyde_text = (msg.get("content") or "").strip()
                if not hyde_text and msg.get("reasoning_content"):
                    rc = msg["reasoning_content"]
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    hyde_text = lines[-1].strip("\"'").replace("**", "") if lines else None
                if hyde_text and len(hyde_text) > 20:
                    print(f"  📄 Гипотетический документ сгенерирован ({len(hyde_text)} симв.)")
                else:
                    hyde_text = None
    except Exception:
        pass

    if not hyde_text:
        print("  ⚠️ HyDE: LLM недоступен, работаю без гипотетического документа")
        hyde_text = query

    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []

    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]

    # Эмбеддинг запроса и гипотетического документа
    q_emb = hipporag.embedding_model.encode([query])
    hyde_emb = hipporag.embedding_model.encode([hyde_text])
    all_emb = np.array(chunk_store.embeddings)

    query_scores = np.dot(all_emb, q_emb.T).squeeze()
    hyde_scores = np.dot(all_emb, hyde_emb.T).squeeze()

    # BM25 candidates
    q_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(q_tokens)
    candidate_idx = np.argsort(bm25_scores)[::-1][:candidate_pool]

    scored = []
    for idx in candidate_idx:
        if idx >= n_chunks:
            continue
        text = chunk_store.texts[idx]
        tl = text.lower()
        qs = float(query_scores[idx])
        hs = float(hyde_scores[idx])
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        # Комбинируем: query (40%) + hyde (40%) + keyword+sign (20%)
        final = qs * 0.4 + hs * 0.4 + (qs + hs) * 0.5 * boost * 0.2
        scored.append((text, final, qs, hs, boost))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Rerank с русским CE
    top_50 = scored[:50]
    rest = scored[50:]
    try:
        ce = _get_crossencoder_ru()
        pairs = [(query, c[0][:512]) for c in top_50]
        xc_scores = ce.predict(pairs, show_progress_bar=False)
        reranked = []
        for i, c in enumerate(top_50):
            xc = float(xc_scores[i])
            text = c[0]
            old = c[1]
            boost = c[4]
            final = old * 0.6 + xc * 0.4
            reranked.append((text, final, xc, c[2], boost))
        reranked.sort(key=lambda x: x[1], reverse=True)
        scored = reranked + rest
    except Exception:
        pass

    return scored[:top_k], hyde_text
