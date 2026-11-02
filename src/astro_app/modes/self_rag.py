"""Режимы self_rag — перенос из main.py (шаг 6, байт-в-байт)."""

import httpx
import numpy as np

from ..config import LLM_BASE_URL, LLM_NAME
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

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
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

    # Шаг 2: собираем текст чанков для LLM
    chunks_text = ""
    for i, (text, source, mp, xp) in enumerate(merged, 1):
        chunks_text += f"\n[Источник {i} — {source} стр.{mp}-{xp}]\n{text[:1200]}\n"

    # Шаг 3: LLM
    answer = None
    try:
        r_health = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r_health.status_code == 200:
            prompt = {
                "model": LLM_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ответь на вопрос читателя, используя ТОЛЬКО текст источников.\n\n"
                            "АЛГОРИТМ:\n"
                            "1. Вопрос может быть ЛЮБОЙ темы: автор, книга, издание, понятие, "
                            "знак зодиака, зона дома. Определи тип вопроса и найди прямые ответы "
                            "в источниках.\n"
                            "2. Для вопросов вида «какая планета отвечает за зону» — "
                            "ищи «знаку соответствует зона» + «управитель знака — планета»\n"
                            "3. Для остальных вопросов — приведи прямое содержание найденного: "
                            "названия, формулировки, факты, дословно как в источниках\n"
                            "4. Если в источниках есть нужная информация, но она сформулирована "
                            "ИНАЧЕ, чем вопрос (например «Психологический фэншуй» вместо "
                            "«название книги») — приведи её как ответ. Не считай это отсутствием "
                            "информации.\n"
                            "5. Для вопросов «перечисли всё» — выведи полный список\n"
                            "6. Ссылайся на источники: [Источник 1]\n"
                            "7. «нет данных» пиши ТОЛЬКО если в источниках действительно нет "
                            "ничего по теме вопроса. Отсутствие ГОТОВОЙ формулировки "
                            "не равно отсутствию информации: если в источниках есть нужные "
                            "слова, факты и соответствия, из которых ответ выводится — "
                            "составь и дай ответ, а не отказ"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Вопрос: {query}\n\n"
                            f"Источники:\n{chunks_text}\n\n"
                            f"Ответь на основе источников."
                        ),
                    },
                ],
                "max_tokens": 2048,
                "temperature": 0.15,
            }
            resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                answer = (msg.get("content") or "").strip()
                if not answer and msg.get("reasoning_content"):
                    rc = msg["reasoning_content"]
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    answer = lines[-1].strip("\"'").replace("**", "") if lines else answer
    except Exception:
        pass

    if not answer:
        return [(scored[0][0], scored[0][1], "", 0, 0, 0)]

    result_text = f"=== ОТВЕТ ===\n\n{answer}\n\n=== ИСТОЧНИКИ ===\n{chunks_text[:3000]}"
    return [(result_text, 999.0, 0, 0, 0)]
