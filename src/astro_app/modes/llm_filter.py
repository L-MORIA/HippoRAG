"""Режимы llm_filter — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import numpy as np

from ..config import LLM_BASE_URL, LLM_NAME
from ..primitives import _build_bm25, _get_chunk_meta
from ..query import _expand_query, split_query_terms
from ..signs import _chunk_sign_score, _count_sign, _keyword_boost, _sign_in_query
from ..textutils import _tokenize
from .small_to_big import search_small_to_big_ru


def _filter_by_section(candidate_idx, chunk_store, q_lower, keep_general=0.15):
    """
    Отфильтровать кандидаты: оставить только чанки из раздела
    релевантного знака зодиака + немного общих чанков для контекста.
    """
    sign = _sign_in_query(q_lower)
    if not sign:
        return candidate_idx

    filtered = []
    general = []
    for idx in candidate_idx:
        tl = chunk_store.texts[idx].lower()
        if _count_sign(tl, sign) > 0:
            filtered.append(idx)
        else:
            general.append(idx)

    keep_gen = max(int(len(candidate_idx) * keep_general), 5)
    result = filtered + general[:keep_gen]
    if not filtered:
        return candidate_idx  # если не нашли ничего — возвращаем всё
    return result


def _llm_filter_results(query, results, max_results=5):
    """
    Универсальный LLM-фильтр для результатов любого метода поиска.
    Отправляет топ-10 результатов в LLM, просит отобрать релевантные.
    """
    import json as _json

    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    try:
        r = httpx.get(f"{api_url}/v1/models", timeout=3)
        if r.status_code != 200:
            return results[:max_results]
    except:
        return results[:max_results]

    if not results:
        return results

    # Собираем топ-10 результатов в json для LLM
    candidates = []
    for i, r in enumerate(results[:10]):
        text = r[0][:300] if isinstance(r, (list, tuple)) and len(r) > 0 else str(r)[:300]
        source = r[2] if isinstance(r, (list, tuple)) and len(r) > 2 else f"result_{i}"
        candidates.append({"id": i, "source": str(source), "text": text})

    prompt = {
        "model": LLM_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — фильтр качества поиска. Оцени каждый результат по критериям:\n"
                    "- Содержит ли текст ответ на вопрос?\n"
                    "- Есть ли в тексте ключевые слова из вопроса?\n\n"
                    "Верни JSON-список id релевантных результатов. "
                    'Формат: {"relevant_ids": [0, 2, 5]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {query}\n\n"
                    f"Результаты: {_json.dumps(candidates, ensure_ascii=False)}\n\n"
                    f"Какие id релевантны?"
                ),
            },
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }
    try:
        resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
        if resp.status_code == 200:
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content and msg.get("reasoning_content"):
                rc = msg["reasoning_content"]
                lines = [l.strip() for l in rc.split("\n") if l.strip()]
                content = lines[-1].strip("\"'").replace("**", "") if lines else content
            # Извлекаем JSON
            import re as _re

            json_match = _re.search(r'\{.*"relevant_ids".*\}', content, _re.DOTALL)
            if json_match:
                filtered = _json.loads(json_match.group())
                relevant = filtered.get("relevant_ids", [])
                return [results[i] for i in relevant if i < len(results)]
    except:
        pass
    return results[:max_results]


def search_with_llm_filter(hipporag, query):
    """
    Комбинация: поиск через Small-to-big + RU-CE (метод 6)
    + LLM фильтрация нерелевантных результатов.
    """
    results = search_small_to_big_ru(hipporag, query, top_k=10)
    if not results:
        return []
    print(f"  🤖 LLM фильтрация {len(results)} результатов...", flush=True)
    filtered = _llm_filter_results(query, results, max_results=5)
    print(f"  ✅ После фильтра: {len(filtered)} результатов", flush=True)
    return filtered if filtered else results[:3]


_QWEN_MODEL = None


_FRIDA_MODEL = None


def _get_qwen():
    global _QWEN_MODEL
    if _QWEN_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка Qwen3-Embedding-0.6B...", flush=True)
        _QWEN_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("  ✅ Qwen3 загружен", flush=True)
    return _QWEN_MODEL


def _get_frida():
    global _FRIDA_MODEL
    if _FRIDA_MODEL is None:
        from sentence_transformers import SentenceTransformer

        print("  ⏳ Загрузка FRIDA (0.8B)...", flush=True)
        _FRIDA_MODEL = SentenceTransformer("ai-forever/FRIDA")
        print("  ✅ FRIDA загружена", flush=True)
    return _FRIDA_MODEL


def search_qwen_frida(hipporag, query, top_k=10):
    """
    Метод 18: BM25 + Qwen3-Embedding (dense) + FRIDA (русский reranker) + LLM фильтр.
    """
    chunk_store = hipporag.chunk_embedding_store
    if len(chunk_store.embeddings) == 0:
        return []
    n_chunks = len(chunk_store.texts)
    bm25 = _build_bm25(chunk_store)
    terms, q_lower = split_query_terms(query)
    if not terms:
        terms = [w for w in query.lower().split() if len(w) > 2]
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_candidates = np.argsort(bm25_scores)[::-1][:300]

    # Qwen3 dense embedding (только для топ-50 кандидатов)
    print("  🤖 Qwen3-Embedding...", flush=True)
    qwen = _get_qwen()
    q_emb = qwen.encode([f"search_query: {query}"])[0]
    top50 = bm25_candidates[:50]
    top_texts = [chunk_store.texts[i] for i in top50]
    top_embs = qwen.encode([f"search_document: {t}" for t in top_texts])

    # RRF: BM25 + Qwen3
    bm25_pos = {idx: p for p, idx in enumerate(bm25_candidates)}
    qwen_scores = np.dot(top_embs, q_emb)
    qwen_rank = np.argsort(qwen_scores)[::-1]
    qwen_pos = {top50[i]: p for p, i in enumerate(qwen_rank)}
    k = 60
    scored = []
    for idx in top50:
        rrf = 1 / (k + bm25_pos.get(idx, 1e6)) + 1 / (k + qwen_pos.get(idx, 1e6))
        boost = _keyword_boost(chunk_store.texts[idx].lower(), q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        scored.append((idx, rrf, float(qwen_scores[top50.tolist().index(idx)]), boost))
    for idx in bm25_candidates[50:]:
        scored.append((idx, 1 / (k + bm25_pos.get(idx, 1e6)), 0, 0))
    scored.sort(key=lambda x: x[1], reverse=True)

    # FRIDA reranking топ-20
    print("  🤖 FRIDA reranking...", flush=True)
    frida = _get_frida()
    top20 = scored[:20]
    frida_scores = frida.encode([f"search_query: {query}" for _ in top20])
    frida_docs = frida.encode(
        [f"search_document: {chunk_store.texts[idx][:512]}" for idx, *_ in top20]
    )
    frida_sims = (frida_scores * frida_docs).sum(axis=1)

    for i, (idx, rrf, qs, boost) in enumerate(top20):
        fs = float(frida_sims[i])
        final = fs * 0.5 + rrf * 0.3 + qs * 0.2 * (1.0 + boost)
        top20[i] = (idx, final, fs, boost)
    top20.sort(key=lambda x: x[1], reverse=True)
    all_results = [(chunk_store.texts[idx], float(fs), idx) for idx, final, fs, boost in top20]
    all_results += [(chunk_store.texts[idx], 0, idx) for idx, rrf, qs, boost in scored[20:]]

    # LLM фильтрация
    print("  🤖 LLM фильтрация...", flush=True)
    filtered = _llm_filter_results(query, all_results, max_results=5)

    # Контекстное окно ±1 для отфильтрованных
    used_ranges = set()
    merged = []
    for text, fs, idx in filtered:
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
        ft = "\n\n".join(parts)
        bf = _keyword_boost(ft.lower(), q_lower, terms)
        merged.append((ft, fs, source, mp, xp, bf))
        if len(merged) >= top_k:
            break

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]
