"""Режимы graph — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import numpy as np

from hipporag.utils.misc_utils import min_max_normalize

from .. import config
from ..config import LLM_BASE_URL
from ..query import _expand_query, split_query_terms
from ..signs import _chunk_sign_score, _keyword_boost, _sign_in_query
from .bm25_hybrid import search_bm25_hybrid_ru, search_rrf_context
from .filtered import search_filtered
from .self_rag import search_self_rag
from .small_to_big import search_small_to_big_ru


def search_hipporag_graph(hipporag, query, top_k=10, linking_entities=30, passage_weight=0.3):
    """Графовый поиск: entity linking + PPR на графе."""
    hipporag.prepare_retrieval_objects()

    # Определяем знак зодиака для точного entity linking
    _, q_lower = split_query_terms(query)
    terms, _ = split_query_terms(query)

    q_emb = hipporag.embedding_model.encode([query])
    entity_scores = np.dot(hipporag.entity_embeddings, q_emb.T).squeeze()

    # Sign dominance boost для entity scoring
    sign_in_query = _sign_in_query(q_lower)
    if sign_in_query:
        for i, e_text in enumerate(hipporag.entity_embedding_store.texts):
            if sign_in_query in str(e_text).lower():
                entity_scores[i] *= 1.5

    node_weights = np.zeros(hipporag.graph.vcount())
    sorted_entity_idx = np.argsort(entity_scores)[::-1][:linking_entities]
    for e_idx in sorted_entity_idx:
        vid = hipporag.entity_node_idxs[e_idx]
        node_weights[vid] = entity_scores[e_idx]

    dpr_ids, dpr_scores = hipporag.dense_passage_retrieval(query)
    norm_dpr = min_max_normalize(dpr_scores)
    for i, pid in enumerate(dpr_ids):
        pkey = hipporag.passage_node_keys[pid]
        if pkey in hipporag.node_name_to_vertex_idx:
            vid = hipporag.node_name_to_vertex_idx[pkey]
            node_weights[vid] += norm_dpr[i] * passage_weight

    if sum(node_weights) == 0:
        ppr_ids, ppr_scores = dpr_ids, dpr_scores
    else:
        ppr_ids, ppr_scores = hipporag.run_ppr(node_weights, damping=0.3)

    chunk_store = hipporag.chunk_embedding_store
    scored = []
    for i in range(min(len(ppr_ids), len(dpr_ids))):
        idx = ppr_ids[i]
        text = chunk_store.texts[idx]
        ppr_score = float(ppr_scores[i])
        dpr_idx = np.where(dpr_ids == idx)[0]
        dpr_score = float(dpr_scores[dpr_idx[0]]) if len(dpr_idx) > 0 else 0.0
        tl = text.lower()
        boost = _keyword_boost(tl, q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, idx, q_lower)
        boost += sign_score
        final_score = ppr_score * (1.0 + boost)
        scored.append((text, final_score, ppr_score, dpr_score, boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def search_graphrag_agent(hipporag, query):
    """
    Метод 19: GraphRAG Agent.
    Router → Search → Self-Correction → LLM Answer.
    Анализирует запрос, выбирает лучший метод, проверяет качество,
    при необходимости переформулирует и пробует снова.
    """
    import time as _time

    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    terms, q_lower = split_query_terms(query)
    exp_terms = _expand_query(q_lower, terms)

    def has_any_term(text, term_list):
        tl = text.lower()
        return any(t in tl for t in term_list if len(t) > 2)

    # --- Шаг 1: Router — определяем тип запроса ---
    is_global = any(w in q_lower for w in ["перечисли", "список", "все", "какие", "сколько"])
    is_relation = any(w in q_lower for w in ["связан", "отличается", "общего", "связь"])
    is_multi_hop = len(terms) >= 4 and not is_global

    # --- Шаг 2: Выбираем метод ---
    s2b = lambda q: search_small_to_big_ru(hipporag, q, top_k=8)
    bm25ru = lambda q: search_bm25_hybrid_ru(hipporag, q, top_k=8)
    rrf = lambda q: search_rrf_context(hipporag, q, top_k=8)
    filt = lambda q: search_filtered(hipporag, q, top_k=8)
    srag = lambda q: search_self_rag(hipporag, q)

    methods = []
    if is_global:
        methods = [(rrf, "RRF"), (s2b, "Small→Big")]
    elif is_relation:
        methods = [(filt, "Filter"), (rrf, "RRF")]
    elif is_multi_hop:
        methods = [(srag, "Self-RAG"), (s2b, "Small→Big")]
    else:
        methods = [(bm25ru, "BM25+RU"), (s2b, "Small→Big")]

    # Исправляем search_and_check: передаём lambda без top_k
    def search_and_check2(fn, name, tk=8):
        t0 = _time.time()
        try:
            res = fn(query)
        except:
            res = []
        dt = _time.time() - t0
        if not res:
            return [], 0, dt
        relevant = sum(
            1
            for r in res
            if has_any_term(r[0] if isinstance(r, (list, tuple)) else str(r), exp_terms)
        )
        return res, relevant / max(len(res), 1), dt

    # --- Шаг 3: Каскадный поиск с самокоррекцией ---
    best_results = []
    best_quality = 0
    best_method = ""
    log = []

    for fn, name in methods:
        if best_quality >= 0.8:
            break
        res, quality, dt = search_and_check2(fn, name)
        log.append(f"{name}: quality={quality:.0%} ({dt:.0f}s)")
        print(f"  📊 {log[-1]}", flush=True)
        if res and (not best_results or quality > best_quality):
            best_results = res
            best_quality = quality
            best_method = name

    # Если все методы дали плохой результат — пробуем LLM переформулировку
    if best_quality < 0.3:
        try:
            r = httpx.get(f"{api_url}/v1/models", timeout=3)
            if r.status_code == 200:
                prompt = {
                    "model": config.LLM_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Переформулируй запрос для поиска. Выдели 2-5 ключевых слов.",
                        },
                        {"role": "user", "content": f"Запрос: {query}"},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,
                }
                resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
                if resp.status_code == 200:
                    data = resp.json()
                    msg = data["choices"][0]["message"]
                    new_q = (msg.get("content") or "").strip().strip("\"'")

                    if new_q and 5 < len(new_q) < 100:
                        print(f"  🔄 Переформулировка: '{query}' → '{new_q}'", flush=True)
                        rrf_retry = lambda q: search_rrf_context(hipporag, q, top_k=10)
                        res, quality, dt = search_and_check2(rrf_retry, "RRF(retry)", tk=10)
                        if quality > best_quality:
                            best_results = res
                            best_method = "RRF(retry)"
                            log.append(f"RRF(retry): quality={quality:.0%}")
        except:
            pass

    # Если лучший метод — Self-RAG, используем его ответ напрямую
    if best_method == "Self-RAG" and best_results:
        return best_results[:1]

    # --- Шаг 4: LLM формирует ответ ---
    answer = None
    if best_results:
        # Собираем текст
        chunks_text = ""
        for i, r in enumerate(best_results[:5]):
            doc = r[0] if isinstance(r, (list, tuple)) else str(r)
            src = r[2] if isinstance(r, (list, tuple)) and len(r) > 2 else ""
            src_str = str(src) if src else ""
            chunks_text += f"\n[Источник {i+1} {src_str}]\n{doc[:1000]}\n"

        answer = None
        try:
            r = httpx.get(f"{api_url}/v1/models", timeout=3)
            if r.status_code == 200:
                prompt = {
                    "model": config.LLM_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Ответь на вопрос на основе источников.\n\n"
                                "АЛГОРИТМ:\n"
                                "1. Найди в источниках прямые ответы и соответствия\n"
                                "2. Для вопросов вида «какая планета отвечает за зону» — "
                                "ищи «знаку соответствует зона» + «управитель знака — планета»\n"
                                "3. Для вопросов «перечисли всё» — выведи полный список\n"
                                "4. Для остальных вопросов (автор, книга, понятие) — приведи прямое "
                                "содержание найденного, дословно как в источниках\n"
                                "5. Если нужная информация есть, но сформулирована иначе, чем "
                                "вопрос — приведи её. Отсутствием информации считай только "
                                "полное отсутствие чего-либо по теме.\n"
                                "6. Ссылайся на источники: [Источник 1]\n"
                                "7. «нет данных» пиши ТОЛЬКО если в источниках действительно нет "
                                "ничего по теме вопроса"
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Вопрос: {query}\n\nИсточники:\n{chunks_text}",
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
        except:
            pass

    # Раньше функция заканчивалась здесь без return: маршрутизация, каскадный
    # поиск и ответ LLM выполнялись, а результат выбрасывался — режим 19 всегда
    # возвращал None и показывал «Ничего не найдено».
    if answer:
        return [(answer, 999.0, 0, 0, 0)]
    return best_results[:1]
