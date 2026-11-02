"""Режимы sentence_dat — перенос из main.py (шаг 6, AST-извлечение из git HEAD)."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .. import config
from ..config import LLM_BASE_URL
from ..primitives import _get_chunk_meta
from ..query import _expand_query, split_query_terms
from ..signs import _chunk_sign_score, _keyword_boost
from ..textutils import _tokenize
from .bm25_hybrid import search_bm25_hybrid_ru
from .small_to_big import search_small_to_big_ru

BIBLIO_QUERY_PATTERNS = (
    r"как(?:ие|ую|их|ой|ого)\s+книг",
    r"названи[еяй]\s+книг",
    r"кто\s+(?:автор|написал|писатель)",
    r"что\s+написал",
    r"как(?:ие|ой|ую|их)\s+(?:произведени|работ)",
    r"список\s+(?:книг|произведени)",
    r"библиограф",
    r"издани[еяй]\s+книг",
    r"какая\s+книга",
)


def _is_bibliographic_query(query):
    """Вопрос про книги/автора/издания, а не про содержание (P1-7)."""
    ql = query.lower()
    return any(re.search(pattern, ql) for pattern in BIBLIO_QUERY_PATTERNS)


def search_sentence_level(hipporag, query, top_k=10):
    if _is_bibliographic_query(query):
        # Название книги — не предложение: «Психологический фэншуй: Как
        # прокачать «плохой» фэншуй» не попадает в sentence-level выдачу.
        # Для библиографических вопросов уходим на chunk-level поиск (P1-7).
        print("  Библиографический вопрос → chunk-level (BM25+RU-CE)", flush=True)
        return search_bm25_hybrid_ru(hipporag, query, top_k=top_k)
    import re as _re

    chunk_store = hipporag.chunk_embedding_store
    terms, q_lower = split_query_terms(query)
    exp = _expand_query(q_lower, terms)
    q_tokens = list(set(_tokenize(query) + exp))
    if not hasattr(search_sentence_level, "sentences_cache"):
        sents = []
        for idx, text in enumerate(chunk_store.texts):
            for s in _re.split(r"(?<=[.!?])\s+", text):
                s = s.strip()
                if len(s) > 15:
                    sents.append((s.lower(), s, idx))
        search_sentence_level.sentences_cache = sents
        print(f"  Предложений: {len(sents)}", flush=True)
    sentences = search_sentence_level.sentences_cache
    sent_tokens = [_tokenize(s[0]) for s in sentences]
    sent_bm25 = BM25Okapi(sent_tokens)
    sent_scores = sent_bm25.get_scores(q_tokens)
    sent_rank = np.argsort(sent_scores)[::-1]
    seen_chunks = set()
    merged = []
    for si in sent_rank:
        if sent_scores[si] == 0:
            continue
        sent_lower, sent_orig, chunk_idx = sentences[si]
        if chunk_idx in seen_chunks:
            continue
        seen_chunks.add(chunk_idx)
        source, page = _get_chunk_meta(chunk_store, chunk_idx)
        full_text = chunk_store.texts[chunk_idx]
        boost = _keyword_boost(full_text.lower(), q_lower, terms)
        sign_score = _chunk_sign_score(chunk_store, chunk_idx, q_lower)
        boost += sign_score
        final_score = float(sent_scores[si]) * (1.0 + boost)
        display_text = (
            full_text.replace(sent_orig, f"\u2192{sent_orig}\u2190", 1)
            if sent_orig in full_text
            else full_text
        )
        merged.append((display_text, final_score, source or "", page, 0, boost))
        if len(merged) >= top_k:
            break
    if not merged:
        return search_small_to_big_ru(hipporag, query, top_k=min(5, top_k))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:top_k]


def _dat_evaluate(query, sent_top1, dense_top1):
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    if not sent_top1 or not dense_top1:
        return 3, 3
    prompt_text = (
        "Ты — оценщик качества поиска. Оцени два результата поиска по шкале от 0 до 5.\n\n"
        "Критерии:\n"
        "5 — документ напрямую отвечает на вопрос\n"
        "3-4 — концептуально близок, содержит частичный ответ или связанные сущности\n"
        "1-2 — слабо связан, вводит в заблуждение\n"
        "0 — полностью нерелевантен\n\n"
        f"Вопрос: {query}\n\n"
        f"Результат 1 (поиск по предложениям):\n{sent_top1[:500]}\n\n"
        f"Результат 2 (семантический поиск):\n{dense_top1[:500]}\n\n"
        "Верни два числа через пробел: оценка_1 оценка_2.\nПример: 4 3"
    )
    for attempt in range(2):
        try:
            r = httpx.post(
                f"{api_url}/v1/chat/completions",
                json={
                    "model": config.LLM_NAME,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "max_tokens": 50,
                    "temperature": 0.1,
                },
                timeout=300,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"].get("content", "").strip()
                nums = re.findall(r"\d+", content)
                if len(nums) >= 2:
                    return min(int(nums[0]), 5), min(int(nums[1]), 5)
                if attempt == 0:
                    prompt_text += "\n\nОтветь ТОЛЬКО двумя числами через пробел."
                    continue
        except:
            if attempt == 0:
                continue
    return 3, 3


def _dat_compute_alpha(score_sentence, score_dense):
    if score_sentence == 0 and score_dense == 0:
        return 0.5
    if score_dense == 5 and score_sentence != 5:
        return 1.0
    if score_sentence == 5 and score_dense != 5:
        return 0.0
    denom = score_dense + score_sentence
    if denom == 0:
        return 0.5
    return round(score_dense / denom, 1)


def _dat_combine_results(sentence_results, dense_results, alpha, top_k=15):
    combined = {}
    s_scores = [r[1] for r in sentence_results] if sentence_results else [0]
    s_max, s_min = max(s_scores), min(s_scores)
    s_range = s_max - s_min if s_max > s_min else 1
    d_scores = [r[1] for r in dense_results] if dense_results else [0]
    d_max, d_min = max(d_scores), min(d_scores)
    d_range = d_max - d_min if d_max > d_min else 1
    for r in sentence_results:
        text, score = r[0], r[1]
        src = r[2] if len(r) > 2 else ""
        pg = r[3] if len(r) > 3 else 0
        norm = (score - s_min) / s_range
        if str(text)[:200].strip() in combined:
            combined[str(text)[:200].strip()]["s_norm"] = norm
        else:
            combined[str(text)[:200].strip()] = {
                "text": text,
                "source": src,
                "page": pg,
                "s_norm": norm,
                "d_norm": 0.0,
            }
    for r in dense_results:
        text, score = str(r[0]) if not isinstance(r[0], str) else r[0], r[1]
        src = r[2] if len(r) > 2 else ""
        pg = r[3] if len(r) > 3 else 0
        norm = (score - d_min) / d_range
        k = str(text)[:200].strip()
        if k in combined:
            combined[k]["d_norm"] = norm
        else:
            combined[k] = {"text": text, "source": src, "page": pg, "s_norm": 0.0, "d_norm": norm}
    for v in combined.values():
        v["final"] = alpha * v["d_norm"] + (1 - alpha) * v["s_norm"]
    sorted_items = sorted(combined.values(), key=lambda x: x["final"], reverse=True)
    return [
        (v["text"], v["final"], v["source"], v["page"], 0, v["final"]) for v in sorted_items[:top_k]
    ]


def search_dat_sentence_llm(hipporag, query, top_k=10):
    import time as _time

    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    t0 = _time.time()
    print("  DAT Sentence-level + LLM", flush=True)
    sentence_results = search_sentence_level(hipporag, query, top_k=30)
    t1 = _time.time()
    print(f"  Sentence-level: {len(sentence_results)} рез. ({t1-t0:.1f}с)", flush=True)
    dense_results = search_small_to_big_ru(hipporag, query, top_k=30)
    t2 = _time.time()
    print(f"  Dense: {len(dense_results)} рез. ({t2-t1:.1f}с)", flush=True)
    sent_top1_text = (
        sentence_results[0][0][:500] if sentence_results and len(sentence_results[0]) > 0 else ""
    )
    dense_top1_text = (
        dense_results[0][0][:500] if dense_results and len(dense_results[0]) > 0 else ""
    )
    s_score, d_score = _dat_evaluate(query, sent_top1_text, dense_top1_text)
    alpha = _dat_compute_alpha(s_score, d_score)
    t3 = _time.time()
    print(
        f"  DAT: sentence={s_score}/5, dense={d_score}/5 -> alpha={alpha} ({t3-t2:.1f}с)",
        flush=True,
    )
    combined = _dat_combine_results(sentence_results, dense_results, alpha, top_k=15)
    if not combined:
        combined = sentence_results[:top_k]
    t4 = _time.time()
    print(f"  Combined: {len(combined)} рез. ({t4-t3:.1f}с)", flush=True)
    chunks_text = ""
    for i, r in enumerate(combined[:8], 1):
        text = r[0] if isinstance(r, tuple) and len(r) > 0 else str(r)
        src = r[2] if isinstance(r, tuple) and len(r) > 2 else ""
        pg = r[3] if isinstance(r, tuple) and len(r) > 3 else 0
        chunks_text += f"\n[Источник {i} - {src} стр.{pg}]\n{text[:1500]}\n"
    answer = None
    prompt_base = {
        "model": config.LLM_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты \u2014 эксперт по астрологии и фэншуй. "
                    "Ответь на вопрос, используя ТОЛЬКО текст источников.\n\n"
                    "В тексте знаки зодиака сопоставляются с зонами квартиры через тире: "
                    "\u00abЗНАК \u2014 ЗОНА\u00bb. Например: \u00abОвен \u2014 прихожая, ... \u2014 плита\u00bb.\n\n"
                    "Таблица планет-управителей:\n"
                    "Овен \u2192 Марс, Телец \u2192 Венера, Близнецы \u2192 Меркурий, Рак \u2192 Луна, "
                    "Лев \u2192 Солнце, Дева \u2192 Меркурий, Весы \u2192 Венера, "
                    "Скорпион \u2192 Плутон/Марс, Стрелец \u2192 Юпитер, "
                    "Козерог \u2192 Сатурн, Водолей \u2192 Уран, Рыбы \u2192 Нептун\n\n"
                    "Алгоритм:\n"
                    "1. Найди \u00abЗНАК \u2014 ... ЗОНА\u00bb в источниках\n"
                    "2. Определи планету-управителя знака\n"
                    "3. Вывод: планета отвечает за зону\n"
                    "4. Пример: \u00abОВЕН \u2014 плита\u00bb + Овен\u2192Марс \u2192 Марс отвечает за плиту\n"
                    "5. Алгоритм знак→зона применяй ТОЛЬКО если вопрос про знак или зону. "
                    "На остальные вопросы (книги, автор, издание, понятие) отвечай по "
                    "источникам дословно — например, перечисли названия книг, если они там есть\n"
                    "6. Если нужной информации в источниках действительно нет — напиши "
                    "\u00abнет данных\u00bb"
                ),
            },
            {"role": "user", "content": f"Вопрос: {query}\n\nИсточники:\n{chunks_text}"},
        ],
        "max_tokens": 2048,
        "temperature": 0.15,
    }
    prompts_to_try = [
        prompt_base,
        {
            "model": config.LLM_NAME,
            "messages": [
                {"role": "user", "content": f"Вопрос: {query}\n\nИсточники:\n{chunks_text}"}
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        },
        {
            "model": config.LLM_NAME,
            "messages": [{"role": "user", "content": f"{query}\n\n{chunks_text[:500]}"}],
            "max_tokens": 512,
            "temperature": 0.1,
        },
    ]
    for attempt, prompt in enumerate(prompts_to_try):
        try:
            resp = httpx.post(f"{api_url}/v1/chat/completions", json=prompt, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                finish = data["choices"][0].get("finish_reason", "")
                content = (msg.get("content") or "").strip()
                if content:
                    answer = content
                    break
                rc = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
                if rc:
                    m = re.search(r"(?:Ответ|Answer)\s*:?\s*(.+)", rc, re.IGNORECASE | re.DOTALL)
                    if m:
                        answer = m.group(1).strip()
                        break
                    m = re.search(r"So the answer is:\s*(.+)", rc, re.IGNORECASE)
                    if m:
                        answer = m.group(1).strip()
                        break
                    lines = [l.strip() for l in rc.split("\n") if l.strip()]
                    if lines:
                        answer = lines[-1].strip("\"'").replace("**", "").strip()
                        break
                print(f"  LLM пусто: попытка {attempt+1}, finish={finish}", flush=True)
        except Exception as e:
            print(f"  LLM error (попытка {attempt+1}): {e}", flush=True)

    t5 = _time.time()
    print(f"  LLM ответ: {len(answer or '')} симв. ({t5-t4:.1f}с)", flush=True)
    print(f"  Всего: {t5-t0:.1f}с", flush=True)

    if not answer:
        # LLM-free fallback: извлекаем "ЗНАК — ЗОНА" из результатов и строим ответ
        SIGN_PLANET = {
            "овен": "Марс",
            "телец": "Венера",
            "близнецы": "Меркурий",
            "рак": "Луна",
            "лев": "Солнце",
            "дева": "Меркурий",
            "весы": "Венера",
            "скорпион": "Плутон",
            "стрелец": "Юпитер",
            "козерог": "Сатурн",
            "водолей": "Уран",
            "рыбы": "Нептун",
        }
        best_match = None
        # Берём sentence-level результаты (они точнее по смыслу)
        for r in sentence_results:
            txt = (r[0] if isinstance(r, tuple) and len(r) > 0 else str(r)).lower()
            for sign, planet in SIGN_PLANET.items():
                m = re.search(rf"{sign}\s*—\s*[^.]{{0,200}}?плит", txt)
                if m:
                    start = max(0, m.start() - 20)
                    ctx = txt[start : m.end() + 30]
                    best_match = (sign.capitalize(), planet, ctx)
                    break
            if best_match:
                break
        if best_match:
            answer = f"По данным источников, за плиту отвечает планета {best_match[1]} (через знак {best_match[0]})."
            print(f"  LLM-free fallback: {best_match[0]} \u2192 {best_match[1]}", flush=True)
        else:
            return combined[:top_k]

    result_text = (
        f"=== ОТВЕТ ===\n\n{answer}\n\n"
        f"=== МЕТОД ===\n"
        f"DAT Sentence-level + LLM | alpha={alpha} (sentence={s_score}/5, dense={d_score}/5)\n"
        f"Время: {t5-t0:.1f}с\n\n"
        f"=== ИСТОЧНИКИ ===\n{chunks_text[:3000]}"
    )
    return [(result_text, 999.0, 0, 0, 0)]
