#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система тестирования и оценки качества поиска HippoRAG.
Использует gemma-4-e4b как независимого арбитра для оценки ответов.
"""

import os, sys, json, re, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import httpx
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

LLM_NAME = os.getenv('LLM_NAME', 'gemma-4-e4b')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
API_URL = LLM_BASE_URL.replace('/v1', '') if '/v1' in LLM_BASE_URL else LLM_BASE_URL

SAVE_DIR = 'outputs/astrology_test'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# ═══════════════════════════════════════════════════════════
# 1. ТЕСТОВЫЙ НАБОР ЗАПРОСОВ
# ═══════════════════════════════════════════════════════════

TEST_QUERIES = [
    {
        "id": "марс_плита",
        "query": "Какая планета отвечает за плиту?",
        "expected_terms": ["марс", "плита"],
        "description": "Связь планеты и зоны"
    },
    {
        "id": "овен_ремонт",
        "query": "С чего начинать ремонт Овну?",
        "expected_terms": ["овен", "ремонт"],
        "description": "Ремонт для знака"
    },
    {
        "id": "телец_характер",
        "query": "Какие черты характера у Тельца в интерьере?",
        "expected_terms": ["телец", "интерьер"],
        "description": "Характеристика знака"
    },
    {
        "id": "фэншуй_прихожая",
        "query": "Какое значение имеет прихожая в фэншуй?",
        "expected_terms": ["прихожая", "фэншуй"],
        "description": "Зона фэншуй"
    },
    {
        "id": "число_гуа",
        "query": "число гуа 6",
        "expected_terms": ["гуа", "6"],
        "description": "Специфический термин"
    },
]

# ═══════════════════════════════════════════════════════════
# 2. ОЦЕНКА ЧЕРЕЗ LLM (gemma-4-e4b)
# ═══════════════════════════════════════════════════════════

def llm_available():
    try:
        r = httpx.get(f"{API_URL}/v1/models", timeout=3)
        return r.status_code == 200
    except:
        return False


def ask_llm(system, user, max_tokens=512, timeout=60):
    payload = {
        "model": LLM_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    try:
        r = httpx.post(f"{API_URL}/v1/chat/completions", json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content and msg.get("reasoning_content"):
                rc = msg["reasoning_content"]
                lines = [l.strip() for l in rc.split('\n') if l.strip()]
                content = lines[-1].strip('"\'').replace('**', '') if lines else ""
            return content
    except:
        pass
    return ""


def evaluate_result(query, result_text, expected_terms):
    """Оценить результат поиска через LLM-арбитра."""
    if not result_text or len(result_text) < 20:
        return {"score": 0, "verdict": "Нет результата", "details": ""}

    # Автоматическая оценка по наличию ожидаемых терминов
    tl = result_text.lower()
    term_score = sum(1 for t in expected_terms if t in tl) / max(len(expected_terms), 1)
    term_hits = [t for t in expected_terms if t in tl]
    term_miss = [t for t in expected_terms if t not in tl]

    # Оценка через LLM
    llm_score = 0.0
    llm_verdict = ""
    if llm_available():
        eval_prompt = (
            "Ты — эксперт по оценке качества поиска. Оцени ответ на запрос по шкале 0-10.\n\n"
            "Критерии:\n"
            "- Релевантность: ответ по теме запроса (0-4)\n"
            "- Полнота: охватывает ли ответ ключевые аспекты (0-3)\n"
            "- Полезность: можно ли использовать ответ на практике (0-3)\n\n"
            "Формат ответа: строго одно число от 0 до 10 и краткий комментарий."
        )
        eval_result = ask_llm(eval_prompt,
            f"Запрос: {query}\n\nОтвет системы:\n{result_text[:2000]}",
            max_tokens=100)
        if eval_result:
            numbers = re.findall(r'\d+', eval_result)
            if numbers:
                llm_score = min(float(numbers[0]), 10.0) / 10.0
            llm_verdict = eval_result[:200]

    final_score = max(term_score, llm_score) * 0.6 + term_score * 0.4

    return {
        "score": round(final_score, 3),
        "term_score": round(term_score, 3),
        "llm_score": round(llm_score, 3),
        "term_hits": term_hits,
        "term_miss": term_miss,
        "llm_verdict": llm_verdict[:150] if llm_verdict else "",
    }


# ═══════════════════════════════════════════════════════════
# 3. ЗАПУСК ТЕСТОВ
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("ОБЪЕКТИВНОЕ ТЕСТИРОВАНИЕ ВСЕХ МЕТОДОВ (1-16, 20-23)")
    print("=" * 80)

    config = BaseConfig(
        save_dir=SAVE_DIR,
        llm_name=LLM_NAME,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        linking_top_k=100, retrieval_top_k=200,
        passage_node_weight=0.05, embedding_batch_size=8,
    )
    hipporag = HippoRAG(global_config=config)

    from main import (
        search_multi_vector, search_hipporag_graph,
        search_bm25_hybrid, search_small_to_big,
        search_bm25_hybrid_ru, search_small_to_big_ru,
        search_hyde, search_filtered,
        extract_sign_full, search_rrf_context,
        search_e5_rerank, search_mxbai, search_colbert,
        search_self_rag, extract_sign_summary, extract_sign_polished,
        search_sentence_level,
        search_dat_sentence_llm,
        search_semantic_filter,
        search_qa_cache
    )

    METHODS = [
        (1, "Multi-vector+LLM",     lambda q: search_multi_vector(hipporag, q)[0]),
        (2, "Графовый PPR",         lambda q: search_hipporag_graph(hipporag, q)),
        (3, "BM25+CE-en",           lambda q: search_bm25_hybrid(hipporag, q)),
        (4, "Small→Big+CE-en",      lambda q: search_small_to_big(hipporag, q)),
        (5, "BM25+RU-CE",           lambda q: search_bm25_hybrid_ru(hipporag, q)),
        (6, "Small→Big+RU-CE",      lambda q: search_small_to_big_ru(hipporag, q)),
        (7, "HyDE",                 lambda q: search_hyde(hipporag, q)[0]),
        (8, "Фильтр+RU-CE",         lambda q: search_filtered(hipporag, q)),
        (9, "Весь знак (DOCX)",     lambda q: extract_sign_full(hipporag, q)),
        (10, "Резюме LLM",          lambda q: extract_sign_summary(hipporag, q)),
        (11, "Обработка LLM",       lambda q: extract_sign_polished(hipporag, q)),
        (12, "Self-RAG ⭐",         lambda q: search_self_rag(hipporag, q)),
        (13, "RRF+контекст",        lambda q: search_rrf_context(hipporag, q)),
        (14, "E5 sentence ⭐",      lambda q: search_e5_rerank(hipporag, q)),
        (15, "RRF+sign",            lambda q: search_mxbai(hipporag, q)),
        (16, "ColBERT-v2",          lambda q: search_colbert(hipporag, q)),
        (20, "Sentence-level ⭐",   lambda q: search_sentence_level(hipporag, q)),
        (21, "DAT+LLM ⭐",         lambda q: search_dat_sentence_llm(hipporag, q)),
        (22, "SemFilter ⭐",      lambda q: search_semantic_filter(hipporag, q)),
        (23, "QACache ⭐",        lambda q: search_qa_cache(hipporag, q)),
    ]

    all_scores = {}  # method_name -> {query_id -> score}

    for test in TEST_QUERIES:
        q = test["query"]
        qid = test["id"]
        expected = test["expected_terms"]
        print(f"\n{'='*60}")
        print(f"Запрос: {q}")
        print(f"{'='*60}")

        for mid, mname, mfunc in METHODS:
            try:
                res = mfunc(q)
                if not res:
                    result_text = ""
                elif isinstance(res, list) and len(res) > 0:
                    if len(res[0]) >= 1:
                        result_text = res[0][0][:500]
                    else:
                        result_text = str(res[0])
                else:
                    result_text = str(res)[:500]

                tl = result_text.lower() if result_text else ""
                term_score = sum(1 for t in expected if t in tl) / max(len(expected), 1) if result_text else 0
                term_hits = [t for t in expected if t in tl]

                if qid not in all_scores:
                    all_scores[qid] = {}
                all_scores[qid][mname] = round(term_score, 2)
                print(f"  [{mid:2d}] {mname:<20} → {term_score:.0%} {term_hits}", flush=True)

            except Exception as e:
                if qid not in all_scores:
                    all_scores[qid] = {}
                all_scores[qid][mname] = 0
                print(f"  [{mid:2d}] {mname:<20} → ERROR: {str(e)[:50]}", flush=True)

    # ═══════════════════════════════════════════════════════
    # ИТОГИ
    # ═══════════════════════════════════════════════════════

    print("\n\n" + "=" * 80)
    print("ИТОГИ: СРЕДНИЙ СКОР ПО ВСЕМ МЕТОДАМ")
    print("=" * 80)

    # Средний скор для каждого метода
    method_avgs = {}
    for mname in [m[1] for m in METHODS]:
        scores = []
        for qid in all_scores:
            if mname in all_scores[qid]:
                scores.append(all_scores[qid][mname])
        if scores:
            method_avgs[mname] = sum(scores) / len(scores)

    # Таблица
    print(f"\n{'Метод':<25} {'Средний скор':<15}")
    print('-' * 42)
    for mname, avg in sorted(method_avgs.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(avg * 20) + '░' * (20 - int(avg * 20))
        print(f"{mname:<25} {avg:.0%}  {bar}")

    print("\nТоп-3 метода:")
    top3 = sorted(method_avgs.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (mname, avg) in enumerate(top3, 1):
        print(f"  {i}. {mname} — {avg:.0%}")

    # Сохраняем
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries": [t["id"] for t in TEST_QUERIES],
        "method_averages": {k: round(v, 3) for k, v in
                           sorted(method_avgs.items(), key=lambda x: x[1], reverse=True)},
        "all_scores": all_scores,
    }
    report_path = out_dir / "test_results.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены: {report_path}")


if __name__ == "__main__":
    main()
