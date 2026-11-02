#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv

from hipporag import HippoRAG

# Загрузка переменных окружения
load_dotenv()

# Инициализация HippoRAG
hipporag = HippoRAG(
    save_dir="outputs/astrology",
    llm_model_name=os.getenv("LLM_NAME", "gemma-4-e4b"),
    embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    llm_base_url=os.getenv("LLM_BASE_URL", "http://172.18.0.1:1234/v1"),
)

# Подготовка к поиску
if not hipporag.ready_to_retrieve:
    hipporag.prepare_retrieval_objects()

# Тестовый запрос
query = "Овен"
print(f"Поиск по запросу: '{query}'")
print("=" * 80)

# Получаем эмбеддинги запроса
hipporag.get_query_embeddings([query])

# Получаем оценки фактов
query_fact_scores = hipporag.get_fact_scores(query)

print(f"\nНайдено фактов с оценками: {len(query_fact_scores)}")
print("=" * 80)

# Показываем топ-10 фактов
top_indices = sorted(
    range(len(query_fact_scores)), key=lambda i: query_fact_scores[i], reverse=True
)[:10]

for i, idx in enumerate(top_indices, 1):
    fact_key = hipporag.fact_node_keys[idx]
    score = query_fact_scores[idx]
    print(f"\n--- Факт {i} (score: {score:.4f}) ---")
    print(f"Ключ: {fact_key}")

    # Попробуем найти этот факт в графе
    try:
        node_idx = hipporag.graph.vs.find(name=fact_key).index
        neighbors = hipporag.graph.neighbors(node_idx)
        print(f"Соседей в графе: {len(neighbors)}")

        # Показываем сам факт (тройку)
        if "|" in fact_key:
            parts = fact_key.split("|")
            if len(parts) == 3:
                print(f"Тройка: [{parts[0]}] -> [{parts[1]}] -> [{parts[2]}]")
    except Exception:
        print(f"Узел не найден в графе")
