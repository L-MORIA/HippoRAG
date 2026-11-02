"""
Диагностический скрипт для проверки DPR поиска
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():
    print("=" * 80)
    print("ДИАГНОСТИКА DPR")
    print("=" * 80)

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.retrieval_top_k = 10

    print("\nИнициализация HippoRAG...")
    hipporag = HippoRAG(global_config=config)

    print(f"✓ Загружено чанков: {len(hipporag.chunk_embedding_store.hash_ids)}")

    # Подготовка к поиску
    if not hipporag.ready_to_retrieve:
        hipporag.prepare_retrieval_objects()

    # Тестовый запрос
    query = "Какие особенности квартиры для знака Овен?"
    print(f"\n{'='*80}")
    print(f"Запрос: {query}")
    print("=" * 80)

    # Получаем эмбеддинг запроса
    print("\n1. Получение эмбеддинга запроса...")
    from hipporag.prompts.linking import get_query_instruction

    query_embedding = hipporag.embedding_model.batch_encode(
        query, instruction=get_query_instruction("query_to_passage"), norm=True
    )
    print(f"   Размерность эмбеддинга запроса: {query_embedding.shape}")
    print(f"   Норма эмбеддинга: {np.linalg.norm(query_embedding):.4f}")

    # Проверяем passage_embeddings
    print(f"\n2. Проверка passage_embeddings...")
    print(f"   Размерность: {hipporag.passage_embeddings.shape}")
    print(f"   Количество документов: {hipporag.passage_embeddings.shape[0]}")

    # Вычисляем схожесть
    print(f"\n3. Вычисление схожести...")
    query_doc_scores = np.dot(hipporag.passage_embeddings, query_embedding.T)
    query_doc_scores = (
        np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
    )

    print(f"   Минимальный score: {query_doc_scores.min():.4f}")
    print(f"   Максимальный score: {query_doc_scores.max():.4f}")
    print(f"   Средний score: {query_doc_scores.mean():.4f}")
    print(f"   Медианный score: {np.median(query_doc_scores):.4f}")

    # Топ-10 результатов ДО нормализации
    print(f"\n4. Топ-10 результатов ДО нормализации:")
    sorted_indices = np.argsort(query_doc_scores)[::-1][:10]
    for i, idx in enumerate(sorted_indices, 1):
        score = query_doc_scores[idx]
        doc_key = hipporag.passage_node_keys[idx]
        doc_data = hipporag.chunk_embedding_store.get_row(doc_key)
        content = doc_data.get("content", "")[:150]
        print(f"\n   {i}. Score: {score:.4f}")
        print(f"      Текст: {content}...")

    # Нормализация
    from hipporag.utils.misc_utils import min_max_normalize

    query_doc_scores_norm = min_max_normalize(query_doc_scores)

    print(f"\n5. После нормализации:")
    print(f"   Минимальный score: {query_doc_scores_norm.min():.4f}")
    print(f"   Максимальный score: {query_doc_scores_norm.max():.4f}")
    print(f"   Средний score: {query_doc_scores_norm.mean():.4f}")

    # Топ-10 результатов ПОСЛЕ нормализации
    print(f"\n6. Топ-10 результатов ПОСЛЕ нормализации:")
    sorted_indices_norm = np.argsort(query_doc_scores_norm)[::-1][:10]
    for i, idx in enumerate(sorted_indices_norm, 1):
        score = query_doc_scores_norm[idx]
        doc_key = hipporag.passage_node_keys[idx]
        doc_data = hipporag.chunk_embedding_store.get_row(doc_key)
        content = doc_data.get("content", "")[:150]
        print(f"\n   {i}. Score: {score:.4f}")
        print(f"      Текст: {content}...")

    # Проверим несколько случайных документов на наличие слова "Овен"
    print(f"\n7. Поиск документов со словом 'Овен':")
    found_count = 0
    for i, doc_key in enumerate(hipporag.passage_node_keys):
        doc_data = hipporag.chunk_embedding_store.get_row(doc_key)
        content = doc_data.get("content", "")
        if "Овен" in content or "овен" in content:
            score = query_doc_scores[i]
            score_norm = query_doc_scores_norm[i]
            print(f"\n   Документ #{i}")
            print(f"   Score (raw): {score:.4f}, Score (norm): {score_norm:.4f}")
            print(f"   Текст: {content[:200]}...")
            found_count += 1
            if found_count >= 5:
                break

    if found_count == 0:
        print("   ⚠ Не найдено документов со словом 'Овен'!")


if __name__ == "__main__":
    main()
