#!/usr/bin/env python3
"""
Тест релевантности результатов поиска.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def test_query(hipporag, query, expected_keywords):
    """Тестирует один запрос и проверяет релевантность."""
    print(f"\n{'='*80}")
    print(f"Запрос: {query}")
    print(f"{'='*80}")

    results = hipporag.retrieve([query], num_to_retrieve=3)

    if results and len(results) > 0:
        query_result = results[0]
        retrieved_docs = query_result.docs if hasattr(query_result, "docs") else []

        print(f"\nНайдено результатов: {len(retrieved_docs)}")

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc if isinstance(doc, str) else str(doc)
            print(f"\n--- Результат {i} (длина: {len(content)} символов) ---")
            print(content[:400])

            # Проверка релевантности
            found_keywords = [kw for kw in expected_keywords if kw.lower() in content.lower()]
            if found_keywords:
                print(f"\n✓ Найдены ключевые слова: {', '.join(found_keywords)}")
            else:
                print(f"\n✗ НЕ найдены ожидаемые ключевые слова: {', '.join(expected_keywords)}")
    else:
        print("\n✗ Результаты не найдены")


def main():
    """Главная функция."""

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology_base"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "intfloat/multilingual-e5-base"
    config.retrieval_top_k = 3

    print("Инициализация HippoRAG...")
    hipporag = HippoRAG(global_config=config)

    print(f"✓ Загружено чанков: {len(hipporag.chunk_embedding_store.hash_ids)}")
    print(f"✓ Загружено entities: {len(hipporag.entity_embedding_store.hash_ids)}")
    print(f"✓ Загружено facts: {len(hipporag.fact_embedding_store.hash_ids)}")

    # Тестовые запросы с ожидаемыми ключевыми словами
    test_cases = [
        ("Какие особенности квартиры для знака Овен?", ["Овен", "Овна", "прихожая", "коридор"]),
        ("Что говорит фэншуй о планировке для Тельца?", ["Телец", "Тельца", "кухня", "стол"]),
        ("Какие цвета подходят для знака Близнецы?", ["Близнец", "Близнецы", "коридор"]),
        ("Как обустроить спальню для Рака?", ["Рак", "Рака", "спальня", "кухня"]),
    ]

    for query, keywords in test_cases:
        test_query(hipporag, query, keywords)

    print(f"\n{'='*80}")
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
