"""
Тестовый скрипт с использованием DPR и multilingual-e5-base модели.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ ПОИСКА (DPR + MULTILINGUAL-E5-BASE)")
    print("=" * 80)

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology_base"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "intfloat/multilingual-e5-base"
    config.retrieval_top_k = 5

    print("\nИнициализация HippoRAG...")
    hipporag = HippoRAG(global_config=config)

    print(f"✓ Загружено чанков: {len(hipporag.chunk_embedding_store.hash_ids)}")
    print(f"✓ Загружено entities: {len(hipporag.entity_embedding_store.hash_ids)}")
    print(f"✓ Загружено facts: {len(hipporag.fact_embedding_store.hash_ids)}")

    # Тестовые запросы
    test_queries = [
        "Какие особенности квартиры для знака Овен?",
        "Что говорит фэншуй о планировке для Тельца?",
        "Какие цвета подходят для знака Близнецы?",
    ]

    print("\n" + "=" * 80)
    print("ТЕСТОВЫЕ ЗАПРОСЫ (DPR + MULTILINGUAL-E5-BASE)")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Запрос {i}: {query}")
        print("=" * 80)

        try:
            # Используем DPR метод
            print("Выполняется поиск (DPR)...")
            results = hipporag.retrieve_dpr([query], num_to_retrieve=5)

            if results and len(results) > 0:
                query_result = results[0]
                retrieved_docs = query_result.docs
                doc_scores = query_result.doc_scores if query_result.doc_scores is not None else []

                print(f"\n✓ Найдено результатов: {len(retrieved_docs)}")

                for j, doc_content in enumerate(retrieved_docs[:5], 1):
                    print(f"\n--- Результат {j} ---")

                    if doc_scores is not None and j - 1 < len(doc_scores):
                        print(f"Score: {doc_scores[j-1]:.4f}")

                    print(f"\nТекст (первые 300 символов):")
                    print(doc_content[:300] + "..." if len(doc_content) > 300 else doc_content)
            else:
                print("✗ Результаты не найдены")

        except Exception as e:
            print(f"✗ Ошибка при поиске: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\n📊 Статистика:")
    print(f"  - Проиндексировано чанков: {len(hipporag.chunk_embedding_store.hash_ids)}")
    print(f"  - Извлечено entities: {len(hipporag.entity_embedding_store.hash_ids)}")
    print(f"  - Извлечено facts: {len(hipporag.fact_embedding_store.hash_ids)}")
    print("\n💡 Примечание:")
    print("  Используется модель intfloat/multilingual-e5-base (768 dim)")
    print("  для максимального качества поиска на русском языке.")


if __name__ == "__main__":
    main()
