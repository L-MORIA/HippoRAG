"""
Упрощенный тестовый скрипт для поиска в проиндексированной базе HippoRAG.
Использует правильный API метод retrieve().
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ СИСТЕМЫ ПОИСКА HIPPORAG")
    print("=" * 80)

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology_base"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "intfloat/multilingual-e5-base"
    config.retrieval_top_k = 3  # Топ-3 результата

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
    print("ТЕСТОВЫЕ ЗАПРОСЫ")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Запрос {i}: {query}")
        print("=" * 80)

        try:
            # Поиск с HippoRAG - передаем список запросов
            print("Выполняется поиск...")
            results = hipporag.retrieve([query], num_to_retrieve=3)

            # results - это список QuerySolution объектов
            if results and len(results) > 0:
                query_result = results[0]  # Берем результат для первого запроса

                # QuerySolution содержит docs (не retrieved_docs!)
                retrieved_docs = query_result.docs if hasattr(query_result, "docs") else []

                print(f"\n✓ Найдено результатов: {len(retrieved_docs)}")

                for j, doc_id in enumerate(retrieved_docs[:3], 1):
                    print(f"\n--- Результат {j} ---")

                    # doc_id уже содержит текст чанка
                    content = doc_id if isinstance(doc_id, str) else str(doc_id)

                    print(f"\nТекст (первые 500 символов):")
                    print(content[:500] + "..." if len(content) > 500 else content)
            else:
                print("✗ Результаты не найдены")

        except Exception as e:
            print(f"✗ Ошибка при поиске: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\nДля дальнейшего тестирования вы можете:")
    print("1. Запустить этот скрипт снова: python test_search_simple.py")
    print("2. Изменить запросы в коде скрипта")
    print("3. Дождаться завершения полной индексации для большей базы данных")


if __name__ == "__main__":
    main()
