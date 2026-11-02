"""
Тестовый скрипт для поиска в проиндексированной базе HippoRAG.
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
    config.save_dir = "outputs/astrology"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

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
        "Расскажи про астрологический фэншуй для Рака",
        "Какие планеты влияют на знак Льва?",
    ]

    print("\n" + "=" * 80)
    print("ТЕСТОВЫЕ ЗАПРОСЫ")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Запрос {i}: {query}")
        print("=" * 80)

        try:
            # Поиск с HippoRAG
            results = hipporag.rank_docs(query, top_k=3)

            print(f"\nНайдено результатов: {len(results)}")

            for j, result in enumerate(results, 1):
                print(f"\n--- Результат {j} ---")
                print(f"Score: {result.get('score', 'N/A')}")

                # Получаем содержимое чанка
                chunk_id = result.get("chunk_id") or result.get("hash_id")
                if chunk_id:
                    chunk_data = hipporag.chunk_embedding_store.get_row(chunk_id)
                    content = chunk_data.get("content", "")
                    metadata = chunk_data.get("metadata", {})

                    print(f"Источник: {metadata.get('source_file', 'unknown')}")
                    print(f"Страница: {metadata.get('page_num', 'N/A')}")
                    print(f"\nТекст (первые 300 символов):")
                    print(content[:300] + "..." if len(content) > 300 else content)
                else:
                    print("Содержимое недоступно")

        except Exception as e:
            print(f"Ошибка при поиске: {e}")
            import traceback

            traceback.print_exc()

    # Интерактивный режим
    print("\n" + "=" * 80)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 80)
    print("Введите ваш вопрос (или 'exit' для выхода):")

    while True:
        try:
            query = input("\n> ").strip()

            if query.lower() in ["exit", "quit", "выход"]:
                print("До свидания!")
                break

            if not query:
                continue

            print(f"\nПоиск: {query}")
            print("-" * 80)

            results = hipporag.rank_docs(query, top_k=3)

            print(f"\nНайдено результатов: {len(results)}")

            for j, result in enumerate(results, 1):
                print(f"\n--- Результат {j} ---")
                print(f"Score: {result.get('score', 'N/A')}")

                chunk_id = result.get("chunk_id") or result.get("hash_id")
                if chunk_id:
                    chunk_data = hipporag.chunk_embedding_store.get_row(chunk_id)
                    content = chunk_data.get("content", "")
                    metadata = chunk_data.get("metadata", {})

                    print(f"Источник: {metadata.get('source_file', 'unknown')}")
                    print(f"Страница: {metadata.get('page_num', 'N/A')}")
                    print(f"\nТекст:")
                    print(content[:500] + "..." if len(content) > 500 else content)

        except KeyboardInterrupt:
            print("\n\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
