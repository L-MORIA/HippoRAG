#!/usr/bin/env python3
"""
Скрипт для тестирования поиска по первому индексированному PDF-документу.
"""

import os
import sys

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.lm_studio_manager import setup_environment


def main():
    """Главная функция тестирования поиска."""
    print("🔍 Тестирование поиска по индексированному документу")
    print("=" * 50)

    try:
        # Настройка конфигурации
        config = BaseConfig()
        config.save_dir = "outputs/astrology"
        config.llm_name = "google/gemma-4-e4b"
        config.llm_base_url = "http://172.18.0.1:1234/v1"
        # Используем ту же embedding модель, что и при индексации
        config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

        # Проверяем, что индекс существует
        index_path = os.path.join(
            config.save_dir,
            "google_gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2",
        )
        if not os.path.exists(index_path):
            print(f"❌ Индекс не найден по пути: {index_path}")
            print("Пожалуйста, сначала выполните индексацию документа.")
            return

        print(f"✅ Индекс найден: {index_path}")

        # Автоматическая настройка окружения
        print("\n🚀 Настройка окружения...")
        lm_ready, embedding_ready = setup_environment(
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
            lm_studio_url=config.llm_base_url.replace("/v1", ""),
        )

        if not embedding_ready:
            print("❌ Не удалось загрузить embedding модель.")
            return

        if not lm_ready:
            print("⚠️  LM Studio не готов. Продолжение с ограниченной функциональностью.")

        # Создание экземпляра HippoRAG
        print("🔄 Загрузка индекса...")
        hipporag = HippoRAG(global_config=config)

        # Тестовые запросы
        test_queries = [
            "Какие знаки зодиака упоминаются в документе?",
            "Какие методы фэншуй описаны в документе?",
            "Кто автор этого документа?",
        ]

        print("\n🔎 Запуск тестовых запросов:")
        for i, query in enumerate(test_queries, 1):
            print(f"\nВопрос {i}: {query}")
            print("-" * 30)

            try:
                # Выполняем поиск
                results = hipporag.retrieve([query], num_to_retrieve=3)

                if results:
                    print("📝 Результаты поиска:")
                    # results является списком QuerySolution объектов
                    solution = results[0]  # Берем первый (и единственный) QuerySolution
                    for j, (doc, score) in enumerate(
                        zip(solution.docs[:2], solution.doc_scores[:2])
                    ):  # Показываем максимум 2 результата
                        print(f"  Результат {j+1}:")
                        print(f"    Текст: {doc[:200]}...")
                        print(f"    Оценка: {score:.4f}")
                        print()
                else:
                    print("  ❌ Ничего не найдено.")

            except Exception as e:
                print(f"  ❌ Ошибка при поиске: {str(e)}")
                import traceback

                traceback.print_exc()

        print("\n✅ Тестирование поиска завершено.")

    except Exception as e:
        print(f"❌ Ошибка во время тестирования: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
