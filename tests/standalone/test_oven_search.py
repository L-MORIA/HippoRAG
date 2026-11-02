#!/usr/bin/env python3
"""
Тестовый скрипт для проверки поиска информации об Овне.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Настройка логирования
logging.basicConfig(level=logging.WARNING)  # Уменьшаем количество логов


def main():
    print("🔍 Тестирование поиска информации об Овне")
    print("=" * 50)

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology"
    config.dataset = "astrology"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://172.18.0.1:1234/v1"
    config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    print(f"📂 Используем индекс: {config.save_dir}")
    print(f"🤖 Модель LLM: {config.llm_name}")
    print(f"🔤 Embedding модель: {config.embedding_model_name}")

    # Создание экземпляра HippoRAG
    print("\n⏳ Загрузка индекса...")
    hipporag = HippoRAG(global_config=config)
    print("✅ Индекс загружен!")

    # Тестовые запросы об Овне
    queries = [
        "Овен",
        "Какие характеристики Овна?",
        "Что соответствует знаку Овен в квартире?",
        "прихожая Овен",
        "точка силы Овна",
    ]

    for query in queries:
        print(f"\n{'='*50}")
        print(f"🔎 Запрос: {query}")
        print("=" * 50)

        try:
            # Выполняем поиск БЕЗ генерации ответа, только retrieval
            results = hipporag.retrieve([query], num_to_retrieve=5)

            if results and len(results) > 0:
                solution = results[0]
                print(f"✅ Найдено {len(solution.docs)} документов")

                # Показываем первые 3 результата
                for i, (doc, score) in enumerate(
                    zip(solution.docs[:3], solution.doc_scores[:3]), 1
                ):
                    print(f"\n📄 Результат {i} (оценка: {score:.4f}):")
                    print(f"   {doc[:300]}...")

                    # Показываем цитаты если есть
                    if solution.citations and i - 1 < len(solution.citations):
                        citation = solution.citations[i - 1]
                        print(f"   📚 Источник: {citation.doc_title}, стр. {citation.page_num}")
            else:
                print("❌ Ничего не найдено")

        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("✅ Тестирование завершено")


if __name__ == "__main__":
    main()
