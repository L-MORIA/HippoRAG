#!/usr/bin/env python3
"""
Интерактивный скрипт для ручного тестирования поиска в HippoRAG.
Позволяет вводить запросы вручную и получать результаты.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def print_separator(char="=", length=80):
    """Печать разделителя."""
    print(char * length)


def print_result(idx, content, max_chars=800):
    """Печать одного результата поиска."""
    print(f"\n{'─' * 80}")
    print(f"Результат {idx}")
    print(f"{'─' * 80}")
    
    if len(content) > max_chars:
        print(content[:max_chars] + "...")
        print(f"\n[Показано {max_chars} из {len(content)} символов]")
    else:
        print(content)


def main():
    """Главная функция интерактивного поиска."""
    
    print_separator()
    print("ИНТЕРАКТИВНЫЙ ПОИСК В HIPPORAG")
    print_separator()
    print()
    
    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology_base"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "intfloat/multilingual-e5-base"
    config.retrieval_top_k = 5  # Топ-5 результатов
    
    print("⏳ Инициализация HippoRAG...")
    print(f"   Директория: {config.save_dir}")
    print(f"   LLM: {config.llm_name}")
    print(f"   Embedding: {config.embedding_model_name}")
    print()
    
    try:
        hipporag = HippoRAG(global_config=config)
        
        print("✓ Система готова!")
        print(f"  • Загружено чанков: {len(hipporag.chunk_embedding_store.hash_ids)}")
        print(f"  • Загружено entities: {len(hipporag.entity_embedding_store.hash_ids)}")
        print(f"  • Загружено facts: {len(hipporag.fact_embedding_store.hash_ids)}")
        print()
        
    except Exception as e:
        print(f"✗ Ошибка инициализации: {e}")
        return
    
    print_separator()
    print("ИНСТРУКЦИИ")
    print_separator()
    print("• Введите ваш запрос и нажмите Enter")
    print("• Для выхода введите 'exit', 'quit' или 'q'")
    print("• Для очистки экрана введите 'clear' или 'cls'")
    print()
    
    # Примеры запросов
    print("Примеры запросов:")
    examples = [
        "Какие особенности квартиры для знака Овен?",
        "Что говорит фэншуй о планировке для Тельца?",
        "Какие цвета подходят для знака Близнецы?",
        "Как обустроить спальню для Рака?",
        "Какая мебель подходит для Льва?",
    ]
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example}")
    print()
    
    query_count = 0
    
    # Основной цикл
    while True:
        print_separator("─")
        try:
            query = input("\n🔍 Ваш запрос: ").strip()
            
            if not query:
                continue
            
            # Команды выхода
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 До свидания!")
                break
            
            # Команда очистки экрана
            if query.lower() in ['clear', 'cls']:
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            query_count += 1
            
            print(f"\n⏳ Поиск... (запрос #{query_count})")
            
            # Выполняем поиск
            results = hipporag.retrieve([query], num_to_retrieve=5)
            
            if results and len(results) > 0:
                query_result = results[0]
                retrieved_docs = query_result.docs if hasattr(query_result, 'docs') else []
                
                if retrieved_docs:
                    print(f"\n✓ Найдено результатов: {len(retrieved_docs)}")
                    
                    for i, doc in enumerate(retrieved_docs, 1):
                        content = doc if isinstance(doc, str) else str(doc)
                        print_result(i, content)
                else:
                    print("\n⚠ Результаты не найдены")
            else:
                print("\n⚠ Результаты не найдены")
        
        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
            break
        
        except Exception as e:
            print(f"\n✗ Ошибка при поиске: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print_separator()
    print(f"Всего выполнено запросов: {query_count}")
    print_separator()


if __name__ == "__main__":
    main()
