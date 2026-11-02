#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv

from hipporag import HippoRAG

# Загрузка переменных окружения
load_dotenv()

# Инициализация HippoRAG
print("Инициализация HippoRAG...")
hipporag = HippoRAG(
    save_dir="outputs/astrology",
    llm_model_name=os.getenv("LLM_NAME", "gemma-4-e4b"),
    embedding_model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    llm_base_url=os.getenv("LLM_BASE_URL", "http://172.18.0.1:1234/v1"),
)

print("\n" + "=" * 80)
print("РУЧНОЕ ТЕСТИРОВАНИЕ ПОИСКА")
print("=" * 80)

# Список тестовых запросов
test_queries = [
    "Овен",
    "Телец",
    "Близнецы",
    "Рак",
    "Лев",
    "Дева",
    "Весы",
    "Скорпион",
    "Стрелец",
    "Козерог",
    "Водолей",
    "Рыбы",
    "фэншуй",
    "гороскоп",
    "планеты",
]

print("\nДоступные тестовые запросы:")
for i, q in enumerate(test_queries, 1):
    print(f"{i}. {q}")

print("\nВведите:")
print("- номер запроса (1-15)")
print("- свой запрос")
print("- 'exit' для выхода")

while True:
    print("\n" + "-" * 80)
    user_input = input("\nВаш запрос: ").strip()

    if user_input.lower() == "exit":
        print("Выход...")
        break

    # Проверяем, это номер или текст
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(test_queries):
            query = test_queries[idx]
        else:
            print(f"Ошибка: введите номер от 1 до {len(test_queries)}")
            continue
    else:
        query = user_input

    if not query:
        print("Ошибка: пустой запрос")
        continue

    print(f"\n{'='*80}")
    print(f"ПОИСК: '{query}'")
    print("=" * 80)

    # Выполняем поиск
    try:
        results = hipporag.retrieve([query], num_to_retrieve=5)

        if not results or not results[0].docs:
            print("\n❌ Ничего не найдено")
            continue

        query_solution = results[0]
        docs = query_solution.docs
        scores = query_solution.doc_scores if query_solution.doc_scores else [0] * len(docs)

        print(f"\n✅ Найдено документов: {len(docs)}")
        print(f"{'='*80}\n")

        for i, (doc, score) in enumerate(zip(docs, scores), 1):
            print(f"{'─'*80}")
            print(f"📄 ДОКУМЕНТ {i} (score: {score:.4f})")
            print(f"{'─'*80}")
            print(f"Длина: {len(doc)} символов")
            print(f"\nТекст:")
            print(doc)
            print()

        # Статистика
        print(f"{'='*80}")
        print("СТАТИСТИКА:")
        print(f"  Всего документов: {len(docs)}")
        print(f"  Средняя длина: {sum(len(d) for d in docs) / len(docs):.0f} символов")
        print(f"  Мин. длина: {min(len(d) for d in docs)} символов")
        print(f"  Макс. длина: {max(len(d) for d in docs)} символов")

        # Проверка на наличие запроса в результатах
        found_in = sum(1 for doc in docs if query.lower() in doc.lower())
        print(f"  Запрос найден в {found_in}/{len(docs)} документах")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n❌ Ошибка при поиске: {e}")
        import traceback

        traceback.print_exc()

print("\nГотово!")
