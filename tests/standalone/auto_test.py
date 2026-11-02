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
print("АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ ПОИСКА")
print("=" * 80)

# Список тестовых запросов
test_queries = ["Овен", "Телец", "Скорпион", "фэншуй", "карьера"]

for query_num, query in enumerate(test_queries, 1):
    print(f"\n{'='*80}")
    print(f"ТЕСТ {query_num}/{len(test_queries)}: '{query}'")
    print("=" * 80)

    try:
        results = hipporag.retrieve([query], num_to_retrieve=3)

        if not results or not results[0].docs:
            print("\n❌ Ничего не найдено")
            continue

        query_solution = results[0]
        docs = query_solution.docs

        # Безопасная проверка doc_scores (может быть numpy array или None)
        if query_solution.doc_scores is not None and len(query_solution.doc_scores) > 0:
            scores = query_solution.doc_scores
        else:
            scores = [0] * len(docs)

        print(f"\n✅ Найдено документов: {len(docs)}")

        for i, (doc, score) in enumerate(zip(docs[:3], scores[:3]), 1):
            print(f"\n{'─'*80}")
            print(f"📄 ДОКУМЕНТ {i} (score: {score:.4f})")
            print(f"{'─'*80}")
            print(f"Длина: {len(doc)} символов")

            # Показываем первые 200 символов
            preview = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"\nПревью:\n{preview}")

            # Проверяем наличие запроса
            if query.lower() in doc.lower():
                print(f"✅ Запрос '{query}' найден в тексте")
            else:
                print(f"⚠️  Запрос '{query}' НЕ найден в тексте (но документ релевантен)")

        # Статистика
        print(f"\n{'='*80}")
        print("СТАТИСТИКА:")
        print(f"  Всего документов: {len(docs)}")
        print(f"  Средняя длина: {sum(len(d) for d in docs) / len(docs):.0f} символов")
        print(f"  Мин. длина: {min(len(d) for d in docs)} символов")
        print(f"  Макс. длина: {max(len(d) for d in docs)} символов")

        # Проверка на наличие запроса в результатах
        found_in = sum(1 for doc in docs if query.lower() in doc.lower())
        print(f"  Запрос найден в {found_in}/{len(docs)} документах")

    except Exception as e:
        print(f"\n❌ Ошибка при поиске: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 80)
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print("=" * 80)
