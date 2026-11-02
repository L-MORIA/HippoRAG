"""Ручная проверка поиска после очистки entities."""

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def manual_search():
    """Интерактивный поиск по индексу."""

    print("=" * 80)
    print("РУЧНАЯ ПРОВЕРКА ПОИСКА")
    print("=" * 80)

    # Путь к очищенному индексу
    index_path = "outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2"

    # Загрузка модели
    print("\nЗагрузка embedding модели...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Загрузка chunks
    chunk_file = os.path.join(index_path, "chunk_embeddings", "vdb_chunk.parquet")
    print(f"Загрузка chunks из индекса...")
    df_chunks = pd.read_parquet(chunk_file)
    print(f"✅ Загружено {len(df_chunks)} chunks")

    # Загрузка entities
    entity_file = os.path.join(index_path, "entity_embeddings", "vdb_entity.parquet")
    df_entities = pd.read_parquet(entity_file)
    print(f"✅ Загружено {len(df_entities)} entities")

    print("\n" + "=" * 80)
    print("Введите запрос для поиска (или 'exit' для выхода)")
    print("Примеры: Овен, Телец, фэншуй, карьера, любовь")
    print("=" * 80)

    while True:
        query = input("\n🔍 Ваш запрос: ").strip()

        if query.lower() in ["exit", "quit", "выход"]:
            print("\n👋 До свидания!")
            break

        if not query:
            print("⚠️  Пустой запрос. Попробуйте снова.")
            continue

        print(f"\n{'='*80}")
        print(f"Поиск по запросу: '{query}'")
        print(f"{'='*80}")

        # Получение embedding запроса
        query_embedding = model.encode(query, convert_to_numpy=True)

        # Вычисление similarity
        similarities = []
        for idx, row in df_chunks.iterrows():
            chunk_embedding = np.array(row["embedding"])
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((idx, similarity, row["content"]))

        # Сортировка
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Вывод топ-5
        print(f"\n📊 ТОП-5 РЕЗУЛЬТАТОВ:\n")
        for i, (idx, score, content) in enumerate(similarities[:5], 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   {content[:250]}...")

            # Проверка релевантности
            if query.lower() in content.lower():
                print(f"   ✅ Содержит '{query}'")
            else:
                print(f"   ⚠️  НЕ содержит '{query}'")
            print()

        # Статистика
        relevant_count = sum(
            1 for _, _, content in similarities[:5] if query.lower() in content.lower()
        )
        print(f"📈 Релевантных в топ-5: {relevant_count}/5 ({relevant_count*20}%)")

        # Позиции релевантных
        positions = []
        for i, (idx, score, content) in enumerate(similarities, 1):
            if query.lower() in content.lower():
                positions.append(i)

        if positions:
            print(f"📍 Всего найдено релевантных: {len(positions)}")
            print(f"📍 Первые 10 позиций: {positions[:10]}")
        else:
            print(f"⚠️  Релевантных документов не найдено")


if __name__ == "__main__":
    try:
        manual_search()
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем. До свидания!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()
