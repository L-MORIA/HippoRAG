"""Простая проверка данных в очищенном индексе."""

import os

import pandas as pd

# Путь к очищенному индексу
index_path = (
    "outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2"
)

print("=" * 80)
print("ПРОВЕРКА ОЧИЩЕННОГО ИНДЕКСА")
print("=" * 80)

# Проверка entities
entity_file = os.path.join(index_path, "entity_embeddings", "vdb_entity.parquet")
if os.path.exists(entity_file):
    df_entities = pd.read_parquet(entity_file)
    print(f"\n✅ ENTITIES: {len(df_entities)}")
    print("\nТоп-20 entities:")
    for i, entity in enumerate(df_entities["content"].head(20), 1):
        print(f"  {i:2}. {entity}")

    # Проверка на мусор
    garbage = []
    for entity in df_entities["content"]:
        if entity.isdigit() or not entity or len(entity) < 2:
            garbage.append(entity)

    if garbage:
        print(f"\n⚠️  Найдено {len(garbage)} мусорных entities:")
        for g in garbage[:10]:
            print(f"  - '{g}'")
    else:
        print("\n✅ Мусорных entities не найдено!")
else:
    print(f"❌ Файл не найден: {entity_file}")

# Проверка chunks
chunk_file = os.path.join(index_path, "chunk_embeddings", "vdb_chunk.parquet")
if os.path.exists(chunk_file):
    df_chunks = pd.read_parquet(chunk_file)
    print(f"\n✅ CHUNKS: {len(df_chunks)}")

    # Поиск чанков с "Овен"
    oven_chunks = df_chunks[df_chunks["content"].str.contains("Овен", case=False, na=False)]
    print(f"\n🔍 Чанков с 'Овен': {len(oven_chunks)}")

    if len(oven_chunks) > 0:
        print("\nПримеры чанков с 'Овен':")
        for i, (idx, row) in enumerate(oven_chunks.head(3).iterrows(), 1):
            content = row["content"][:200]
            print(f"\n  {i}. {content}...")
else:
    print(f"❌ Файл не найден: {chunk_file}")

# Проверка facts
fact_file = os.path.join(index_path, "fact_embeddings", "vdb_fact.parquet")
if os.path.exists(fact_file):
    df_facts = pd.read_parquet(fact_file)
    print(f"\n✅ FACTS: {len(df_facts)}")
else:
    print(f"❌ Файл не найден: {fact_file}")

print("\n" + "=" * 80)
print("ПРОВЕРКА ЗАВЕРШЕНА")
print("=" * 80)
