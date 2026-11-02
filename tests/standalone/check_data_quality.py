#!/usr/bin/env python3
"""
Проверка качества данных в индексе.
"""

import json

import pandas as pd

# Читаем индекс
df = pd.read_parquet(
    "outputs/astrology/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2/chunk_embeddings/vdb_chunk.parquet"
)

print(f"Total chunks: {len(df)}")

# Проверяем длину контента
df["content_length"] = df["content"].str.len()

print(f"\nСтатистика длины контента:")
print(f'  Минимум: {df["content_length"].min()} символов')
print(f'  Максимум: {df["content_length"].max()} символов')
print(f'  Среднее: {df["content_length"].mean():.0f} символов')
print(f'  Медиана: {df["content_length"].median():.0f} символов')

# Проверяем, сколько чанков слишком короткие
short_chunks = df[df["content_length"] < 50]
print(
    f"\nКоличество чанков короче 50 символов: {len(short_chunks)} ({len(short_chunks)/len(df)*100:.1f}%)"
)

# Показываем примеры коротких чанков
if len(short_chunks) > 0:
    print("\nПримеры коротких чанков:")
    for i in range(min(5, len(short_chunks))):
        chunk = short_chunks.iloc[i]
        print(f'\n  Чанк {i+1} (длина: {chunk["content_length"]}):')
        print(f'    Content: "{chunk["content"]}"')
        meta = json.loads(chunk["metadata"])
        print(f'    Source: {meta.get("source_file", "N/A")}')

# Показываем примеры нормальных чанков
normal_chunks = df[df["content_length"] >= 200]
print(
    f"\n\nКоличество нормальных чанков (>= 200 символов): {len(normal_chunks)} ({len(normal_chunks)/len(df)*100:.1f}%)"
)

if len(normal_chunks) > 0:
    print("\nПримеры нормальных чанков:")
    for i in range(min(3, len(normal_chunks))):
        chunk = normal_chunks.iloc[i]
        print(f'\n  Чанк {i+1} (длина: {chunk["content_length"]}):')
        print(f'    Content: {chunk["content"][:300]}...')
        meta = json.loads(chunk["metadata"])
        print(f'    Source: {meta.get("source_file", "N/A")}')

# Ищем чанки с упоминанием Овна
oven_chunks = df[df["content"].str.contains("Овен", case=False, na=False)]
print(f'\n\nЧанки с упоминанием "Овен": {len(oven_chunks)}')

if len(oven_chunks) > 0:
    print('\nПримеры чанков с "Овен":')
    for i in range(min(3, len(oven_chunks))):
        chunk = oven_chunks.iloc[i]
        print(f'\n  Чанк {i+1} (длина: {chunk["content_length"]}):')
        print(f'    Content: {chunk["content"][:400]}...')
        meta = json.loads(chunk["metadata"])
        print(f'    Source: {meta.get("source_file", "N/A")}')
        print(f'    Page: {meta.get("page_num", "N/A")}')
