#!/usr/bin/env python3
"""
Скрипт для проверки содержимого индекса.
"""

import json

import pandas as pd

# Читаем индекс
df = pd.read_parquet(
    "outputs/astrology/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2/chunk_embeddings/vdb_chunk.parquet"
)

print(f"Total chunks: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Проверяем первые 5 чанков
print("\n=== First 5 chunks ===")
for i in range(min(5, len(df))):
    print(f"\n--- Chunk {i+1} ---")
    print(f'Content: {df.iloc[i]["content"][:200]}...')
    meta = json.loads(df.iloc[i]["metadata"])
    print(f'Source: {meta.get("source_file", "N/A")}')
    print(f'Page: {meta.get("page_num", "N/A")}')

# Ищем упоминания Овна
print('\n\n=== Searching for "Овен" ===')
oven_chunks = df[df["content"].str.contains("Овен", case=False, na=False)]
print(f'Found {len(oven_chunks)} chunks mentioning "Овен"')

if len(oven_chunks) > 0:
    print('\nFirst 3 chunks with "Овен":')
    for i in range(min(3, len(oven_chunks))):
        chunk = oven_chunks.iloc[i]
        print(f"\n--- Match {i+1} ---")
        print(f'Content: {chunk["content"][:300]}...')
        meta = json.loads(chunk["metadata"])
        print(f'Source: {meta.get("source_file", "N/A")}')
        print(f'Page: {meta.get("page_num", "N/A")}')
