#!/usr/bin/env python3
"""
Детальная проверка того, что возвращает система при поиске.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Отключаем лишние логи
logging.basicConfig(level=logging.ERROR)


def main():
    print("🔍 Детальная проверка результатов поиска")
    print("=" * 60)

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology"
    config.dataset = "astrology"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://172.18.0.1:1234/v1"
    config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Загружаем индекс напрямую
    print("\n📂 Загружаем индекс напрямую из parquet...")
    df = pd.read_parquet(
        "outputs/astrology/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2/chunk_embeddings/vdb_chunk.parquet"
    )

    # Ищем чанк с Овном
    oven_chunks = df[
        df["content"].str.contains("ОВЕН.*Начнём мы описание", case=False, na=False, regex=True)
    ]

    if len(oven_chunks) > 0:
        print(f"✅ Найден чанк с описанием Овна в индексе")
        chunk = oven_chunks.iloc[0]
        print(f"\n📄 Содержимое из индекса (первые 500 символов):")
        print(chunk["content"][:500])
        print(f"\n📊 Полная длина: {len(chunk['content'])} символов")
        print(f"📝 Hash ID: {chunk['hash_id']}")

    # Теперь загружаем через HippoRAG
    print("\n\n🤖 Загружаем через HippoRAG...")
    hipporag = HippoRAG(global_config=config)

    # Выполняем поиск
    print("\n🔎 Выполняем поиск: 'Овен прихожая'")
    results = hipporag.retrieve(["Овен прихожая"], num_to_retrieve=3)

    if results and len(results) > 0:
        solution = results[0]
        print(f"\n✅ Найдено {len(solution.docs)} документов")

        for i, (doc, score) in enumerate(zip(solution.docs[:3], solution.doc_scores[:3]), 1):
            print(f"\n{'='*60}")
            print(f"📄 Результат {i} (оценка: {score:.4f})")
            print(f"📊 Длина текста: {len(doc)} символов")
            print(f"📝 Содержимое:")
            print(doc)
            print(f"{'='*60}")
    else:
        print("❌ Ничего не найдено")

    # Проверяем chunk_embedding_store напрямую
    print("\n\n🔍 Проверяем chunk_embedding_store напрямую...")
    all_texts = hipporag.chunk_embedding_store.get_all_texts()
    print(f"📊 Всего текстов в store: {len(all_texts)}")

    # Ищем текст с Овном
    oven_texts = [t for t in all_texts if "ОВЕН" in t and "Начнём" in t]
    if oven_texts:
        print(f"\n✅ Найдено {len(oven_texts)} текстов с описанием Овна")
        print(f"\n📄 Первый текст (первые 500 символов):")
        print(oven_texts[0][:500])
        print(f"\n📊 Полная длина: {len(oven_texts[0])} символов")
    else:
        print("❌ Тексты с описанием Овна не найдены в store")


if __name__ == "__main__":
    main()
