#!/usr/bin/env python3
"""
Полный тест системы HippoRAG с исправленной конфигурацией
"""

import os
import sys

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(__file__))

os.environ["LLM_NAME"] = "google/gemma-4-e4b"
os.environ["LLM_BASE_URL"] = "http://localhost:1234/v1"

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def main():
    print("=" * 60)
    print("ПОЛНЫЙ ТЕСТ СИСТЕМЫ С ИСПРАВЛЕННОЙ КОНФИГУРАЦИЕЙ")
    print("=" * 60)
    print()

    config = BaseConfig()
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    config.save_dir = "outputs/astrology"
    config.dataset = "astrology"

    print(f"LLM: {config.llm_name}")
    print(f"LLM URL: {config.llm_base_url}")
    print(f"Embedding: {config.embedding_model_name}")
    print(f"Save dir: {config.save_dir}")
    print()

    try:
        rag = HippoRAG(global_config=config)
        print("✅ Система загружена успешно!")
        print()

        # Тест поиска
        query = "число гуа 7"
        print(f'Тестовый запрос: "{query}"')
        print()

        results = rag.retrieve_dpr(queries=[query], num_to_retrieve=3)
        result = results[0]

        print("✅ Результаты поиска:")
        for i, (doc, score) in enumerate(zip(result.docs, result.doc_scores), 1):
            target = "гуа 7"
            has_target = target in doc.lower()
            preview = doc[:70].replace("\n", " ")
            print(f"  {i}. [{score:.3f}][{has_target}] {preview}...")

        print()
        print("✅ СИСТЕМА РАБОТАЕТ КОРРЕКТНО!")
        print()
        print("Теперь можно запустить main_lm_studio.py для полного RAG с LLM")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
