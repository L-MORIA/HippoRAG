#!/usr/bin/env python3
"""
Автоматический тест поиска с multilingual моделью
"""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    # Настройка конфигурации
    config = BaseConfig()
    config.save_dir = "outputs/astrology"
    config.dataset = "astrology"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://172.18.0.1:1234/v1"
    config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    os.environ["LLM_NAME"] = "gemma-4-e4b"
    os.environ["LLM_BASE_URL"] = "http://172.18.0.1:1234/v1"

    logger.info(f"Using LOCAL LLM: {config.llm_name} at {config.llm_base_url}")
    logger.info(f"Using MULTILINGUAL embedding model: {config.embedding_model_name}")

    # Создание экземпляра HippoRAG
    logger.info("Загрузка индексов...")
    hipporag = HippoRAG(global_config=config)
    logger.info("Система готова к поиску!")

    # Тестовый запрос
    test_query = "что такое число гуа 5"

    print("\n" + "=" * 80)
    print(f"ТЕСТОВЫЙ ЗАПРОС: {test_query}")
    print("=" * 80)

    # Выполняем поиск
    results = hipporag.rag_qa(queries=[test_query])

    # results - это tuple (queries_solutions, messages, metadata)
    queries_solutions = results[0]
    result = queries_solutions[0]

    print(f"\n{'='*80}")
    print("РЕЗУЛЬТАТ:")
    print("=" * 80)
    print(f"\nОтвет: {result.answer}")

    if result.citations:
        print(f"\nЦитаты из источников ({len(result.citations)} шт.):")
        for i, citation in enumerate(result.citations[:10], 1):
            source_info = f"[{citation.doc_title}"
            if citation.page_num:
                source_info += f", стр. {citation.page_num}"
            if citation.section:
                source_info += f", секция: {citation.section}"
            source_info += "]"
            print(f"  {i}. {source_info}")

    if result.docs:
        print(f"\nРелевантные фрагменты документов (первые 3 из {len(result.docs)}):")
        for i, doc in enumerate(result.docs[:3], 1):
            snippet = doc[:200] + "..." if len(doc) > 200 else doc
            print(f"\n  {i}. {snippet}")

    print("\n" + "=" * 80)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == "__main__":
    main()
