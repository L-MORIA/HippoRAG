"""Тест поиска после очистки entities."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import logging

from dotenv import load_dotenv

from hipporag.HippoRAG import HippoRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Используем очищенный индекс
os.environ["SAVE_DIR"] = "outputs/astrology_test"
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

load_dotenv()


def test_search(query: str):
    """Тест поиска по запросу."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ЗАПРОС: {query}")
    logger.info(f"{'='*80}")

    # Инициализация HippoRAG
    logger.info("Инициализация HippoRAG с очищенным индексом...")
    rag = HippoRAG()

    # Поиск
    logger.info(f"Поиск по запросу: '{query}'")
    results = rag.rank_docs(query, top_k=5)

    # Вывод результатов
    logger.info(f"\nНайдено {len(results)} результатов:")
    for i, (doc, score) in enumerate(results, 1):
        logger.info(f"\n--- Результат {i} (score: {score:.4f}) ---")
        logger.info(f"{doc[:300]}...")

    return results


if __name__ == "__main__":
    query = "Овен"
    results = test_search(query)

    print("\n" + "=" * 80)
    print(f"ИТОГО: Найдено {len(results)} документов по запросу '{query}'")
    print("=" * 80)
