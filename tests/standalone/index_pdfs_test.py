#!/usr/bin/env python3
"""
Тестовый скрипт индексации PDF-документов с ограничениями для быстрой проверки.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from multiprocessing import freeze_support

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("indexing_test.log"), logging.StreamHandler(sys.stdout)],
    )


def find_pdf_files(directory, limit=1):
    """Найти PDF-файлы в директории (с ограничением)."""
    pdf_files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Директория {directory} не существует")

    for pdf_file in directory_path.rglob("*.pdf"):
        pdf_files.append(str(pdf_file.resolve()))
        if len(pdf_files) >= limit:
            break

    return pdf_files


def main():
    """Главная функция тестовой индексации."""
    parser = argparse.ArgumentParser(description="Тестовая индексация PDF для Astrology RAG")
    parser.add_argument("--pdf_dir", default="data/pdfs", help="Директория с PDF-файлами")
    parser.add_argument(
        "--output_dir", default="outputs/astrology_test", help="Директория для сохранения индексов"
    )
    parser.add_argument(
        "--max_pdfs",
        type=int,
        default=1,
        help="Максимальное количество PDF для обработки (по умолчанию: 1)",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=50,
        help="Максимальное количество чанков для индексации (по умолчанию: 50)",
    )

    args = parser.parse_args()

    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Проверяем наличие директории с PDF
        if not os.path.exists(args.pdf_dir):
            logger.error(f"Директория {args.pdf_dir} не существует")
            return

        # Ищем PDF-файлы (ограниченное количество)
        pdf_files = find_pdf_files(args.pdf_dir, limit=args.max_pdfs)

        if not pdf_files:
            logger.warning(f"В директории {args.pdf_dir} не найдено PDF-файлов")
            return

        logger.info(f"Найдено {len(pdf_files)} PDF-файлов для тестовой индексации")
        for pdf in pdf_files:
            logger.info(f"  - {pdf}")

        # Настройка конфигурации
        config = BaseConfig()
        config.save_dir = args.output_dir
        config.llm_name = "gemma-4-e4b"
        config.llm_base_url = "http://localhost:1234/v1"
        config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.force_index_from_scratch = False  # Используем существующий индекс

        # Создание экземпляра HippoRAG
        logger.info("Инициализация HippoRAG...")
        hipporag = HippoRAG(global_config=config)

        # Индексация PDF-файлов с ограничением
        logger.info(f"Начало тестовой индексации (макс. {args.max_chunks} чанков)...")

        # Парсим PDF и ограничиваем количество чанков
        from hipporag.pdf_ingestion import TextPDFParser

        all_chunks = []
        all_metadatas = []

        for pdf_path in pdf_files:
            logger.info(f"Обработка {pdf_path}")
            parser = TextPDFParser()

            # Ограничиваем количество страниц для быстрого теста
            chunks = parser.parse(pdf_path, max_pages=10)

            for chunk in chunks[: args.max_chunks]:
                all_chunks.append(chunk.text)
                all_metadatas.append(
                    {
                        "source_file": chunk.source_file,
                        "doc_title": chunk.doc_title,
                        "page_num": chunk.page_num,
                        "section": chunk.section or "",
                        "content": chunk.text,
                    }
                )

            if len(all_chunks) >= args.max_chunks:
                all_chunks = all_chunks[: args.max_chunks]
                all_metadatas = all_metadatas[: args.max_chunks]
                break

        logger.info(f"Извлечено {len(all_chunks)} чанков для индексации")

        if all_chunks:
            hipporag.index(all_chunks, _chunks_meta=all_metadatas)
            logger.info("✓ Тестовая индексация успешно завершена!")
        else:
            logger.warning("Нет чанков для индексации")

    except Exception as e:
        logger.error(f"Ошибка во время тестовой индексации: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    freeze_support()
    main()
