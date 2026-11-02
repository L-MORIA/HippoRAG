#!/usr/bin/env python3
"""
Скрипт для удобного добавления и индексации новых PDF документов.
Автоматизирует процесс: копирование → индексация → проверка
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def setup_logging():
    """Настройка логирования."""
    log_file = f"add_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def copy_pdfs_to_data_dir(source_paths, data_dir="data/pdfs"):
    """
    Копирует PDF файлы в директорию data/pdfs.
    
    Args:
        source_paths: Список путей к PDF файлам или директориям
        data_dir: Целевая директория
    
    Returns:
        Список скопированных файлов
    """
    logger = logging.getLogger(__name__)
    
    os.makedirs(data_dir, exist_ok=True)
    copied_files = []
    
    for source in source_paths:
        source_path = Path(source)
        
        if not source_path.exists():
            logger.warning(f"Файл/директория не найдена: {source}")
            continue
        
        if source_path.is_file() and source_path.suffix.lower() == '.pdf':
            # Копируем отдельный PDF файл
            dest = Path(data_dir) / source_path.name
            shutil.copy2(source_path, dest)
            logger.info(f"Скопирован: {source} → {dest}")
            copied_files.append(str(dest))
        
        elif source_path.is_dir():
            # Копируем все PDF из директории
            for pdf_file in source_path.rglob("*.pdf"):
                # Сохраняем структуру поддиректорий
                relative_path = pdf_file.relative_to(source_path)
                dest_dir = Path(data_dir) / relative_path.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest = dest_dir / pdf_file.name
                shutil.copy2(pdf_file, dest)
                logger.info(f"Скопирован: {pdf_file} → {dest}")
                copied_files.append(str(dest))
    
    return copied_files


def count_pdfs(data_dir="data/pdfs"):
    """Подсчитывает количество PDF файлов в директории."""
    if not os.path.exists(data_dir):
        return 0
    
    pdf_files = list(Path(data_dir).rglob("*.pdf"))
    return len(pdf_files)


def get_index_size(output_dir="outputs/astrology"):
    """Получает размер индексов в МБ."""
    if not os.path.exists(output_dir):
        return 0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(output_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # Конвертируем в МБ


def index_documents(pdf_dir="data/pdfs", output_dir="outputs/astrology", 
                   enable_ocr=False, ocr_language="eng+rus"):
    """
    Индексирует PDF документы.
    
    Args:
        pdf_dir: Директория с PDF файлами
        output_dir: Директория для сохранения индексов
        enable_ocr: Включить OCR
        ocr_language: Язык для OCR
    """
    logger = logging.getLogger(__name__)
    
    # Проверяем наличие PDF файлов
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"В директории {pdf_dir} не найдено PDF файлов")
        return False
    
    logger.info(f"Найдено {len(pdf_files)} PDF файлов для индексации")
    
    try:
        # Настройка конфигурации
        config = BaseConfig()
        config.save_dir = output_dir
        config.llm_name = "gemma-4-e4b"
        config.llm_base_url = "http://172.18.0.1:1234/v1"
        config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Создание экземпляра HippoRAG
        logger.info("Инициализация HippoRAG...")
        hipporag = HippoRAG(global_config=config)
        
        # Настройка OCR при необходимости
        ocr_config = None
        if enable_ocr:
            ocr_config = {
                "enabled": True,
                "language": ocr_language,
                "dpi": 300,
                "engine_mode": 1
            }
            logger.info(f"OCR включен с языком: {ocr_language}")
        
        # Индексация PDF файлов
        logger.info("Начало индексации документов...")
        hipporag.index_pdfs([str(f) for f in pdf_files], ocr_config=ocr_config)
        
        logger.info("✅ Индексация успешно завершена!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время индексации: {str(e)}")
        return False


def print_summary(pdf_dir="data/pdfs", output_dir="outputs/astrology"):
    """Выводит сводку по индексам."""
    logger = logging.getLogger(__name__)
    
    pdf_count = count_pdfs(pdf_dir)
    index_size = get_index_size(output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("СВОДКА")
    logger.info("="*60)
    logger.info(f"PDF файлов в {pdf_dir}: {pdf_count}")
    logger.info(f"Размер индексов: {index_size:.2f} МБ")
    logger.info(f"Директория индексов: {output_dir}")
    logger.info("="*60 + "\n")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Добавление и индексация PDF документов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Индексировать всё в data/pdfs/
  python add_documents.py

  # Скопировать PDF из другой папки и индексировать
  python add_documents.py --source /path/to/pdfs

  # Скопировать несколько файлов
  python add_documents.py --source file1.pdf file2.pdf

  # С OCR для сканированных документов
  python add_documents.py --enable_ocr --ocr_language rus

  # Только скопировать, без индексации
  python add_documents.py --source /path/to/pdfs --no_index

  # Только индексировать, без копирования
  python add_documents.py --no_copy
        """
    )
    
    parser.add_argument(
        "--source",
        nargs="+",
        help="Путь(и) к PDF файлам или директориям для копирования"
    )
    parser.add_argument(
        "--pdf_dir",
        default="data/pdfs",
        help="Директория с PDF файлами (по умолчанию: data/pdfs)"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/astrology",
        help="Директория для сохранения индексов (по умолчанию: outputs/astrology)"
    )
    parser.add_argument(
        "--enable_ocr",
        action="store_true",
        help="Включить OCR для сканированных документов"
    )
    parser.add_argument(
        "--ocr_language",
        default="eng+rus",
        help="Язык OCR (по умолчанию: eng+rus)"
    )
    parser.add_argument(
        "--no_copy",
        action="store_true",
        help="Не копировать файлы, только индексировать"
    )
    parser.add_argument(
        "--no_index",
        action="store_true",
        help="Не индексировать, только копировать файлы"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("ДОБАВЛЕНИЕ И ИНДЕКСАЦИЯ PDF ДОКУМЕНТОВ")
    logger.info("="*60)
    
    try:
        # Шаг 1: Копирование файлов
        if not args.no_copy and args.source:
            logger.info("\n📋 Шаг 1: Копирование PDF файлов...")
            copied = copy_pdfs_to_data_dir(args.source, args.pdf_dir)
            if copied:
                logger.info(f"✅ Скопировано {len(copied)} файлов")
            else:
                logger.warning("⚠️  Не удалось скопировать файлы")
        
        # Шаг 2: Индексация
        if not args.no_index:
            logger.info("\n🔍 Шаг 2: Индексация документов...")
            success = index_documents(
                args.pdf_dir,
                args.output_dir,
                args.enable_ocr,
                args.ocr_language
            )
            
            if success:
                logger.info("✅ Индексация завершена успешно")
            else:
                logger.error("❌ Индексация не удалась")
                return 1
        
        # Вывод сводки
        print_summary(args.pdf_dir, args.output_dir)
        
        logger.info("✅ Все операции завершены успешно!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
