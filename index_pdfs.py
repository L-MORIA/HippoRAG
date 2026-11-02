#!/usr/bin/env python3
"""
Скрипт индексации PDF-документов для астрологической поисковой системы.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig, build_working_dir
from hipporag.utils.lm_studio_manager import setup_environment
from multiprocessing import freeze_support


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("indexing.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def clean_old_index(output_dir, llm_name, embedding_model):
    """
    Очистить старые данные индекса перед новой индексацией.
    Удаляет parquet файлы и кэш NER, чтобы избежать загрузки старых мусорных entities.
    """
    import shutil
    
    # Путь к рабочей директории индекса — единая формула с ядром (S-22)
    working_dir = build_working_dir(output_dir, llm_name, embedding_model)
    
    logger = logging.getLogger(__name__)
    
    if os.path.exists(working_dir):
        logger.info(f"🧹 Очистка старого индекса: {working_dir}")
        try:
            shutil.rmtree(working_dir)
            logger.info("   ✅ Старый индекс удалён")
        except Exception as e:
            logger.warning(f"   ⚠️  Не удалось удалить старый индекс: {e}")
    
    # Удаляем кэш NER
    ner_cache = os.path.join(output_dir, f"openie_results_ner_{llm_name}.json")
    if os.path.exists(ner_cache):
        logger.info(f"🧹 Очистка кэша NER: {ner_cache}")
        try:
            os.remove(ner_cache)
            logger.info("   ✅ Кэш NER удалён")
        except Exception as e:
            logger.warning(f"   ⚠️  Не удалось удалить кэш NER: {e}")


def find_pdf_files(directory):
    """Найти все PDF-файлы в директории и поддиректориях."""
    pdf_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Директория {directory} не существует")
    
    for pdf_file in directory_path.rglob("*.pdf"):
        pdf_files.append(str(pdf_file.resolve()))
    
    return pdf_files


def main():
    """Главная функция индексации."""
    parser = argparse.ArgumentParser(description="Индексация PDF-документов для Astrology RAG")
    parser.add_argument(
        "--pdf_dir",
        default="data/pdfs",
        help="Директория с PDF-файлами для индексации (по умолчанию: data/pdfs)"
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
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Проверяем наличие директории с PDF
        if not os.path.exists(args.pdf_dir):
            logger.info(f"Создание директории для PDF: {args.pdf_dir}")
            os.makedirs(args.pdf_dir, exist_ok=True)
            logger.info("Пожалуйста, поместите ваши PDF-файлы в директорию и запустите скрипт заново")
            return
        
        # Ищем PDF-файлы
        pdf_files = find_pdf_files(args.pdf_dir)
        
        if not pdf_files:
            logger.warning(f"В директории {args.pdf_dir} не найдено PDF-файлов")
            logger.info("Попытка индексации из JSON файла...")
            
            # Пробуем загрузить из JSON
            json_path = os.path.join(args.pdf_dir, "test_astrology.json")
            if os.path.exists(json_path):
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                corpus = data.get("corpus", [])
                if not corpus:
                    logger.warning("В JSON нет данных corpus")
                    return
                    
                logger.info(f"Загружено {len(corpus)} документов из JSON")
                
                config = BaseConfig()
                config.save_dir = args.output_dir
                config.llm_name = os.getenv('LLM_NAME', 'gemma-4-e4b')
                config.llm_base_url = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
                # Use MULTILINGUAL embedding model for better Russian support
                config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                
                # Очистка старых данных перед индексацией
                logger.info("\n" + "="*60)
                logger.info("Очистка старых данных индекса...")
                logger.info("="*60)
                clean_old_index(
                    output_dir=config.save_dir,
                    llm_name=config.llm_name,
                    embedding_model=config.embedding_model_name
                )
                
                # Автоматическая настройка окружения
                logger.info("\n" + "="*60)
                logger.info("Проверка и настройка окружения...")
                logger.info("="*60)
                lm_ready, embedding_ready = setup_environment(
                    llm_model_name=config.llm_name,
                    embedding_model_name=config.embedding_model_name,
                    lm_studio_url=config.llm_base_url.replace('/v1', '')
                )
                
                if not embedding_ready:
                    logger.error("Не удалось загрузить embedding модель. Завершение работы.")
                    return
                
                if not lm_ready:
                    logger.warning("LM Studio не готов. Продолжение с ограниченной функциональностью.")
                
                hipporag = HippoRAG(global_config=config)
                
                texts = [doc.get("text", "") for doc in corpus]
                metadatas = [
                    {
                        "source_file": doc.get("title", ""),
                        "doc_title": doc.get("title", ""),
                        "page_num": 0,
                        "section": "",
                        "content": doc.get("text", "")
                    }
                    for doc in corpus
                ]
                
                logger.info("Индексация документов...")
                hipporag.index(texts, _chunks_meta=metadatas)
                logger.info("Индексация успешно завершена!")
                return
            else:
                logger.warning(f"JSON файл {json_path} не найден")
                return
        
        logger.info(f"Найдено {len(pdf_files)} PDF-файлов для индексации")
        for pdf in pdf_files:
            logger.info(f"  - {pdf}")
        
        # Настройка конфигурации
        config = BaseConfig()
        config.save_dir = args.output_dir
        config.llm_name = os.getenv('LLM_NAME', 'gemma-4-e4b')
        config.llm_base_url = os.getenv('LLM_BASE_URL', 'http://localhost:1234/v1')
        # Use MULTILINGUAL embedding model for better Russian support
        config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        
        # Очистка старых данных перед индексацией
        logger.info("\n" + "="*60)
        logger.info("Очистка старых данных индекса...")
        logger.info("="*60)
        clean_old_index(
            output_dir=config.save_dir,
            llm_name=config.llm_name,
            embedding_model=config.embedding_model_name
        )
        
        # Автоматическая настройка окружения
        logger.info("\n" + "="*60)
        logger.info("Проверка и настройка окружения...")
        logger.info("="*60)
        lm_ready, embedding_ready = setup_environment(
            llm_model_name=config.llm_name,
            embedding_model_name=config.embedding_model_name,
            lm_studio_url=config.llm_base_url.replace('/v1', '')
        )
        
        if not embedding_ready:
            logger.error("Не удалось загрузить embedding модель. Завершение работы.")
            return
        
        if not lm_ready:
            logger.warning("LM Studio не готов. Продолжение с ограниченной функциональностью.")
        
        # Создание экземпляра HippoRAG
        hipporag = HippoRAG(global_config=config)
        
        # Настройка OCR при необходимости
        ocr_config = None
        if args.enable_ocr:
            ocr_config = {
                "enabled": True,
                "language": args.ocr_language,
                "dpi": 300,
                "engine_mode": 1
            }
            logger.info(f"OCR включен с языком: {args.ocr_language}")
        
        # Индексация PDF-файлов
        logger.info("Начало индексации документов...")
        hipporag.index_pdfs(pdf_files, ocr_config=ocr_config)
        logger.info("Индексация успешно завершена!")
        
    except Exception as e:
        logger.error(f"Ошибка во время индексации: {str(e)}")
        raise


if __name__ == "__main__":
    freeze_support()
    main()