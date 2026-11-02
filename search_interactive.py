#!/usr/bin/env python3
"""
Интерактивный скрипт поиска для астрологической поисковой системы.
"""

import os
import sys
import logging

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.lm_studio_manager import setup_environment


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_citations(citations):
    """Вывод цитат в удобочитаемом формате."""
    if not citations:
        return
    
    print("\nЦитаты из источников:")
    for i, citation in enumerate(citations, 1):
        source_info = f"[{citation.doc_title}"
        if citation.page_num:
            source_info += f", стр. {citation.page_num}"
        if citation.section:
            source_info += f", секция: {citation.section}"
        source_info += "]"
        print(f"  {i}. {source_info}")


def main():
    """Главная функция интерактивного поиска."""
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Настройка конфигурации
        config = BaseConfig()
        config.save_dir = "outputs/astrology_test"
        config.dataset = "astrology"  # Set dataset name to use astrology prompt template
        
        # ALWAYS use local LM Studio
        import os
        config.llm_name = "gemma-4-e4b"  # Match indexed directory name
        config.llm_base_url = "http://172.18.0.1:1234/v1"  # Force local LLM
        # Use MULTILINGUAL embedding model for better Russian support
        config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        
        # Also set env vars for any subprocess that reads them
        os.environ["LLM_NAME"] = "gemma-4-e4b"
        os.environ["LLM_BASE_URL"] = "http://172.18.0.1:1234/v1"
        
        logger.info(f"Using LOCAL LLM: {config.llm_name} at {config.llm_base_url}")
        
        # Автоматическая настройка окружения: запуск LM Studio и загрузка моделей
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
            logger.warning("LM Studio не готов. Продолжение без генерации ответов (только поиск).")
        
        # Создание экземпляра HippoRAG
        logger.info("Загрузка индексов...")
        hipporag = HippoRAG(global_config=config)
        logger.info("Система готова к поиску!")
        
        print("\n=== Интерактивный поиск в астрологических документах ===")
        print("Введите 'выход' или 'quit' для завершения работы")
        print("=" * 55)
        
        while True:
            try:
                # Получаем вопрос от пользователя
                question = input("\nВведите ваш вопрос по астрологии: ").strip()
                
                # Проверяем команды выхода
                if question.lower() in ['выход', 'quit', 'exit', '']:
                    print("Спасибо за использование системы поиска!")
                    break
                
                # Выполняем поиск
                print(f"\nОбработка запроса: {question}")
                results = hipporag.rag_qa(queries=[question])
                
                # results - это tuple (queries_solutions, messages, metadata)
                queries_solutions = results[0]
                result = queries_solutions[0]
                
                print(f"\nОтвет: {result.answer}")
                print_citations(result.citations)
                
                # Выводим первые несколько релевантных документов
                if result.docs:
                    print(f"\nРелевантные фрагменты документов ({min(3, len(result.docs))} из {len(result.docs)}):")
                    for i, doc in enumerate(result.docs[:3]):
                        snippet = doc[:200] + "..." if len(doc) > 200 else doc
                        print(f"  {i+1}. {snippet}")
                        
            except KeyboardInterrupt:
                print("\n\nПолучен сигнал прерывания. Завершение работы...")
                break
            except EOFError:
                print("\nВвод завершен. До свидания!")
                break
            except Exception as e:
                logger.error(f"Ошибка при обработке запроса: {str(e)}")
                print(f"Произошла ошибка: {str(e)}")
                print("Пожалуйста, попробуйте другой запрос.")
                
    except Exception as e:
        logger.error(f"Ошибка инициализации системы: {str(e)}")
        print(f"Не удалось инициализировать систему поиска: {str(e)}")
        print("Убедитесь, что индексация была выполнена успешно.")


if __name__ == "__main__":
    main()