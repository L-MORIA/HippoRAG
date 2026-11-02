#!/usr/bin/env python3
"""
Скрипт поиска по списку заранее подготовленных вопросов.
"""

import os
import sys
import logging
import json
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("search.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_questions_from_file(filepath):
    """Загрузка вопросов из текстового файла."""
    if not os.path.exists(filepath):
        return None
    
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Игнорируем пустые строки и комментарии
                questions.append(line)
    
    return questions


def save_results_to_file(results, output_file):
    """Сохранение результатов в файл."""
    output_data = []
    for result in results:
        output_data.append({
            "question": result.question,
            "answer": result.answer,
            "citations": [
                {
                    "source_file": citation.source_file,
                    "doc_title": citation.doc_title,
                    "page_num": citation.page_num,
                    "section": citation.section
                }
                for citation in result.citations
            ]
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def main():
    """Главная функция поиска по вопросам."""
    # Настройка логирования
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Стандартные вопросы по астрологии
    default_questions = [
        "Что означает Марс в Овне?",
        "Как влияет Венера в Тельце на личностные отношения?",
        "Какие аспекты важны для анализа гороскопа?",
        "Что такое натальный карты и как их интерпретировать?",
        "Какие планеты отвечают за карьеру в астрологии?",
        "Что означает соединение Юпитера и Сатурна?",
        "Как влияет Луна в разных знаках зодиака на характер человека?",
        "Какие дома отвечают за брак и семейную жизнь?"
    ]
    
    try:
        # Попытка загрузить вопросы из файла
        questions = load_questions_from_file("data/questions.txt")
        if questions is None:
            logger.info("Файл с вопросами не найден, используем стандартные вопросы")
            questions = default_questions
        else:
            logger.info(f"Загружено {len(questions)} вопросов из файла")
        
        # Настройка конфигурации
        config = BaseConfig()
        config.save_dir = "outputs/astrology_test"
        
        # Создание экземпляра HippoRAG
        logger.info("Загрузка индексов...")
        hipporag = HippoRAG(global_config=config)
        logger.info("Система готова к поиску!")
        
        # Выполняем поиск по всем вопросам
        logger.info(f"Выполнение поиска по {len(questions)} вопросам...")
        results = hipporag.rag_qa(queries=questions)
        logger.info("Поиск завершен!")
        
        # Выводим результаты
        print("\n" + "=" * 60)
        print("Результаты поиска")
        print("=" * 60)
        
        for i, result in enumerate(results):
            print(f"\nВопрос {i+1}: {result.question}")
            print(f"Ответ: {result.answer}")
            
            if result.citations:
                print("Цитаты из источников:")
                for j, citation in enumerate(result.citations):
                    source_info = f"[{citation.doc_title}"
                    if citation.page_num:
                        source_info += f", стр. {citation.page_num}"
                    if citation.section:
                        source_info += f", секция: {citation.section}"
                    source_info += "]"
                    print(f"  {j+1}. {source_info}")
        
        # Сохраняем результаты в файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{timestamp}.json"
        save_results_to_file(results, output_file)
        logger.info(f"Результаты сохранены в файл: {output_file}")
        
    except Exception as e:
        logger.error(f"Ошибка во время поиска: {str(e)}")
        raise


if __name__ == "__main__":
    main()