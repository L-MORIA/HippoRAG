#!/usr/bin/env python3
"""
Быстрый запуск Astrology RAG
"""
import sys
sys.path.insert(0, '.')

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Конфигурация
config = BaseConfig()
config.save_dir = "outputs/astrology_test"
config.llm_name = "gemma-4-e4b"
config.llm_base_url = "http://localhost:1234/v1"
config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Создание системы
print("Загрузка RAG...")
rag = HippoRAG(global_config=config)
print("Готово!\n")

# Поиск
while True:
    q = input("Вопрос: ").strip()
    if not q:
        break
    
    results = rag.rag_qa([q])
    answers = results[0][0]
    print(f"\nОтвет: {answers.answer}\n")
    print(f"Цитат: {len(answers.citations)}\n")