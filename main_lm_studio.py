#!/usr/bin/env python3
"""
Быстрый запуск Astrology RAG с LM Studio.
Использует локальную модель вместо OpenAI API.
"""

import os
import sys

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(__file__))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


# ==============================================================================
# КОНФИГУРАЦИЯ LM STUDIO
# ==============================================================================

# URL LM Studio сервера (по умолчанию localhost:1234)
LM_STUDIO_URL = "http://localhost:1234/v1"

# Модель Gemma 4 из LM Studio
# Примечание: используем имя без префикса "google/" чтобы совпадало с существующими индексами
LM_MODEL = "gemma-4-e4b"

# ==============================================================================


def create_hipporag_with_lm_studio():
    """Создание HippoRAG с настроенным LM Studio."""
    
    print("=" * 50)
    print("  ASTROLOGY RAG + LM STUDIO")
    print("=" * 50)
    print(f"URL: {LM_STUDIO_URL}")
    print(f"Model: gemma-4-e4b")
    print()
    
    # Создание конфигурации
    config = BaseConfig()
    config.llm_name = LM_MODEL
    config.llm_base_url = LM_STUDIO_URL
    config.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    config.save_dir = "outputs/astrology_test"
    
    print("Создание системы...")
    
    try:
        hipporag = HippoRAG(global_config=config)
        print("✓ Система создана!")
        return hipporag
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return None


def test_connection():
    """Проверка подключения к LM Studio."""
    try:
        import httpx
        url_base = LM_STUDIO_URL.replace('/v1', '')
        response = httpx.get(f"{url_base}/models", timeout=5.0)
        if response.status_code == 200:
            models = response.json()
            print(f"✓ LM Studio подключен! Моделей: {len(models.get('data', []))}")
            return True
        return False
    except Exception as e:
        print(f"✗ LM Studio не доступен: {e}")
        return False


def interactive_mode(hipporag):
    """Интерактивный режим поиска."""
    print("\n" + "=" * 50)
    print("  ИНТЕРАКТИВНЫЙ ПОИСК")
    print("=" * 50)
    print("Введите 'выход' для завершения")
    print()
    
    while True:
        try:
            question = input("\nВаш вопрос: ").strip()
            
            if question.lower() in ['выход', 'quit', 'exit', 'q']:
                print("До свидания!")
                break
            
            if not question:
                continue
            
            print("Поиск...")
            results = hipporag.rag_qa(queries=[question])
            result = results[0]
            
            print(f"\nОтвет: {result.answer}")
            
            if result.citations:
                print("\nЦитаты:")
                for c in result.citations:
                    source = f"[{c.doc_title}"
                    if c.page_num:
                        source += f", стр. {c.page_num}"
                    source += "]"
                    print(f"  {source}")
            
        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def main():
    """Главная функция."""
    # Проверка LM Studio
    print("Проверка LM Studio...")
    if not test_connection():
        print("""
Убедитесь что:
1. LM Studio запущен
2. Модель загружена и сервер запущен
3. URL соответствует настройкам в этом файле

Запустите LM Studio и нажмите Enter...
        """)
        input()
        
        if not test_connection():
            print("Не удается подключиться к LM Studio.")
            print("Проверьте настройки в файле main_lm_studio.py")
            return
    
    # Создание системы
    hipporag = create_hipporag_with_lm_studio()
    if hipporag is None:
        return
    
    # Интерактивный режим
    interactive_mode(hipporag)


if __name__ == "__main__":
    main()