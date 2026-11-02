#!/usr/bin/env python3
"""
Скрипт настройки LM Studio для Astrology RAG.
Использует LM Studio как локальный OpenAI-совместимый сервер.
"""

import os
import sys
import subprocess

def print_info():
    print("=" * 50)
    print("  НАСТРОЙКА LM STUDIO ДЛЯ ASTROLOGY RAG")
    print("=" * 50)
    print("""
Этот скрипт поможет настроить систему для работы с локальными LLM.

ТРЕБОВАНИЯ:
1. LM Studio (https://lmstudio.ai)
2. Скачанная модель (gemma, llama, mistral)

КАК ЗАПУСТИТЬ:

1. ОТКРОЙТЕ LM Studio
   - Скачайте модель (например @Google/Gemma-3-4b-It)
   - Выберите модель в списке
   - Нажмите кнопку "Start Server"
   - Скопируйте URL (по умолчанию http://localhost:1234/v1)

2. ВЕРНИТЕСЬ СЮДА И ВВЕДИТЕ URL

ПОДДЕРЖИВАЕМАЯ МОДЕЛЬ:
- gemma-4-e4b (модель из LM Studio)

ДЛЯ ВЫХОДА ВВЕДИТЕ: q
""")


def get_lm_studio_url():
    """Получение URL LM Studio от пользователя."""
    print("\nВведите URL LM Studio (по умолчанию: http://localhost:1234/v1)")
    url = input("URL: ").strip()
    
    if not url:
        url = "http://localhost:1234/v1"
    
    if url.lower() == 'q':
        return None
        
    # Добавляем /v1 если нет
    if '/v1' not in url:
        url = url.rstrip('/') + '/v1'
    
    return url


def get_model_name():
    """Получение имени модели."""
    print("\nВведите имя модели из LM Studio")
    print("По умолчанию: gemma-4-e4b")
    model = input("Модель: ").strip()
    
    if not model:
        model = "gemma-4-e4b"  # По умолчанию
    
    return model


def test_connection(url, model):
    """Проверка подключения к LM Studio."""
    print(f"\nПроверка подключения к {url}...")
    print(f"Модель: {model}")
    print()
    
    try:
        # Проверяем доступность
        import httpx
        response = httpx.get(url.replace('/v1', '/models'), timeout=5.0)
        if response.status_code == 200:
            print("✓ Подключение успешно!")
            models = response.json()
            print(f"Доступные модели: {len(models.get('data', []))}")
            return True
        else:
            print(f"✗ Ошибка подключения: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def create_config(url, model):
    """Создание конфигурации."""
    print("\nСоздание конфигурации...")
    
    # Записываем в файл конфигурации
    config_content = f'''# Конфигурация LM Studio для Astrology RAG
# Автоматически сгенерировано

LM_STUDIO_URL = "{url}"
LM_MODEL = "{model}"
EMBEDDING_MODEL = "nvidia/NV-Embed-v2"
'''
    
    with open('lm_studio_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✓ Конфигурация сохранена в lm_studio_config.py")
    return True


def main():
    """Главная функция."""
    print_info()
    
    while True:
        url = get_lm_studio_url()
        if url is None:
            print("До свидания!")
            break
        
        model = get_model_name()
        
        if test_connection(url, model):
            create_config(url, model)
            print("\n" + "=" * 50)
            print("  НАСТРОЙКА ЗАВЕРШЕНА!")
            print("=" * 50)
            print(f"""
Следующие шаги:

1. Убедитесь что LM Studio сервер запущен
2. Запустите индексацию:
   python index_pdfs.py --pdf_dir data/pdfs

3. Запустите поиск:
   python search_interactive.py

КОНФИГУРАЦИЯ СОХРАНЕНА: lm_studio_config.py
            """)
            break
        
        print("\nПопробуйте снова или введите 'q' для выхода.")


if __name__ == "__main__":
    main()