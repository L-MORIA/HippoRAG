#!/usr/bin/env python3
"""
Тестирование системы Astrology RAG с LM Studio
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Проверка импортов."""
    print("=" * 50)
    print("  ТЕСТ 1: ИМПОРТЫ")
    print("=" * 50)

    try:
        from hipporag import HippoRAG

        print("✓ HippoRAG импортирован")
    except Exception as e:
        print(f"✗ HippoRAG: {e}")
        return False

    try:
        from hipporag.utils.config_utils import BaseConfig

        print("✓ BaseConfig импортирован")
    except Exception as e:
        print(f"✗ BaseConfig: {e}")
        return False

    try:
        from hipporag.pdf_ingestion import TextPDFParser

        print("✓ TextPDFParser импортирован")
    except Exception as e:
        print(f"✗ TextPDFParser: {e}")
        return False

    return True


def test_config():
    """Проверка конфигурации."""
    print("\n" + "=" * 50)
    print("  ТЕСТ 2: КОНФИГУРАЦИЯ")
    print("=" * 50)

    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig()
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.save_dir = "outputs/test"

    print(f"LLM: {config.llm_name}")
    print(f"URL: {config.llm_base_url}")
    print(f"Embedding: {config.embedding_model_name}")
    print(f"Output: {config.save_dir}")
    print("✓ Конфигурация создана")

    return config


def test_lm_studio():
    """Проверка LM Studio."""
    print("\n" + "=" * 50)
    print("  ТЕСТ 3: LM STUDIO")
    print("=" * 50)

    try:
        import httpx

        r = httpx.get("http://localhost:1234/v1/models", timeout=5)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            print(f"✓ LM Studio подключен!")
            print(f"  Моделей доступно: {len(models)}")
            return True
    except Exception as e:
        print(f"✗ LM Studio недоступен: {e}")

    return False


def test_rag_creation(config):
    """Проверка создания RAG."""
    print("\n" + "=" * 50)
    print("  ТЕСТ 4: СОЗДАНИЕ RAG")
    print("=" * 50)

    try:
        from hipporag import HippoRAG

        rag = HippoRAG(global_config=config)
        print("✓ HippoRAG создан!")
        return rag
    except Exception as e:
        print(f"✗ Ошибка создания: {e}")
        return None


def main():
    """Главная функция тестирования."""
    print("\n" + "#" * 50)
    print("#  ТЕСТИРОВАНИЕ ASTROLOGY RAG")
    print("#" * 50 + "\n")

    results = []

    # Тест 1: Импорты
    results.append(("Импорты", test_imports()))

    if not results[-1][1]:
        print("\n✗ Критическая ошибка - импорты не работают!")
        return

    # Тест 2: Конфигурация
    config = test_config()
    results.append(("Конфигурация", config is not None))

    # Тест 3: LM Studio
    results.append(("LM Studio", test_lm_studio()))

    # Тест 4: Создание RAG (только если LM Studio работает)
    if results[2][1]:
        results.append(("Создание RAG", test_rag_creation(config) is not None))

    # Итоги
    print("\n" + "=" * 50)
    print("  РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)

    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}: {'OK' if result else 'ОШИБКА'}")

    print()
    all_ok = all(r[1] for r in results)
    if all_ok:
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    else:
        print("✗ ЕСТЬ ОШИБКИ - проверьте вывод выше")


if __name__ == "__main__":
    main()
