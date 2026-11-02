#!/usr/bin/env python3
"""
Astrology RAG - Локальный запуск с LM Studio (Gemma 4)
"""

import os
import sys
import argparse

# Добавляем путь
sys.path.insert(0, os.path.dirname(__file__))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig


# ============================================================================
# КОНФИГУРАЦИЯ - ИЗМЕНИТЕ ЗДЕСЬ ЕСЛИ НУЖНО
# ============================================================================
LM_STUDIO_URL = "http://localhost:1234/v1"
LM_MODEL = "gemma-4-e4b"
EMBEDDING_MODEL = "nvidia/NV-Embed-v2"
OUTPUT_DIR = "outputs/astrology_test"
# ============================================================================


def create_rag():
    """Создание системы RAG."""
    print("=" * 50)
    print("  ASTROLOGY RAG + LM STUDIO (Gemma 4)")
    print("=" * 50)
    print(f"LLM: {LM_MODEL}")
    print(f"URL: {LM_STUDIO_URL}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print()
    
    # Настройка конфигурации
    config = BaseConfig()
    config.llm_name = LM_MODEL
    config.llm_base_url = LM_STUDIO_URL
    config.embedding_model_name = EMBEDDING_MODEL
    config.save_dir = OUTPUT_DIR
    
    print("Создание системы...")
    try:
        hipporag = HippoRAG(global_config=config)
        print("✓ Система создана!")
        return hipporag
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return None


def test_lm_studio():
    """Проверка подключения к LM Studio."""
    try:
        import httpx
        url_base = LM_STUDIO_URL.replace('/v1', '')
        resp = httpx.get(f"{url_base}/models", timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('data', [])
            print(f"✓ LM Studio подключен! Моделей: {len(models)}")
            return True
    except Exception as e:
        print(f"✗ LM Studio недоступен: {e}")
    return False


def interactive_search(hipporag):
    """Интерактивный поиск."""
    print("\n" + "=" * 50)
    print("  ИНТЕРАКТИВНЫЙ ПОИСК")
    print("=" * 50)
    print("Введите 'выход' для завершения")
    print()
    
    while True:
        try:
            q = input("\nВопрос: ").strip()
            
            if q.lower() in ['выход', 'quit', 'exit', 'q']:
                print("До свидания!")
                break
            
            if not q:
                continue
            
            print("Поиск...")
            results = hipporag.rag_qa(queries=[q])
            r = results[0]
            
            print(f"\nОтвет: {r.answer}")
            
            if r.citations:
                print("\nИсточники:")
                for c in r.citations:
                    src = f"[{c.doc_title}"
                    if c.page_num:
                        src += f", стр. {c.page_num}"
                    src += "]"
                    print(f"  {src}")
                    
        except KeyboardInterrupt:
            print("\n\nВыход...")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Astrology RAG с LM Studio")
    parser.add_argument('--mode', choices=['index', 'search', 'both'], default='both',
                      help='Режим работы')
    parser.add_argument('--pdf_dir', default='data/pdfs',
                      help='Директория с PDF')
    args = parser.parse_args()
    
    # Проверка LM Studio
    if not test_lm_studio():
        print("""
Убедитесь что:
1. LM Studio запущен
2. Модель gemma-4-e4b загружена
3. Нажмите Start Server в LM Studio
        """)
        return
    
    # Создание RAG
    rag = create_rag()
    if rag is None:
        return
    
    # Индексация если нужно
    if args.mode in ['index', 'both']:
        pdf_dir = args.pdf_dir
        if os.path.exists(pdf_dir):
            from pathlib import Path
            pdfs = list(Path(pdf_dir).glob("*.pdf"))
            if pdfs:
                print(f"\nИндексация {len(pdfs)} PDF файлов...")
                pdf_paths = [str(p) for p in pdfs]
                rag.index_pdfs(pdf_paths)
                print("✓ Индексация завершена!")
    
    # Поиск если нужно
    if args.mode in ['search', 'both']:
        interactive_search(rag)


if __name__ == "__main__":
    main()