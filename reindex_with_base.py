"""
Переиндексация с multilingual-e5-base моделью (лучшее качество)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

def main():
    print("=" * 80)
    print("ПЕРЕИНДЕКСАЦИЯ С MULTILINGUAL-E5-BASE")
    print("=" * 80)
    
    # Настройка конфигурации с base моделью (лучшее качество)
    config = BaseConfig()
    config.save_dir = "outputs/astrology_base"
    config.llm_name = "gemma-4-e4b"
    config.llm_base_url = "http://localhost:1234/v1"
    config.embedding_model_name = "intfloat/multilingual-e5-base"  # Base модель - лучше качество
    config.retrieval_top_k = 5
    
    print(f"\n✓ Конфигурация:")
    print(f"  - LLM: {config.llm_name}")
    print(f"  - Embedding модель: {config.embedding_model_name}")
    print(f"  - Размерность эмбеддингов: 768 (vs 384 у small)")
    print(f"  - Директория: {config.save_dir}")
    
    # Путь к исходным данным
    data_dir = Path("data/pdfs")
    
    if not data_dir.exists():
        print(f"\n✗ Директория с данными не найдена: {data_dir}")
        return
    
    print(f"\n✓ Директория с данными: {data_dir}")
    
    # Получаем список файлов для индексации
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n✗ PDF файлы не найдены в {data_dir}")
        return
    
    print(f"\n✓ Найдено PDF файлов для индексации: {len(pdf_files)}")
    for f in pdf_files:
        print(f"  - {f.name}")
    
    # Инициализация HippoRAG
    print("\nИнициализация HippoRAG...")
    hipporag = HippoRAG(global_config=config)
    
    # Индексация
    print("\n" + "=" * 80)
    print("НАЧАЛО ИНДЕКСАЦИИ")
    print("=" * 80)
    print("\n⚠ Это займет больше времени чем с small моделью, но качество будет выше!")
    
    try:
        hipporag.index_pdfs([str(f) for f in pdf_files], ocr_config=None)
        
        print("\n" + "=" * 80)
        print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        print("=" * 80)
        print(f"\n📊 Статистика:")
        print(f"  - Проиндексировано чанков: {len(hipporag.chunk_embedding_store.hash_ids)}")
        print(f"  - Извлечено entities: {len(hipporag.entity_embedding_store.hash_ids)}")
        print(f"  - Извлечено facts: {len(hipporag.fact_embedding_store.hash_ids)}")
        
    except Exception as e:
        print(f"\n✗ Ошибка при индексации: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
