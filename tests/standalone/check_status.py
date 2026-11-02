#!/usr/bin/env python3
"""
Скрипт для проверки статуса индексации и диагностики проблем.
"""

import json
import os
import sys
from pathlib import Path


def check_index_status():
    """Проверить статус индексации."""
    print("=" * 60)
    print("ПРОВЕРКА СТАТУСА ИНДЕКСАЦИИ")
    print("=" * 60)

    # Проверка директорий
    output_dirs = ["outputs/astrology_test", "outputs/astrology_test"]

    for output_dir in output_dirs:
        print(f"\n📁 Директория: {output_dir}")
        if not os.path.exists(output_dir):
            print("   ❌ Не существует")
            continue

        print("   ✓ Существует")

        # Проверка поддиректорий
        subdirs = list(Path(output_dir).glob("*"))
        if subdirs:
            for subdir in subdirs:
                if subdir.is_dir():
                    print(f"   └─ {subdir.name}")

                    # Проверка файлов индекса
                    graph_file = subdir / "graph.pickle"
                    if graph_file.exists():
                        size_mb = graph_file.stat().st_size / (1024 * 1024)
                        print(f"      ✓ graph.pickle ({size_mb:.2f} MB)")

                    # Проверка embeddings
                    for emb_type in ["chunk_embeddings", "entity_embeddings", "fact_embeddings"]:
                        emb_dir = subdir / emb_type
                        if emb_dir.exists():
                            parquet_file = emb_dir / f"vdb_{emb_type.split('_')[0]}.parquet"
                            if parquet_file.exists():
                                size_mb = parquet_file.stat().st_size / (1024 * 1024)
                                print(f"      ✓ {emb_type} ({size_mb:.2f} MB)")

    # Проверка OpenIE результатов
    print("\n📊 OpenIE результаты:")
    openie_files = list(Path("outputs").rglob("openie_results_*.json"))
    if openie_files:
        for openie_file in openie_files:
            print(f"   ✓ {openie_file}")
            try:
                with open(openie_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    num_docs = len(data.get("docs", []))
                    print(f"      Документов: {num_docs}")
            except Exception as e:
                print(f"      ⚠️ Ошибка чтения: {e}")
    else:
        print("   ❌ Не найдено")

    # Проверка логов
    print("\n📝 Логи:")
    log_files = ["indexing.log", "indexing_test.log"]
    for log_file in log_files:
        if os.path.exists(log_file):
            size_kb = os.path.getsize(log_file) / 1024
            print(f"   ✓ {log_file} ({size_kb:.2f} KB)")
        else:
            print(f"   ❌ {log_file} не найден")

    # Проверка PDF файлов
    print("\n📄 PDF файлы:")
    pdf_dir = Path("data/pdfs")
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"   Найдено PDF файлов: {len(pdf_files)}")
        for pdf_file in pdf_files:
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            print(f"   - {pdf_file.name} ({size_mb:.2f} MB)")
    else:
        print("   ❌ Директория data/pdfs не найдена")

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 60)

    # Проверка наличия индекса
    has_index = any(
        (Path(d) / subdir / "graph.pickle").exists()
        for d in output_dirs
        for subdir in Path(d).glob("*")
        if subdir.is_dir()
    )

    if has_index:
        print("\n✓ Индекс найден. Система готова к работе.")
        print("\nДля поиска используйте:")
        print("  python search_interactive.py")
        print("  python quick_search.py")
    else:
        print("\n⚠️ Индекс не найден. Необходима индексация.")
        print("\nДля быстрого теста:")
        print("  python index_pdfs_test.py --max_pdfs 1 --max_chunks 20")
        print("\nДля полной индексации:")
        print("  python index_pdfs.py")
        print("\n⚠️ ВНИМАНИЕ: Полная индексация может занять несколько часов!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    check_index_status()
