#!/usr/bin/env python3
"""
Тест нового семантического чанкера.
Сравнивает качество старого и нового подходов.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hipporag.pdf_ingestion.chunker import ChunkConfig, chunk_text_with_metadata
from hipporag.pdf_ingestion.semantic_chunker import (
    SemanticChunkConfig,
    chunk_text_semantic,
    clean_pdf_text,
)


def test_text_cleaning():
    """Тест очистки текста."""
    print("=" * 80)
    print("ТЕСТ ОЧИСТКИ ТЕКСТА")
    print("=" * 80)

    # Примеры проблемного текста из PDF
    test_cases = [
        # Слипшиеся слова
        ("словоСлово другоеСлово", "слово Слово другое Слово"),
        # Дефисы переноса
        ("слово-\nслово", "словослово"),
        # Переносы внутри предложения
        ("Это первая\nстрока предложения.", "Это первая строка предложения."),
        # Множественные пробелы
        ("слово    слово", "слово слово"),
    ]

    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = clean_pdf_text(input_text)
        status = "✓" if result == expected else "✗"
        print(f"\nТест {i}: {status}")
        print(f"  Вход:    '{input_text}'")
        print(f"  Ожидал:  '{expected}'")
        print(f"  Получил: '{result}'")


def test_chunking_comparison():
    """Сравнение старого и нового чанкера."""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ СТАРОГО И НОВОГО ЧАНКЕРА")
    print("=" * 80)

    # Пример текста с проблемами
    test_text = """»ябудуназыватьтехлюдей,которыепренебрега-
ют преимуществами своего гороскопа и проявляют худшие
качествасвоегознака. Если исходить из посы-
ла, что точка силы Овна — это прихожая и коридор в ви-
де холла, то начинать выбор квартиры ему надо именно
с этого элемента. Если есть возможность пробить стену,
снять дверь в комнате или на кухне и поменять её на ар-
ку, пусть эта перепланировка будет выполнена."""

    print("\nИсходный текст:")
    print("-" * 80)
    print(test_text[:200] + "...")

    # Старый чанкер
    print("\n" + "=" * 80)
    print("СТАРЫЙ ЧАНКЕР (механическое разделение)")
    print("=" * 80)

    old_chunks = chunk_text_with_metadata(
        text=test_text,
        source_file="test.pdf",
        doc_title="Test",
        page_num=1,
        config=ChunkConfig(chunk_size_tokens=50, chunk_overlap_tokens=10),
    )

    for i, chunk in enumerate(old_chunks, 1):
        print(f"\nЧанк {i} (длина: {len(chunk.text)} символов):")
        print("-" * 80)
        print(chunk.text[:200])

    # Новый чанкер
    print("\n" + "=" * 80)
    print("НОВЫЙ СЕМАНТИЧЕСКИЙ ЧАНКЕР")
    print("=" * 80)

    new_chunks = chunk_text_semantic(
        text=test_text,
        source_file="test.pdf",
        doc_title="Test",
        page_num=1,
        config=SemanticChunkConfig(chunk_size_tokens=50, chunk_overlap_tokens=10),
    )

    for i, chunk in enumerate(new_chunks, 1):
        print(f"\nЧанк {i} (длина: {len(chunk.text)} символов):")
        print("-" * 80)
        print(chunk.text[:200])

    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ")
    print("=" * 80)
    print(f"Старый чанкер: {len(old_chunks)} чанков")
    print(f"Новый чанкер:  {len(new_chunks)} чанков")


def test_real_pdf():
    """Тест на реальном PDF файле."""
    print("\n" + "=" * 80)
    print("ТЕСТ НА РЕАЛЬНОМ PDF")
    print("=" * 80)

    pdf_path = "data/pdfs/978-5-0065-3635-7_2.pdf"

    if not os.path.exists(pdf_path):
        print(f"⚠ PDF файл не найден: {pdf_path}")
        return

    from hipporag.pdf_ingestion.text_pdf_parser import TextPDFParser

    parser = TextPDFParser()

    print(f"\nПарсинг PDF: {pdf_path}")
    print("Используется новый семантический чанкер...")

    try:
        chunks = parser.parse(pdf_path)

        print(f"\n✓ Создано чанков: {len(chunks)}")

        # Показываем первые 3 чанка
        print("\nПримеры чанков:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n{'='*80}")
            print(f"Чанк {i} (страница {chunk.page_num}, длина: {len(chunk.text)} символов)")
            print(f"{'='*80}")
            print(chunk.text[:400])

            # Проверяем качество
            issues = []
            if chunk.text[0].islower() and i > 1:
                issues.append("⚠ Начинается со строчной буквы (возможно обрывок)")
            if not chunk.text[-1] in ".!?":
                issues.append("⚠ Не заканчивается знаком препинания")
            if "  " in chunk.text:
                issues.append("⚠ Множественные пробелы")

            if issues:
                print("\nПроблемы:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("\n✓ Чанк выглядит хорошо")

    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Главная функция."""
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ СЕМАНТИЧЕСКОГО ЧАНКЕРА")
    print("=" * 80)

    test_text_cleaning()
    test_chunking_comparison()
    test_real_pdf()

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    print("\nРекомендации:")
    print("1. Если тесты прошли успешно, запустите переиндексацию:")
    print("   python index_pdfs.py --output_dir outputs/astrology_semantic")
    print("2. Сравните качество поиска с новыми чанками")


if __name__ == "__main__":
    main()
