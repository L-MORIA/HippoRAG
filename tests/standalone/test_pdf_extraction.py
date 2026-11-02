#!/usr/bin/env python3
"""
Сравнение качества извлечения текста из PDF разными библиотеками.
"""

import os


def test_pdfplumber():
    """Тест pdfplumber (текущий)."""
    print("=" * 80)
    print("PDFPLUMBER (текущий)")
    print("=" * 80)

    try:
        import pdfplumber

        pdf_path = "data/pdfs/978-5-0065-3635-7_2.pdf"

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[10]  # Страница 11 (про Овна)
            text = page.extract_text()

            print(f"\nСтраница 11, первые 500 символов:")
            print("-" * 80)
            print(text[:500])

            return text
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def test_pymupdf():
    """Тест PyMuPDF (fitz) - альтернатива."""
    print("\n" + "=" * 80)
    print("PYMUPDF (FITZ) - альтернатива")
    print("=" * 80)

    try:
        import fitz  # PyMuPDF

        pdf_path = "data/pdfs/978-5-0065-3635-7_2.pdf"

        doc = fitz.open(pdf_path)
        page = doc[10]  # Страница 11
        text = page.get_text()

        print(f"\nСтраница 11, первые 500 символов:")
        print("-" * 80)
        print(text[:500])

        doc.close()
        return text
    except ImportError:
        print("PyMuPDF не установлен. Установите: pip install PyMuPDF")
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def test_pypdf():
    """Тест PyPDF2."""
    print("\n" + "=" * 80)
    print("PYPDF2")
    print("=" * 80)

    try:
        import PyPDF2

        pdf_path = "data/pdfs/978-5-0065-3635-7_2.pdf"

        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            page = reader.pages[10]
            text = page.extract_text()

            print(f"\nСтраница 11, первые 500 символов:")
            print("-" * 80)
            print(text[:500])

            return text
    except ImportError:
        print("PyPDF2 не установлен. Установите: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"Ошибка: {e}")
        return None


def analyze_quality(text, name):
    """Анализ качества извлеченного текста."""
    if not text:
        return

    print(f"\n{'='*80}")
    print(f"АНАЛИЗ КАЧЕСТВА: {name}")
    print(f"{'='*80}")

    # Проверяем проблемы
    issues = []

    # 1. Слипшиеся слова (длинные последовательности без пробелов)
    import re

    long_words = re.findall(r"[а-яёА-ЯЁ]{25,}", text[:1000])
    if long_words:
        issues.append(f"⚠ Найдено {len(long_words)} слипшихся слов (>25 символов)")
        print(f"  Примеры: {long_words[:3]}")

    # 2. Пробелы
    space_count = text[:1000].count(" ")
    char_count = len(text[:1000])
    space_ratio = space_count / char_count if char_count > 0 else 0
    print(f"  Соотношение пробелов: {space_ratio:.2%} ({space_count}/{char_count})")
    if space_ratio < 0.10:
        issues.append("⚠ Слишком мало пробелов (вероятно слипшиеся слова)")

    # 3. Переносы строк
    newline_count = text[:1000].count("\n")
    print(f"  Переносов строк: {newline_count}")

    if not issues:
        print("\n✓ Качество текста хорошее!")
    else:
        print("\nПроблемы:")
        for issue in issues:
            print(f"  {issue}")


def main():
    """Главная функция."""
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ БИБЛИОТЕК ДЛЯ ИЗВЛЕЧЕНИЯ ТЕКСТА ИЗ PDF")
    print("=" * 80)

    # Тестируем все библиотеки
    text1 = test_pdfplumber()
    text2 = test_pymupdf()
    text3 = test_pypdf()

    # Анализируем качество
    if text1:
        analyze_quality(text1, "pdfplumber")
    if text2:
        analyze_quality(text2, "PyMuPDF")
    if text3:
        analyze_quality(text3, "PyPDF2")

    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 80)
    print("\nЕсли PyMuPDF показывает лучшее качество:")
    print("1. Установите: pip install PyMuPDF")
    print("2. Обновите text_pdf_parser.py для использования PyMuPDF")
    print("\nЕсли все библиотеки дают слипшиеся слова:")
    print("1. Проблема в самом PDF (плохое качество)")
    print("2. Используйте OCR: python index_pdfs.py --enable_ocr")


if __name__ == "__main__":
    main()
