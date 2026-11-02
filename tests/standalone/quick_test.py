#!/usr/bin/env python3
"""
Быстрый тест импортов
"""

import sys

sys.path.insert(0, ".")

print("Тест 1: Проверка базовых импортов...")

try:
    from hipporag.utils.config_utils import BaseConfig

    print("✓ BaseConfig")
except Exception as e:
    print(f"✗ BaseConfig: {e}")

try:
    from hipporag.pdf_ingestion import TextPDFParser

    print("✓ TextPDFParser")
except Exception as e:
    print(f"✗ TextPDFParser: {e}")

print("\nТест 2: Проверка LM Studio...")

try:
    import httpx

    r = httpx.get("http://localhost:1234/models", timeout=3)
    if r.status_code == 200:
        print("✓ LM Studio подключен!")
    else:
        print(f"✗ LM Studio вернул: {r.status_code}")
except Exception as e:
    print(f"✗ LM Studio недоступен: {e}")

print("\nГотово!")
