#!/usr/bin/env python3
"""
Проверка файла OpenIE результатов.
"""

import json

# Читаем файл OpenIE
with open("outputs/astrology/openie_results_ner_gemma-4-e4b.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys())}")

if "docs" in data:
    print(f'\nTotal docs: {len(data["docs"])}')

    if len(data["docs"]) > 0:
        print(f'\nFirst doc keys: {list(data["docs"][0].keys())}')
        print(f'\nFirst doc idx: {data["docs"][0].get("idx", "N/A")}')
        print(f'First doc extracted_entities: {data["docs"][0].get("extracted_entities", [])[:3]}')
        print(f'First doc extracted_triples: {data["docs"][0].get("extracted_triples", [])[:2]}')
