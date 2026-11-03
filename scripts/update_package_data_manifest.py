#!/usr/bin/env python3
"""Пересобрать манифест package-data (H0-8).

Манифест ``src/astro_app/data/data_manifest.json`` объявляет версию формата
bundled-данных и отпечатки файлов. ``load_synonyms()`` сверяет их с кодом,
поэтому после намеренного изменения ``src/astro_app/data/*.json`` манифест
нужно пересобрать — иначе загрузка данных упадёт с понятной ошибкой.

Запуск из корня репозитория:

    python scripts/update_package_data_manifest.py          # пересобрать
    python scripts/update_package_data_manifest.py --check  # только проверить
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from astro_app import versions  # noqa: E402 — путь к src добавляется выше


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Пересобрать манифест package-data")
    parser.add_argument(
        "--check",
        action="store_true",
        help="только проверить соответствие, ничего не записывать",
    )
    args = parser.parse_args(argv)

    if args.check:
        try:
            manifest = versions.check_package_data()
        except versions.PackageDataError as error:
            print(f"ПРОВЕРКА НЕ ПРОЙДЕНА:\n{error}", file=sys.stderr)
            return 1
        files = ", ".join(sorted(manifest["files"]))
        print(
            f"Package-data соответствуют коду: версия {versions.PACKAGE_DATA_VERSION}; файлы: {files}"
        )
        return 0

    expected = versions.build_package_data_manifest()
    current = None
    path = versions.PACKAGE_DATA_MANIFEST
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            current = None

    if current == expected:
        print(f"Манифест уже актуален: {path}")
        return 0

    versions.write_package_data_manifest()
    print(f"Манифест обновлён: {path}")
    print(f"Версия package-data: {versions.PACKAGE_DATA_VERSION}")
    for name, meta in sorted(expected["files"].items()):
        print(f"  {name}: sha256 {meta['sha256'][:12]}…, записей {meta.get('entries', '—')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
