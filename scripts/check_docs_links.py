#!/usr/bin/env python3
"""Проверка относительных ссылок в Markdown-документах проекта.

Скрипт ищет ссылки и изображения вида ``[текст](путь)`` / ``![alt](путь)``,
пропускает внешние URL, якоря и абсолютные пути и проверяет, что цель
существует относительно файла, в котором находится ссылка.

Запуск:
    python scripts/check_docs_links.py
    python scripts/check_docs_links.py docs README.md README_RU.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "outputs",
    "data",
    "archive",
}

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def iter_markdown(paths: list[Path]):
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".md":
            yield path
        elif path.is_dir():
            for candidate in path.rglob("*.md"):
                if not any(part in SKIP_DIRS for part in candidate.parts):
                    yield candidate


def normalize_target(raw: str) -> str | None:
    target = raw.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split()[0]

    target = target.strip()
    if not target or target.startswith("#"):
        return None
    if SCHEME_RE.match(target) or target.startswith("//"):
        return None

    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target or target.startswith("/"):
        return None
    return target


def check_file(markdown_file: Path) -> list[str]:
    errors: list[str] = []
    text = markdown_file.read_text(encoding="utf-8", errors="replace")
    text = FENCE_RE.sub("", text)
    text = INLINE_CODE_RE.sub("", text)

    for match in LINK_RE.finditer(text):
        target = normalize_target(match.group(1))
        if target is None:
            continue
        resolved = (markdown_file.parent / target).resolve()
        try:
            resolved.relative_to(PROJECT_ROOT)
        except ValueError:
            errors.append(f"{markdown_file}: ссылка выходит за пределы проекта: {target}")
            continue
        if not resolved.exists():
            errors.append(f"{markdown_file}: битая ссылка: {target}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверить относительные ссылки в Markdown")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "README.md", PROJECT_ROOT / "README_RU.md", PROJECT_ROOT / "docs"],
        help="Файлы и каталоги для проверки (по умолчанию README и docs/)",
    )
    args = parser.parse_args()

    files = sorted(set(iter_markdown(args.paths)))
    if not files:
        print("Не найдено Markdown-файлов для проверки.")
        return 1

    errors: list[str] = []
    for markdown_file in files:
        errors.extend(check_file(markdown_file))

    for error in errors:
        print(error)

    print(f"Проверено файлов: {len(files)}; битых ссылок: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
