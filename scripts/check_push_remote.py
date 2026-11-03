#!/usr/bin/env python3
"""Защита от пуша в чужой репозиторий.

Клон этого проекта — форк: ``origin`` указывает на апстрим
``osu-nlp-group/hipporag``, а собственные изменения уходят в ``fork``
(``L-MORIA/HippoRAG``). Случайный пуш в апстрим дорог, поэтому цель пуша
проверяется механически, а не «по внимательности».

Режимы:

* **Хук.** Git вызывает ``pre-push`` с двумя аргументами — имя remote и его
  URL. Скрипт пропускает пуш только в репозиторий владельца проекта.
* **Установка.** ``--install`` кладёт тонкую обёртку в ``.git/hooks/pre-push``:
  каталог ``.git`` не версионируется, поэтому обёртку ставит этот скрипт, а
  логика проверки живёт здесь и версионируется вместе с проектом.
* **Ручная проверка.** ``--check-url URL`` — проверить адрес, ничего не пуша.

Проверка привязана к имени, а не к имени remote: ``origin``, ``fork``,
``upstream`` или прямой ``git push <url>`` — правило одно и то же.

Аварийный обход для осознанных действий: ``GIT_ALLOW_FOREIGN_PUSH=1``.

Запуск:
    python scripts/check_push_remote.py --install
    python scripts/check_push_remote.py --check-url https://github.com/L-MORIA/HippoRAG.git
    python scripts/check_push_remote.py fork https://github.com/L-MORIA/HippoRAG.git
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Единственный репозиторий, в который разрешено пушить.
ALLOWED_SLUG = "l-moria/hipporag"

#: Переменная окружения для осознанного обхода защиты.
BYPASS_ENV = "GIT_ALLOW_FOREIGN_PUSH"

#: Метка в файле хука — по ней отличаем свой хук от чужого при переустановке.
HOOK_MARKER = "check_push_remote.py"

#: ``git@host:owner/repo`` — scp-подобная форма записи remote.
SCP_LIKE_RE = re.compile(r"^(?:[^@/\s]+@)?([^:/\s]+):(?!//)(.+)$")

HOOK_SOURCE = """#!/bin/sh
# pre-push: пуш разрешён только в репозиторий владельца проекта.
# Файл создан командой: python scripts/check_push_remote.py --install
# Логика проверки живёт в scripts/check_push_remote.py (версионируется).
repo_root=$(git rev-parse --show-toplevel) || exit 1
for exe in python python3 py; do
    if command -v "$exe" >/dev/null 2>&1; then
        exec "$exe" "$repo_root/scripts/check_push_remote.py" "$1" "$2"
    fi
done
echo "pre-push: python не найден - цель пуша не проверена" >&2
exit 1
"""


def normalize_slug(url: str) -> str | None:
    """Привести URL remote к виду ``owner/repo`` в нижнем регистре.

    Понимает ``https://…``, ``ssh://…``, ``git@host:owner/repo`` и токен в
    составе URL. Возвращает ``None``, если из адреса не извлекается пара
    «владелец/репозиторий».
    """
    value = url.strip()
    if not value:
        return None

    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[: -len(".git")]

    if "://" in value:
        path = urlsplit(value).path
    else:
        match = SCP_LIKE_RE.match(value)
        path = match.group(2) if match else value

    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[-2]}/{parts[-1]}".lower()


def is_allowed(url: str) -> bool:
    """Разрешён ли пуш по адресу ``url``."""
    return normalize_slug(url) == ALLOWED_SLUG


def check(remote_name: str, remote_url: str) -> int:
    """Проверить цель пуша. Возвращает код выхода для git-хука."""
    if is_allowed(remote_url):
        return 0

    if os.environ.get(BYPASS_ENV) == "1":
        print(
            f"pre-push: ВНИМАНИЕ - пуш в чужой репозиторий разрешён через {BYPASS_ENV}.\n"
            f"  remote: {remote_name}\n"
            f"  цель:   {remote_url}",
            file=sys.stderr,
        )
        return 0

    print(
        "ОТКАЗАНО: пуш в чужой репозиторий.\n"
        f"  remote: {remote_name}\n"
        f"  цель:   {remote_url}\n"
        f"  разрешён только: {ALLOWED_SLUG}\n"
        "\n"
        "Пуш отменён. Если действие осознанное и цель действительно нужна:\n"
        f"  {BYPASS_ENV}=1 git push ...",
        file=sys.stderr,
    )
    return 1


def hooks_dir() -> Path:
    """Каталог хуков текущего репозитория (учитывает worktree)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"не удалось определить каталог хуков: {error}") from error

    candidate = Path(result.stdout.strip())
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def install(force: bool = False) -> int:
    """Установить хук ``pre-push``. Возвращает код выхода."""
    target = hooks_dir() / "pre-push"

    if target.exists() and not force:
        existing = target.read_text(encoding="utf-8", errors="replace")
        if HOOK_MARKER not in existing:
            print(
                f"ОТКАЗАНО: {target} уже существует и создан не этим скриптом.\n"
                "Переустановка перезапишет чужой хук. Если это осознанно:\n"
                "  python scripts/check_push_remote.py --install --force",
                file=sys.stderr,
            )
            return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" обязателен: файл хука исполняет sh, CRLF ломает shebang.
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(HOOK_SOURCE)
    try:
        target.chmod(0o755)
    except OSError:
        pass  # Windows: права не выставляются, git всё равно запустит хук

    print(f"Хук установлен: {target}")
    print(f"Разрешённый репозиторий: {ALLOWED_SLUG}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверить цель git push и установить хук pre-push",
    )
    parser.add_argument("remote_name", nargs="?", help="Имя remote (передаёт git)")
    parser.add_argument("remote_url", nargs="?", help="URL remote (передаёт git)")
    parser.add_argument("--install", action="store_true", help="Установить хук pre-push")
    parser.add_argument("--force", action="store_true", help="Перезаписать существующий хук")
    parser.add_argument("--check-url", metavar="URL", help="Проверить адрес и ничего не пушить")
    args = parser.parse_args(argv)

    if args.install:
        return install(force=args.force)

    if args.check_url is not None:
        return check("(manual)", args.check_url)

    if args.remote_url is not None:
        return check(args.remote_name or "?", args.remote_url)

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
