#!/usr/bin/env python3
"""One-command запуск веб-UI HippoRAG (H2-3).

Проверяет версию Python, наличие ``.env``, доступность LM Studio и каталог
индекса, затем запускает ``uvicorn astro_app.web:app``. Скрипт не импортирует
тяжёлые зависимости проекта до запуска сервера, поэтому его можно вызывать из
``start.bat``/``start.sh`` до установки пакета.

Примеры::

    python scripts/start.py
    python scripts/start.py --host 127.0.0.1 --port 8000
    python scripts/start.py --check-only
    python scripts/start.py --skip-lm-check
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

try:
    from check_lmstudio import check_lmstudio
except ImportError:  # запуск через importlib в тестах
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_lmstudio import check_lmstudio


@dataclass(frozen=True)
class ApiKeyDecision:
    """Решение о режиме API-ключа для выбранного адреса."""

    allowed: bool
    fail_open: bool
    message: str


def env_flag(name: str, default: bool = False) -> bool:
    """Читает булеву переменную окружения в стиле ``1/0/true/false``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        return max(0.1, float(raw))
    except ValueError:
        return default


def load_env_file(path: Path) -> Dict[str, str]:
    """Читает простой ``.env`` без зависимости от python-dotenv."""
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def apply_env(values: Dict[str, str]) -> None:
    """Добавляет значения в окружение, не перетирая уже заданные."""
    for key, value in values.items():
        os.environ.setdefault(key, value)


def ensure_env_file(root: Path) -> Tuple[Path, bool]:
    """Создаёт ``.env`` из ``.env.example``, если его ещё нет.

    Ошибка чтения/записи не должна ронять запуск: в контейнере под non-root
    каталог ``/app`` может быть доступен только на чтение. В этом случае
    возвращаем путь и ``False`` — сервер продолжит работать с переменными
    окружения, переданными снаружи (Docker/compose).
    """
    env_path = root / ".env"
    if env_path.exists():
        return env_path, False
    example = root / ".env.example"
    if not example.exists():
        return env_path, False
    try:
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        return env_path, False
    return env_path, True


def decide_api_key(host: str, api_key: Optional[str], required: bool) -> ApiKeyDecision:
    """Решает, можно ли стартовать без API_KEY.

    Для loopback разрешается fail-open (как в документации API), для адреса
    вне loopback ключ обязателен.
    """
    if api_key and api_key.strip():
        return ApiKeyDecision(True, False, "API_KEY задан")
    if not required:
        return ApiKeyDecision(True, False, "API_KEY_REQUIRED=0")
    if host.strip().lower() in LOOPBACK_HOSTS:
        return ApiKeyDecision(
            True,
            True,
            "API_KEY не задан; для loopback включаю API_KEY_REQUIRED=0",
        )
    return ApiKeyDecision(False, False, "Для адреса вне loopback нужен API_KEY")


def build_uvicorn_command(host: str, port: int, *, reload: bool = False) -> List[str]:
    """Собирает команду запуска uvicorn."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "astro_app.web:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        command.append("--reload")
    return command


def check_python_version(
    version_info: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str]:
    """Проверяет поддерживаемую версию Python (3.10–3.12)."""
    info = version_info or (sys.version_info.major, sys.version_info.minor)
    if info < (3, 10):
        return False, f"Нужен Python 3.10+, сейчас {info[0]}.{info[1]}"
    if info >= (3, 13):
        return (
            True,
            f"Python {info[0]}.{info[1]}: проект проверен на 3.10–3.12, возможны проблемы",
        )
    return True, f"Python {info[0]}.{info[1]}"


def index_status(save_dir: str) -> Tuple[bool, str]:
    """Проверяет, существует ли каталог индекса."""
    path = Path(save_dir)
    if not path.is_absolute():
        path = ROOT / path
    if path.exists():
        return True, f"Индекс: {path}"
    return (
        False,
        f"Индекс не найден: {path}. UI запустится, но /search вернёт пустой "
        "результат. Соберите индекс или загрузите PDF через /upload-and-ingest.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command запуск веб-UI HippoRAG (H2-3).")
    parser.add_argument(
        "--host",
        default=None,
        help="адрес uvicorn (по умолчанию APP_HOST или 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="порт uvicorn (по умолчанию APP_PORT или 8000)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="URL LM Studio; по умолчанию LLM_BASE_URL из .env",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="id модели; по умолчанию LLM_API_NAME или LLM_NAME",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="таймаут проверки LM Studio; по умолчанию STARTUP_CHECK_TIMEOUT",
    )
    parser.add_argument(
        "--skip-lm-check",
        action="store_true",
        help="не проверять LM Studio при старте",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="выполнить проверки и выйти, не запуская сервер",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="запустить uvicorn с --reload (для разработки)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    env_path, created = ensure_env_file(ROOT)
    if created:
        print(
            "[start] Создан .env из .env.example. Проверьте LLM_NAME, " "LLM_API_NAME и SAVE_DIR."
        )
    apply_env(load_env_file(env_path))

    ok, python_message = check_python_version()
    print(f"[start] {python_message}")
    if not ok:
        return 1

    host = args.host or os.getenv("APP_HOST", "127.0.0.1")
    port = args.port if args.port is not None else env_int("APP_PORT", 8000)
    if not 1 <= port <= 65535:
        print(f"[start] Некорректный порт: {port}. Допустимый диапазон 1–65535.")
        return 1
    base_url = args.base_url or os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    model = args.model or os.getenv("LLM_API_NAME") or os.getenv("LLM_NAME")

    check_lm = not args.skip_lm_check and env_flag("STARTUP_CHECK_LM", True)
    if check_lm:
        timeout = (
            args.timeout if args.timeout is not None else env_float("STARTUP_CHECK_TIMEOUT", 5.0)
        )
        result = check_lmstudio(base_url, model, timeout=timeout)
        print(f"[start] {result.message}")
        if not result.ok:
            print(
                "[start] Подсказка: запустите LM Studio → Developer → Start Server "
                "и проверьте LLM_BASE_URL/LLM_API_NAME в .env. "
                "Пропустить проверку: --skip-lm-check или STARTUP_CHECK_LM=0."
            )
            return 1
    else:
        print("[start] Проверка LM Studio отключена")

    decision = decide_api_key(host, os.getenv("API_KEY"), env_flag("API_KEY_REQUIRED", True))
    if not decision.allowed:
        print(f"[start] {decision.message}")
        print(
            "[start] Задайте API_KEY в .env или запускайте на 127.0.0.1. "
            "Для локальной разработки можно API_KEY_REQUIRED=0."
        )
        return 1
    if decision.fail_open:
        os.environ["API_KEY_REQUIRED"] = "0"
        print(f"[start] {decision.message}")

    save_dir = os.getenv("SAVE_DIR", "outputs/astrology_test")
    index_ok, index_message = index_status(save_dir)
    print(f"[start] {index_message}")
    if not index_ok:
        print("[start] Это предупреждение, не ошибка: PDF можно загрузить через API.")

    if importlib.util.find_spec("uvicorn") is None:
        print("[start] uvicorn не установлен. Выполните: python -m pip install -e .")
        return 1

    if args.check_only:
        print("[start] Проверки пройдены.")
        return 0

    command = build_uvicorn_command(host, port, reload=args.reload)
    print("[start] Запуск:", " ".join(command))
    print(f"[start] UI: http://{host}:{port}/  ·  API docs: http://{host}:{port}/docs")
    try:
        return subprocess.call(command, cwd=str(ROOT), env=os.environ.copy())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
