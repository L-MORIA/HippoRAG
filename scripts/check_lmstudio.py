#!/usr/bin/env python3
"""Проверка LM Studio перед запуском HippoRAG (H2-3).

Скрипт использует только стандартную библиотеку: его можно запустить до
установки зависимостей проекта, из ``start.bat``/``start.sh`` или вручную.

Примеры::

    python scripts/check_lmstudio.py
    python scripts/check_lmstudio.py --base-url http://127.0.0.1:1234/v1 \
        --model google/gemma-4-e4b
    python scripts/check_lmstudio.py --json

Код возврата 0 — проверка пройдена, 1 — нет.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class CheckResult:
    """Результат проверки LM Studio."""

    ok: bool
    base_url: str
    model: Optional[str]
    models: tuple[str, ...]
    message: str


def normalize_base_url(base_url: str) -> str:
    """Убирает пробелы и хвостовой слэш из базового URL."""
    return (base_url or "").strip().rstrip("/")


def fetch_models(base_url: str, *, timeout: float = 5.0, opener=None) -> List[str]:
    """Возвращает id моделей из ``GET {base_url}/models``.

    ``opener`` нужен для unit-тестов: по умолчанию используется
    ``urllib.request.urlopen``.
    """
    url = normalize_base_url(base_url) + "/models"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    open_fn = opener or urllib.request.urlopen
    with open_fn(request, timeout=timeout) as response:
        raw = response.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ответ {url} не является JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"ответ {url} не содержит списка data")
    return [
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def check_lmstudio(
    base_url: str,
    model: Optional[str] = None,
    *,
    timeout: float = 5.0,
    opener=None,
) -> CheckResult:
    """Проверяет доступность LM Studio и наличие нужной модели.

    Если ``model`` пуст, проверяется только то, что сервер отвечает и
    возвращает хотя бы одну модель.
    """
    normalized = normalize_base_url(base_url)
    wanted = (model or "").strip() or None
    try:
        models = fetch_models(normalized, timeout=timeout, opener=opener)
    except urllib.error.HTTPError as error:
        return CheckResult(
            False,
            normalized,
            wanted,
            (),
            f"LM Studio вернула HTTP {error.code} для {normalized}/models",
        )
    except urllib.error.URLError as error:
        return CheckResult(
            False,
            normalized,
            wanted,
            (),
            f"LM Studio недоступна по {normalized}: {error.reason}",
        )
    except (TimeoutError, OSError) as error:
        return CheckResult(
            False,
            normalized,
            wanted,
            (),
            f"LM Studio недоступна по {normalized}: {error}",
        )
    except ValueError as error:
        return CheckResult(False, normalized, wanted, (), str(error))

    if not models:
        return CheckResult(
            False,
            normalized,
            wanted,
            (),
            f"LM Studio отвечает, но не вернула ни одной модели ({normalized}/models)",
        )

    if wanted and wanted not in models:
        available = ", ".join(models[:10])
        if len(models) > 10:
            available += f" и ещё {len(models) - 10}"
        return CheckResult(
            False,
            normalized,
            wanted,
            tuple(models),
            f"Модель {wanted!r} не найдена. Доступны: {available}",
        )

    if wanted:
        message = f"LM Studio доступна, модель {wanted!r} найдена"
    else:
        message = f"LM Studio доступна, моделей: {len(models)}"
    return CheckResult(True, normalized, wanted, tuple(models), message)


def _env_timeout(default: float = 5.0) -> float:
    raw = os.getenv("STARTUP_CHECK_TIMEOUT")
    if not raw or not raw.strip():
        return default
    try:
        return max(0.1, float(raw))
    except ValueError:
        return default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Проверка LM Studio и модели перед запуском HippoRAG (H2-3)."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        help="базовый URL OpenAI-совместимого API (по умолчанию LLM_BASE_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_API_NAME") or os.getenv("LLM_NAME") or "",
        help="id модели в LM Studio (по умолчанию LLM_API_NAME или LLM_NAME)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_timeout(),
        help="таймаут запроса в секундах (по умолчанию STARTUP_CHECK_TIMEOUT или 5)",
    )
    parser.add_argument(
        "--no-model-check",
        action="store_true",
        help="проверить только доступность сервера, не сверять id модели",
    )
    parser.add_argument("--json", action="store_true", help="вывести результат в JSON")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="не печатать сообщение при успешной проверке",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_lmstudio(
        args.base_url,
        None if args.no_model_check else args.model,
        timeout=args.timeout,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "base_url": result.base_url,
                    "model": result.model,
                    "models": list(result.models),
                    "message": result.message,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif not args.quiet or not result.ok:
        print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
