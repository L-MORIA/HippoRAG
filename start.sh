#!/usr/bin/env bash
# H2-3: один скрипт для первого запуска и старта веб-UI.
# CLI по-прежнему доступен: python main.py
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

# Предпочитаем поддерживаемые версии 3.10-3.12, даже если python3 в PATH — 3.14.
if [ -z "${PYTHON:-}" ]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi

if [ -z "${PYTHON:-}" ]; then
    echo "[H2-3] python3 не найден. Установите Python 3.10-3.12." >&2
    exit 1
fi

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)'; then
    echo "[H2-3] Нужен Python 3.10-3.12. Найден: $("$PYTHON" -V 2>&1)" >&2
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "[H2-3] Первый запуск: создаю .venv и устанавливаю зависимости..."
    "$PYTHON" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e .
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[H2-3] Создан .env из .env.example. Проверьте LLM_NAME, LLM_API_NAME, SAVE_DIR."
fi

exec .venv/bin/python scripts/start.py "$@"
