"""Файловый журнал сервера: traceback не должен теряться вместе с процессом.

H2-4 пишет структурные события и ``logger.exception`` при 500, но при запуске
через ``scripts/start.py`` всё уходило в stdout дочернего процесса. Если
терминал/сессия закрывались, причина падения исчезала. Модуль добавляет
ротационный файловый обработчик, путь задаётся ``HIPPORAG_LOG_FILE``.

По умолчанию файл не создаётся: тесты и библиотечное использование не должны
писать на диск. ``scripts/start.py`` выставляет переменную сам.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional, Union

LOG_FILE_ENV = "HIPPORAG_LOG_FILE"

#: Один файл — до 10 МБ, храним 5 архивных копий.
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

#: Логгеры uvicorn обрабатывают свои записи сами; чтобы access/error тоже
#: попадали в файл, обработчик добавляется и им.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

#: Структурные события H2-4 (ask_failed, answer_empty, …). У этого логгера
#: propagate=False и свой StreamHandler, поэтому файловый (F-1) добавляется
#: ему напрямую — иначе события не попадают в журнал (F-2).
_EVENTS_LOGGER_NAME = "hipporag.events"


def _resolve(path: Union[str, os.PathLike[str], None]) -> Optional[Path]:
    configured = path or os.getenv(LOG_FILE_ENV)
    if not configured:
        return None
    return Path(configured).expanduser().resolve()


def _has_handler(logger: logging.Logger, log_path: Path) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            if Path(handler.baseFilename) == log_path:
                return True
    return False


def _make_handler(log_path: Path) -> RotatingFileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def configure_file_logging(
    path: Union[str, os.PathLike[str], None] = None,
) -> Optional[Path]:
    """Добавить файловый обработчик root-логгеру и логгерам uvicorn.

    Возвращает путь к журналу или ``None``, если переменная не задана.
    Повторный вызов с тем же путём не создаёт дубликатов обработчиков.
    """
    log_path = _resolve(path)
    if log_path is None:
        return None

    root = logging.getLogger()
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)

    handler = _make_handler(log_path)
    if _has_handler(root, log_path):
        handler.close()
    else:
        root.addHandler(handler)

    for name in (*_UVICORN_LOGGERS, _EVENTS_LOGGER_NAME):
        logger = logging.getLogger(name)
        if _has_handler(logger, log_path):
            continue
        # Один и тот же обработчик можно вешать на несколько логгеров:
        # запись уходит в файл один раз на каждую ветку логирования.
        logger.addHandler(handler)
    return log_path


def remove_file_logging(path: Union[str, os.PathLike[str], None] = None) -> None:
    """Убрать файловые обработчики этого пути (тесты, смена журнала)."""
    log_path = _resolve(path)
    if log_path is None:
        return
    loggers: Iterable[logging.Logger] = [
        logging.getLogger(),
        *(logging.getLogger(name) for name in _UVICORN_LOGGERS),
    ]
    for logger in loggers:
        for handler in list(logger.handlers):
            if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path:
                logger.removeHandler(handler)
                handler.close()
