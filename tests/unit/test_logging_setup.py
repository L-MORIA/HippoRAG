"""Тесты файлового журнала сервера: traceback не должен теряться."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from hipporag.search_api import logging_setup


def _root_file_handlers(log_file) -> list[RotatingFileHandler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, RotatingFileHandler)
        and Path(handler.baseFilename) == log_file.resolve()
    ]


def test_configure_writes_traceback_to_file(tmp_path):
    log_file = tmp_path / "server.log"
    log_path = logging_setup.configure_file_logging(log_file)
    assert log_path == log_file.resolve()

    try:
        raise RuntimeError("boom-500")
    except RuntimeError:
        logging.getLogger("tests.filelog").exception("Ошибка веб-поиска: mode=2")
    finally:
        logging_setup.remove_file_logging(log_file)

    content = log_file.read_text(encoding="utf-8")
    assert "Ошибка веб-поиска: mode=2" in content
    assert "Traceback" in content
    assert "RuntimeError: boom-500" in content


def test_configure_is_idempotent(tmp_path):
    log_file = tmp_path / "server.log"
    try:
        logging_setup.configure_file_logging(log_file)
        logging_setup.configure_file_logging(log_file)
        assert len(_root_file_handlers(log_file)) == 1
    finally:
        logging_setup.remove_file_logging(log_file)


def test_configure_without_env_does_nothing(monkeypatch):
    monkeypatch.delenv(logging_setup.LOG_FILE_ENV, raising=False)

    assert logging_setup.configure_file_logging() is None


def test_configure_reads_env_path(monkeypatch, tmp_path):
    log_file = tmp_path / "nested" / "server.log"
    monkeypatch.setenv(logging_setup.LOG_FILE_ENV, str(log_file))
    try:
        log_path = logging_setup.configure_file_logging()
        assert log_path == log_file.resolve()
    finally:
        logging_setup.remove_file_logging()
    # Каталог создаётся при настройке обработчика.
    assert log_file.parent.is_dir()
