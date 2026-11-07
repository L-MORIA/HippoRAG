"""F-2: причина сбоя доходит до журнала и до ответа клиента.

* ``configure_file_logging`` вешает файловый обработчик и на
  ``hipporag.events`` (события ask_failed/answer_empty попадают в журнал);
* ``_http_error_for`` даёт 500 с именем класса исключения, а сетевые/
  таймаутные сбои — 503 «временно недоступно».
"""

from __future__ import annotations

import logging
from pathlib import Path

from astro_app.web.server import _http_error_for
from hipporag.search_api.logging_setup import (
    _EVENTS_LOGGER_NAME,
    _has_handler,
    configure_file_logging,
)


def _detach_file_handlers(log_path: Path) -> None:
    """Снять файловые обработчики со всех затронутых логгеров."""
    for logger in (logging.getLogger(), logging.getLogger(_EVENTS_LOGGER_NAME)):
        for handler in list(logger.handlers):
            base = getattr(handler, "baseFilename", None)
            if base and Path(base) == log_path:
                logger.removeHandler(handler)
                handler.close()


def test_events_logger_receives_file_handler(tmp_path):
    log_path = configure_file_logging(tmp_path / "server.log")
    assert log_path is not None
    events = logging.getLogger(_EVENTS_LOGGER_NAME)
    try:
        assert _has_handler(events, log_path)
        events.info('{"event":"ask_failed","error_type":"RuntimeError"}')
        body = log_path.read_text(encoding="utf-8")
        assert "ask_failed" in body and "RuntimeError" in body
    finally:
        _detach_file_handlers(log_path)


def test_runtime_error_maps_to_500_with_exception_name():
    info = _http_error_for(RuntimeError("взрыв в пайплайне"))
    assert info.status_code == 500
    assert "RuntimeError" in info.detail
    assert "взрыв" not in info.detail  # текст исключения наружу не отдаётся


def test_timeout_error_maps_to_503(monkeypatch):
    from hipporag.search_api import observability

    monkeypatch.setattr(
        observability, "is_timeout_error", lambda error: isinstance(error, TimeoutError)
    )
    info = _http_error_for(TimeoutError("LLM не ответила"))
    assert info.status_code == 503
    assert "TimeoutError" in info.detail


def test_connection_error_maps_to_503():
    info = _http_error_for(ConnectionError("отказ в соединении"))
    assert info.status_code == 503
    assert "ConnectionError" in info.detail
