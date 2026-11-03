"""Тесты аутентификации API: guard, CORS, не-ASCII ключ (D1/D2/D4/N1/N2/N3).

Guard вызывается напрямую (не через HTTP) — HTTP-заголовки ASCII-only,
а не-ASCII API_KEY из .env — реальный сценарий для русскоязычного проекта.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def server_module():
    """Загружает server.py с заданным API_KEY."""
    import importlib
    import os

    os.environ["API_KEY"] = "test-key-123"
    import hipporag.search_api.server as srv

    importlib.reload(srv)
    srv.hipporag = None  # без загрузки индекса
    yield srv
    # восстановить окружение
    os.environ.pop("API_KEY", None)
    importlib.reload(srv)


def _run_guard(srv, headers, method="GET"):
    """Синхронная обёртка над async api_key_guard."""
    request = MagicMock()
    request.method = method
    request.headers = headers
    call_next = AsyncMock(return_value="PASS_THROUGH")
    return asyncio.run(srv.api_key_guard(request, call_next))


def test_correct_key_passes(server_module):
    r = _run_guard(server_module, {"X-API-Key": "test-key-123"})
    assert r == "PASS_THROUGH"


def test_wrong_key_401(server_module):
    r = _run_guard(server_module, {"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_missing_key_401(server_module):
    r = _run_guard(server_module, {})
    assert r.status_code == 401


def test_401_has_cors_headers(server_module):
    """N2: 401-ответ содержит CORS-заголовки."""
    r = _run_guard(server_module, {})
    assert "Access-Control-Allow-Origin" in r.headers


def test_non_ascii_api_key_does_not_crash(server_module):
    """N1: не-ASCII API_KEY не вызывает TypeError → не 500."""
    server_module.API_KEY = "секретный-ключ-с-кириллицей"
    r = _run_guard(server_module, {"X-API-Key": "ascii-header"})
    assert r.status_code == 401  # не 500
    r = _run_guard(server_module, {"X-API-Key": "секретный-ключ-с-кириллицей"})
    assert r == "PASS_THROUGH"  # точное совпадение → pass-through


def test_preflight_options_bypasses_key(server_module):
    """D2: OPTIONS preflight проходит без ключа."""
    r = _run_guard(server_module, {"Origin": "http://localhost:3000"}, method="OPTIONS")
    assert r == "PASS_THROUGH"


def test_no_api_key_fail_open(server_module):
    """D3: при незаданном API_KEY guard пропускает (fail-open)."""
    server_module.API_KEY = None
    r = _run_guard(server_module, {})
    assert r == "PASS_THROUGH"


def test_compare_digest_bytes_works_with_non_ascii(server_module):
    """N1 фикс: compare_digest работает с байтами любого содержимого."""
    import secrets

    a = "секретный-ключ".encode("utf-8")
    b = "секретный-ключ".encode("utf-8")
    c = "другой-ключ".encode("utf-8")
    assert secrets.compare_digest(a, b) is True
    assert secrets.compare_digest(a, c) is False
