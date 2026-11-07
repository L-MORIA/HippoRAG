"""F-4: сетевые сбои LM Studio в режиме 1 не должны глушить поиск.

``reformulate_query`` при любом ``httpx.HTTPError`` (включая
``RemoteProtocolError``, ``ReadError``, ``PoolTimeout``) возвращает исходный
запрос — поиск продолжается без переформулировки, а не падает 500-й.
"""

from __future__ import annotations

import httpx
import pytest

from astro_app.query import reformulate_query


@pytest.fixture
def hipporag_stub():
    class Stub:
        pass

    return Stub()


def test_remote_protocol_error_returns_original_query(hipporag_stub, monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.RemoteProtocolError("сервер оборвал соединение")

    monkeypatch.setattr(httpx, "get", boom)
    assert reformulate_query(hipporag_stub, "число гуа 1") == "число гуа 1"


def test_read_error_returns_original_query(hipporag_stub, monkeypatch):
    def boom(*args, **kwargs):
        raise httpx.ReadError("обрыв чтения")

    monkeypatch.setattr(httpx, "get", boom)
    assert reformulate_query(hipporag_stub, "зоны овна") == "зоны овна"


def test_health_status_not_200_returns_original_query(hipporag_stub, monkeypatch):
    class Resp:
        status_code = 503

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
    assert reformulate_query(hipporag_stub, "зоны овна") == "зоны овна"
