"""F-7: прогрев при старте и признак готовности в /health."""

from __future__ import annotations

from hipporag.search_api import server as srv
from hipporag.search_api.models import HealthResponse


def test_warmup_sets_ready_and_measures_duration(monkeypatch):
    calls = []

    def fake_search(hipporag, query, top_k=None, **kwargs):
        calls.append((hipporag, query, top_k))

    monkeypatch.setattr("astro_app.modes.bm25_hybrid.search_bm25_hybrid_ru_pages", fake_search)
    srv.WARMUP.update({"ready": False, "error": None, "duration_ms": None})
    try:
        srv._warmup()
        assert srv.WARMUP["ready"] is True
        assert srv.WARMUP["error"] is None
        assert srv.WARMUP["duration_ms"] is not None
        assert calls and calls[0][2] == 1
    finally:
        srv.WARMUP.update({"ready": False, "error": None, "duration_ms": None})


def test_warmup_error_does_not_raise_and_ready_set(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("пул пуст")

    monkeypatch.setattr("astro_app.modes.bm25_hybrid.search_bm25_hybrid_ru_pages", boom)
    srv.WARMUP.update({"ready": False, "error": None, "duration_ms": None})
    try:
        srv._warmup()
        assert srv.WARMUP["ready"] is True
        assert "RuntimeError" in srv.WARMUP["error"]
    finally:
        srv.WARMUP.update({"ready": False, "error": None, "duration_ms": None})


def test_health_response_model_has_ready_field():
    response = HealthResponse(uptime="0h 0m 1s", ready=True)
    assert response.ready is True
    assert HealthResponse(uptime="0h 0m 1s").ready is False
