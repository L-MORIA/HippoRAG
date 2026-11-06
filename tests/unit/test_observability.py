"""Тесты H2-4: метрики, структурные события, request ID и наблюдаемость API."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from astro_app.web import server as web_server
from astro_app.web.models import AskRequest, AskResponse
from hipporag.qa.grounding import ABSTAIN_TEXT
from hipporag.search_api import observability
from hipporag.search_api import server as srv
from hipporag.search_api.models import SearchRequest
from hipporag.utils.custom_types import QuerySolution


class _StubRag:
    def __init__(self, solution):
        self.solution = solution

    def rag_qa(self, queries):  # noqa: ARG002
        return [self.solution], [], [{}]


class _Lifecycle:
    def status(self):
        return {
            "state": "idle",
            "index": {
                "format_version": 1,
                "format_state": "ok",
                "created_at": "2026-11-06T10:00:00",
                "llm_name": "gemma-test",
                "embedding_model": "mpnet-test",
                "documents": 3,
            },
        }


@pytest.fixture(autouse=True)
def clean_metrics():
    observability.metrics.reset()
    srv._GROUNDING_LATENCIES_MS.clear()
    yield
    observability.metrics.reset()
    srv._GROUNDING_LATENCIES_MS.clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(srv, "hipporag", None)
    monkeypatch.setattr(srv, "index_lifecycle", _Lifecycle())
    monkeypatch.setattr(srv, "API_KEY", None)
    monkeypatch.setattr(srv, "API_KEY_REQUIRED", False)
    # web_server.app — тот же объект, но он сохраняет UI-маршруты даже
    # если другой тест перезагрузил hipporag.search_api.server.
    return TestClient(web_server.app)


@pytest.fixture
def event_caplog(caplog):
    """Поймать события: у логгера событий propagate=False."""
    logger = logging.getLogger("hipporag.events")
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _events(caplog):
    return [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "hipporag.events"
    ]


def test_metrics_records_requests_errors_slow_and_empty():
    metrics = observability.Metrics(slow_threshold_ms=100)

    assert metrics.record_request("/search", 200, 150) is True
    assert metrics.record_request("/search", 500, 20) is False
    metrics.record_request("/search", 401, 10)
    metrics.record_empty_answer("no_citations")
    metrics.record_timeout()

    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == 3
    assert snapshot["errors_total"] == 2
    assert snapshot["server_errors_total"] == 1
    assert snapshot["client_errors_total"] == 1
    assert snapshot["slow_requests_total"] == 1
    assert snapshot["timeouts_total"] == 1
    assert snapshot["empty_answers_total"] == 1
    assert snapshot["empty_answers_by_reason"] == {"no_citations": 1}
    assert snapshot["routes"]["/search"]["requests"] == 3
    assert snapshot["routes"]["/search"]["errors"] == 2
    assert snapshot["routes"]["/search"]["avg_ms"] == 60.0
    assert snapshot["routes"]["/search"]["max_ms"] == 150.0
    assert snapshot["latency"] == {
        "count": 3,
        "avg_ms": 60.0,
        "p50_ms": 20.0,
        "p95_ms": 150.0,
        "max_ms": 150.0,
    }


def test_metrics_aggregates_stages():
    metrics = observability.Metrics()
    metrics.record_stage("ask", "retrieval", 100)
    metrics.record_stage("ask", "retrieval", 300)
    metrics.record_stage("search", "rag_qa", 50)

    snapshot = metrics.snapshot()
    assert snapshot["stages"]["ask.retrieval"] == {
        "count": 2,
        "avg_ms": 200.0,
        "max_ms": 300.0,
    }
    assert snapshot["stages"]["search.rag_qa"]["count"] == 1


def test_is_timeout_error_recognizes_client_timeouts():
    class ReadTimeout(Exception):
        pass

    assert observability.is_timeout_error(TimeoutError())
    assert observability.is_timeout_error(ReadTimeout())
    assert not observability.is_timeout_error(ValueError())


def test_classify_empty_answer_distinguishes_reasons():
    assert observability.classify_empty_answer(abstained=False, citations=1) is None
    assert (
        observability.classify_empty_answer(abstained=False, citations=0, grounding_errors=[])
        == "no_citations"
    )
    assert (
        observability.classify_empty_answer(
            abstained=True, citations=0, grounding_errors=["no_fragments"]
        )
        == "no_fragments"
    )
    assert (
        observability.classify_empty_answer(
            abstained=True, citations=0, grounding_errors=["llm_unavailable_or_empty"]
        )
        == "llm_unavailable_or_empty"
    )
    assert (
        observability.classify_empty_answer(
            abstained=True, citations=0, grounding_errors=["no_inline_citations"]
        )
        == "no_inline_citations"
    )
    assert (
        observability.classify_empty_answer(abstained=True, citations=0, grounding_errors=[])
        == "abstained"
    )


def test_log_event_is_json_with_request_id(event_caplog):
    token = observability.set_request_id("req-test")
    try:
        with event_caplog.at_level(logging.INFO, logger="hipporag.events"):
            observability.log_event(
                "answer_empty",
                path="/search",
                reason="no_fragments",
                count=2,
            )
    finally:
        observability.reset_request_id(token)

    payload = json.loads(event_caplog.records[-1].getMessage())
    assert payload["event"] == "answer_empty"
    assert payload["level"] == "info"
    assert payload["request_id"] == "req-test"
    assert payload["path"] == "/search"
    assert payload["reason"] == "no_fragments"
    assert payload["count"] == 2
    assert "ts" in payload


def test_request_id_header_is_generated_and_echoed(client):
    first = client.get("/health")
    assert first.status_code == 200
    assert first.headers["X-Request-ID"]

    second = client.get("/health", headers={"X-Request-ID": "client-123"})
    assert second.headers["X-Request-ID"] == "client-123"

    unsafe = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert unsafe.headers["X-Request-ID"] != "bad id with spaces"
    assert len(unsafe.headers["X-Request-ID"]) == 32


def test_health_exposes_models_created_at_and_metrics(client):
    client.get("/health")
    body = client.get("/health").json()

    assert body["llm_model"] == "gemma-test"
    assert body["embedding_model"] == "mpnet-test"
    assert body["index_created_at"] == "2026-11-06T10:00:00"
    assert body["metrics"]["requests_total"] >= 1
    assert "latency" in body["metrics"]
    assert "p95_ms" in body["metrics"]["latency"]


def test_metrics_endpoint_returns_snapshot(client):
    client.get("/health")
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["requests_total"] >= 1
    assert body["uptime_seconds"] >= 0
    assert "routes" in body
    assert "stages" in body
    assert body["slow_threshold_ms"] > 0


def test_search_records_empty_answer_and_stage_timings(monkeypatch, event_caplog):
    solution = QuerySolution(
        question="Сколько стоит прибор?",
        answer=ABSTAIN_TEXT,
        docs=["Прибор работает от батареи AA."],
        citations=[],
        used_citations=[],
        abstained=True,
        grounding_errors=["no_inline_citations"],
    )
    monkeypatch.setattr(srv, "hipporag", _StubRag(solution))

    with event_caplog.at_level(logging.INFO, logger="hipporag.events"):
        response = asyncio.run(srv.search(SearchRequest(question="Сколько стоит прибор?")))

    assert response.data.abstained is True
    snapshot = observability.metrics.snapshot()
    assert snapshot["empty_answers_total"] == 1
    assert snapshot["empty_answers_by_reason"] == {"no_inline_citations": 1}
    assert snapshot["stages"]["search.rag_qa"]["count"] == 1
    assert snapshot["stages"]["search.build_response"]["count"] == 1
    assert snapshot["stages"]["search.total"]["count"] == 1

    events = _events(event_caplog)
    completed = next(event for event in events if event["event"] == "search_completed")
    assert completed["empty_reason"] == "no_inline_citations"
    assert completed["stages"]["total_ms"] >= 0
    assert any(event["event"] == "answer_empty" for event in events)


def test_ask_records_stage_timings_and_empty_reason(monkeypatch, event_caplog):
    monkeypatch.setattr(web_server.api_server, "hipporag", object())
    answer = AskResponse(
        query="Вопрос",
        mode="5",
        mode_label="Быстрый",
        expert=False,
        answer=ABSTAIN_TEXT,
        confidence=0.0,
        citations=[],
        fragments=[],
        retrieved_docs=[],
        abstained=True,
        grounding_errors=["no_fragments"],
        reformulated_query=None,
        processing_time=0.3,
        retrieval_time=0.1,
        answer_time=0.2,
    )
    monkeypatch.setattr(web_server.service, "answer_question", lambda *_args, **_kwargs: answer)

    with event_caplog.at_level(logging.INFO, logger="hipporag.events"):
        result = asyncio.run(web_server.ask(AskRequest(question="Вопрос", mode="5")))

    assert result is answer
    snapshot = observability.metrics.snapshot()
    assert snapshot["empty_answers_by_reason"] == {"no_fragments": 1}
    assert snapshot["stages"]["ask.retrieval"]["count"] == 1
    assert snapshot["stages"]["ask.answer"]["count"] == 1
    assert snapshot["stages"]["ask.total"]["count"] == 1

    events = _events(event_caplog)
    completed = next(event for event in events if event["event"] == "ask_completed")
    assert completed["empty_reason"] == "no_fragments"
    assert completed["stages"]["retrieval_ms"] == 100.0
    assert any(event["event"] == "answer_empty" for event in events)


def test_dashboard_route_and_assets(client):
    response = client.get("/ui/dashboard")
    assert response.status_code == 200
    assert "Наблюдаемость" in response.text
    assert "/ui/dashboard.js" in response.text

    dashboard_js = (web_server.STATIC_DIR / "dashboard.js").read_text(encoding="utf-8")
    index = (web_server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/metrics" in dashboard_js
    assert "/health" in dashboard_js
    assert "/ui/dashboard" in index
