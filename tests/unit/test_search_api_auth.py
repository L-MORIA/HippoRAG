"""Тесты аутентификации API: guard, CORS, fail-closed, аудит отказов.

Покрытие замечаний аудита:
* D1 — `X-API-Key` в `allow_headers` (браузерный клиент может передать ключ);
* D2 — CORS-preflight (OPTIONS) проходит без ключа;
* D3/D7 — ключ обязателен по умолчанию (fail-closed), opt-out явный;
* D5 — отказы считаются и видны в `/health`;
* N1 — не-ASCII `API_KEY` не приводит к исключению (сравнение по байтам);
* N2/N5 — origin в ответе об отказе берётся из запроса, а не захардкожен;
* Z-1 — порядок регистрации middleware: CORS зарегистрирован **после** guard,
  поэтому в Starlette он внешний слой и добавляет свои заголовки к 401.

Guard проверяется и напрямую (без HTTP — HTTP-заголовки ASCII-only, а
не-ASCII `API_KEY` из `.env` — реальный сценарий для русскоязычного проекта),
и через `TestClient` без запуска lifespan (индекс и модели не нужны).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

API_KEY = "test-key-123"
ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]


@pytest.fixture
def server_module():
    """Загружает server.py с заданным API_KEY."""
    import importlib
    import os

    os.environ["API_KEY"] = API_KEY
    os.environ.pop("API_KEY_REQUIRED", None)
    import hipporag.search_api.server as srv

    importlib.reload(srv)
    yield srv
    os.environ.pop("API_KEY", None)
    importlib.reload(srv)


def _run_guard(srv, headers, method="GET"):
    """Синхронная обёртка над async api_key_guard."""
    request = MagicMock()
    request.method = method
    request.headers = headers
    request.url = MagicMock()
    request.url.path = "/probe"
    call_next = AsyncMock(return_value="PASS_THROUGH")
    return asyncio.run(srv.api_key_guard(request, call_next))


def _client(srv):
    """TestClient без lifespan: индекс и модели не поднимаются."""
    from fastapi.testclient import TestClient

    return TestClient(srv.app)


# --- guard: базовые ветки -------------------------------------------------


def test_correct_key_passes(server_module):
    r = _run_guard(server_module, {"X-API-Key": API_KEY})
    assert r == "PASS_THROUGH"


def test_wrong_key_401(server_module):
    r = _run_guard(server_module, {"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_missing_key_401(server_module):
    r = _run_guard(server_module, {})
    assert r.status_code == 401


def test_preflight_options_bypasses_key(server_module):
    """D2: OPTIONS preflight проходит без ключа."""
    r = _run_guard(server_module, {"Origin": ALLOWED_ORIGINS[0]}, method="OPTIONS")
    assert r == "PASS_THROUGH"


def test_non_ascii_api_key_does_not_crash(server_module):
    """N1: не-ASCII API_KEY не вызывает TypeError → не 500."""
    server_module.API_KEY = "секретный-ключ-с-кириллицей"
    r = _run_guard(server_module, {"X-API-Key": "ascii-header"})
    assert r.status_code == 401  # не 500
    r = _run_guard(server_module, {"X-API-Key": "секретный-ключ-с-кириллицей"})
    assert r == "PASS_THROUGH"  # точное совпадение → pass-through


def test_compare_digest_bytes_works_with_non_ascii():
    """N1 фикс: compare_digest работает с байтами любого содержимого."""
    import secrets

    a = "секретный-ключ".encode("utf-8")
    b = "секретный-ключ".encode("utf-8")
    c = "другой-ключ".encode("utf-8")
    assert secrets.compare_digest(a, b) is True
    assert secrets.compare_digest(a, c) is False


# --- D3/D7: требование ключа ---------------------------------------------


def test_api_key_required_by_default(server_module):
    """D3/D7: по умолчанию ключ обязателен."""
    assert server_module.API_KEY_REQUIRED is True


def test_no_api_key_is_refused_when_required(server_module):
    """D3/D7: без ключа при API_KEY_REQUIRED=1 запрос не обслуживается."""
    server_module.API_KEY = None
    r = _run_guard(server_module, {})
    assert r.status_code == 503
    assert "API_KEY" in r.body.decode("utf-8")


def test_no_api_key_fail_open_only_when_explicitly_allowed(server_module):
    """D3/D7: fail-open — только при явном API_KEY_REQUIRED=0."""
    server_module.API_KEY = None
    server_module.API_KEY_REQUIRED = False
    r = _run_guard(server_module, {})
    assert r == "PASS_THROUGH"


def test_api_key_required_opt_out_is_read_from_env(server_module):
    """API_KEY_REQUIRED=0 читается из окружения."""
    import importlib
    import os

    os.environ["API_KEY"] = API_KEY
    os.environ["API_KEY_REQUIRED"] = "0"
    try:
        import hipporag.search_api.server as srv

        importlib.reload(srv)
        assert srv.API_KEY_REQUIRED is False
    finally:
        os.environ.pop("API_KEY_REQUIRED", None)


# --- D5: аудит отказов ----------------------------------------------------


def test_auth_failures_counter_increments(server_module):
    """D5: каждый отказ увеличивает счётчик."""
    before = server_module.auth_failures
    _run_guard(server_module, {})
    _run_guard(server_module, {"X-API-Key": "wrong"})
    assert server_module.auth_failures == before + 2


def test_health_exposes_auth_state(server_module):
    """D5: /health показывает режим аутентификации и счётчик отказов."""
    client = _client(server_module)
    _run_guard(server_module, {"X-API-Key": "wrong"})
    body = client.get("/health", headers={"X-API-Key": API_KEY}).json()
    assert body["auth_required"] is True
    assert body["auth_failures"] >= 1


# --- N2/N5 + D1: CORS ------------------------------------------------------


def test_401_carries_request_origin(server_module):
    """N2/N5: origin в ответе об отказе соответствует origin запроса."""
    client = _client(server_module)
    for origin in ALLOWED_ORIGINS:
        response = client.get("/health", headers={"Origin": origin, "X-API-Key": "wrong"})
        assert response.status_code == 401
        assert response.headers.get("access-control-allow-origin") == origin


def test_401_has_no_origin_for_foreign_request(server_module):
    """N2/N5: чужой origin не получает разрешающего заголовка."""
    client = _client(server_module)
    response = client.get(
        "/health", headers={"Origin": "https://evil.example.com", "X-API-Key": "wrong"}
    )
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") is None


def test_preflight_allows_api_key_header(server_module):
    """D1: preflight разрешает заголовок X-API-Key."""
    client = _client(server_module)
    response = client.options(
        "/search",
        headers={
            "Origin": ALLOWED_ORIGINS[0],
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-api-key",
        },
    )
    assert response.status_code == 200
    assert "x-api-key" in (response.headers.get("access-control-allow-headers") or "").lower()


# --- Z-1: порядок middleware (CORS — внешний слой) ------------------------


def test_cors_middleware_is_outermost_layer(server_module):
    """Z-1: CORS зарегистрирован после guard → в Starlette он внешний слой.

    Starlette добавляет middleware в начало списка, поэтому последний
    зарегистрированный оказывается внешним. `add_middleware(CORSMiddleware)`
    стоит в `server.py` ниже декоратора `@app.middleware("http")` — значит,
    CORS обрамляет guard, а не наоборот. Тест фиксирует сам порядок, чтобы
    непреднамеренная перестановка регистраций была видна сразу.
    """
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware

    classes = [m.cls for m in server_module.app.user_middleware]
    assert classes[0] is CORSMiddleware, f"CORS должен быть внешним слоем, порядок: {classes}"
    assert (
        classes[1] is BaseHTTPMiddleware
    ), f"guard должен быть внутренним слоем, порядок: {classes}"


def test_401_still_gets_cors_headers_which_proves_layer_order(server_module):
    """Z-1 (поведение): CORS-заголовки на 401 возможны только если CORS снаружи.

    Если бы guard был внешним слоем, его `JSONResponse` ушёл бы мимо
    CORSMiddleware и заголовков `access-control-allow-origin` в 401 не было бы.
    """
    client = _client(server_module)
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGINS[0], "X-API-Key": "wrong"})
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGINS[0]


def test_guard_has_no_own_origin_list(server_module):
    """Z-1: список origin — единственный источник (в CORS), guard его не дублирует.

    Дублирование списка внутри guard — источник расхождений (бывший N2/N5),
    поэтому фиксируем запрет явно.
    """
    import inspect

    source = inspect.getsource(server_module.api_key_guard)
    assert "ALLOWED_ORIGINS" not in source
    assert "localhost" not in source
    assert "allow_origins" not in source
