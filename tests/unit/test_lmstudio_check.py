"""Unit-тесты проверки LM Studio (H2-3).

Проверяется ``scripts/check_lmstudio.py``: разбор ответа ``/v1/models``,
диагностика ошибок и CLI. Сеть не используется — HTTP-слой подменяется
фейковым opener'ом.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_lmstudio.py"


def load_script() -> ModuleType:
    """Загрузить скрипт как модуль (``scripts/`` не является пакетом)."""
    name = "check_lmstudio_h2_3"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


check = load_script()


class FakeResponse:
    """Минимальный ответ urllib для тестов."""

    def __init__(self, payload=None, *, raw: bytes | None = None) -> None:
        self._raw = raw if raw is not None else json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def opener_for(payload=None, *, raw: bytes | None = None):
    """Вернуть opener и журнал вызовов."""
    calls: dict[str, object] = {}

    def opener(request, timeout):
        calls["url"] = request.full_url
        calls["timeout"] = timeout
        return FakeResponse(payload, raw=raw)

    return opener, calls


def test_normalize_base_url_strips_spaces_and_slash():
    assert check.normalize_base_url(" http://localhost:1234/v1/ ") == ("http://localhost:1234/v1")
    assert check.normalize_base_url("") == ""


def test_fetch_models_reads_ids_and_ignores_bad_items():
    opener, calls = opener_for(
        {
            "data": [
                {"id": "google/gemma-4-e4b"},
                {"id": "qwen"},
                {"id": 42},
                "not-a-dict",
            ]
        }
    )

    models = check.fetch_models("http://localhost:1234/v1/", timeout=2.5, opener=opener)

    assert models == ["google/gemma-4-e4b", "qwen"]
    assert calls == {
        "url": "http://localhost:1234/v1/models",
        "timeout": 2.5,
    }


def test_fetch_models_rejects_non_json():
    opener, _ = opener_for(raw=b"not json")

    try:
        check.fetch_models("http://localhost:1234/v1", opener=opener)
    except ValueError as error:
        assert "не является JSON" in str(error)
    else:
        raise AssertionError("ожидалась ValueError")


def test_fetch_models_rejects_payload_without_data_list():
    opener, _ = opener_for({"models": []})

    try:
        check.fetch_models("http://localhost:1234/v1", opener=opener)
    except ValueError as error:
        assert "не содержит списка data" in str(error)
    else:
        raise AssertionError("ожидалась ValueError")


def test_check_reports_success_and_found_model():
    opener, _ = opener_for({"data": [{"id": "google/gemma-4-e4b"}]})

    result = check.check_lmstudio(
        "http://localhost:1234/v1/",
        "google/gemma-4-e4b",
        opener=opener,
    )

    assert result.ok is True
    assert result.base_url == "http://localhost:1234/v1"
    assert result.model == "google/gemma-4-e4b"
    assert result.models == ("google/gemma-4-e4b",)
    assert "найдена" in result.message


def test_check_without_model_only_requires_non_empty_list():
    opener, _ = opener_for({"data": [{"id": "a"}, {"id": "b"}]})

    result = check.check_lmstudio("http://localhost:1234/v1", None, opener=opener)

    assert result.ok is True
    assert result.model is None
    assert "моделей: 2" in result.message


def test_check_reports_missing_model_with_available_list():
    opener, _ = opener_for({"data": [{"id": "a"}, {"id": "b"}]})

    result = check.check_lmstudio(
        "http://localhost:1234/v1",
        "google/gemma-4-e4b",
        opener=opener,
    )

    assert result.ok is False
    assert "не найдена" in result.message
    assert "a, b" in result.message


def test_check_reports_empty_model_list():
    opener, _ = opener_for({"data": []})

    result = check.check_lmstudio("http://localhost:1234/v1", None, opener=opener)

    assert result.ok is False
    assert "ни одной модели" in result.message


def test_check_reports_http_error():
    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    result = check.check_lmstudio("http://localhost:1234/v1", None, opener=opener)

    assert result.ok is False
    assert "HTTP 404" in result.message


def test_check_reports_connection_error():
    def opener(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    result = check.check_lmstudio("http://localhost:1234/v1", None, opener=opener)

    assert result.ok is False
    assert "недоступна" in result.message
    assert "refused" in result.message


def test_check_reports_invalid_json_as_failure():
    opener, _ = opener_for(raw=b"<html>")

    result = check.check_lmstudio("http://localhost:1234/v1", None, opener=opener)

    assert result.ok is False
    assert "не является JSON" in result.message


def test_env_timeout_reads_positive_value(monkeypatch):
    monkeypatch.setenv("STARTUP_CHECK_TIMEOUT", "2.5")

    assert check._env_timeout() == 2.5


def test_env_timeout_falls_back_on_invalid_or_empty(monkeypatch):
    monkeypatch.setenv("STARTUP_CHECK_TIMEOUT", "abc")
    assert check._env_timeout() == 5.0

    monkeypatch.setenv("STARTUP_CHECK_TIMEOUT", "  ")
    assert check._env_timeout() == 5.0

    monkeypatch.delenv("STARTUP_CHECK_TIMEOUT", raising=False)
    assert check._env_timeout() == 5.0


def test_env_timeout_clamps_to_minimum(monkeypatch):
    monkeypatch.setenv("STARTUP_CHECK_TIMEOUT", "-3")

    assert check._env_timeout() == 0.1


def test_parser_defaults_come_from_environment(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LLM_API_NAME", "custom/model")
    monkeypatch.setenv("STARTUP_CHECK_TIMEOUT", "7")

    args = check.build_parser().parse_args([])

    assert args.base_url == "http://127.0.0.1:9999/v1"
    assert args.model == "custom/model"
    assert args.timeout == 7.0


def test_cli_json_success(monkeypatch, capsys):
    def fake(base_url, model, *, timeout):
        return check.CheckResult(True, base_url, model, ("m",), "ok")

    monkeypatch.setattr(check, "check_lmstudio", fake)

    code = check.main(["--json", "--base-url", "http://localhost:1234/v1", "--model", "m"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["models"] == ["m"]


def test_cli_no_model_check_passes_none(monkeypatch):
    seen = {}

    def fake(base_url, model, *, timeout):
        seen["model"] = model
        return check.CheckResult(True, base_url, model, ("m",), "ok")

    monkeypatch.setattr(check, "check_lmstudio", fake)

    assert check.main(["--no-model-check", "--quiet"]) == 0
    assert seen["model"] is None


def test_cli_quiet_hides_success_but_not_failure(monkeypatch, capsys):
    def success(base_url, model, *, timeout):
        return check.CheckResult(True, base_url, model, ("m",), "ok")

    monkeypatch.setattr(check, "check_lmstudio", success)
    assert check.main(["--quiet"]) == 0
    assert capsys.readouterr().out == ""

    def failure(base_url, model, *, timeout):
        return check.CheckResult(False, base_url, model, (), "нет связи")

    monkeypatch.setattr(check, "check_lmstudio", failure)
    assert check.main(["--quiet"]) == 1
    assert "нет связи" in capsys.readouterr().out
