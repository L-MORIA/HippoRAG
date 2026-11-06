"""Unit-тесты one-command запуска (H2-3).

Проверяется ``scripts/start.py``: разбор ``.env``, выбор режима API-ключа,
проверки Python/индекса/LM Studio и сборка команды uvicorn. Тяжёлые
зависимости и сеть не используются.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_PATH = PROJECT_ROOT / "scripts" / "start.py"


def load_start() -> ModuleType:
    """Загрузить ``scripts/start.py`` как модуль."""
    name = "start_h2_3"
    spec = importlib.util.spec_from_file_location(name, START_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить {START_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


start = load_start()

from check_lmstudio import CheckResult  # noqa: E402  (scripts/ уже в sys.path)


def test_env_flag_reads_truthy_and_falsy_values(monkeypatch):
    monkeypatch.setenv("H2_3_FLAG", "1")
    assert start.env_flag("H2_3_FLAG") is True

    for value in ("0", "false", "no", "off", " FALSE "):
        monkeypatch.setenv("H2_3_FLAG", value)
        assert start.env_flag("H2_3_FLAG") is False

    monkeypatch.delenv("H2_3_FLAG", raising=False)
    assert start.env_flag("H2_3_FLAG") is False
    assert start.env_flag("H2_3_FLAG", default=True) is True


def test_env_int_and_float_fall_back_on_invalid(monkeypatch):
    monkeypatch.setenv("H2_3_INT", "42")
    monkeypatch.setenv("H2_3_FLOAT", "2.5")
    assert start.env_int("H2_3_INT", 7) == 42
    assert start.env_float("H2_3_FLOAT", 7.0) == 2.5

    monkeypatch.setenv("H2_3_INT", "not-a-number")
    monkeypatch.setenv("H2_3_FLOAT", "not-a-number")
    assert start.env_int("H2_3_INT", 7) == 7
    assert start.env_float("H2_3_FLOAT", 7.0) == 7.0

    monkeypatch.setenv("H2_3_FLOAT", "-1")
    assert start.env_float("H2_3_FLOAT", 7.0) == 0.1


def test_load_env_file_parses_comments_export_and_quotes(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# комментарий",
                "",
                "LLM_NAME=gemma-4-e4b",
                'export API_KEY="secret"',
                "SINGLE='quoted'",
                "EMPTY=",
                "NO_EQUALS",
            ]
        ),
        encoding="utf-8",
    )

    values = start.load_env_file(env_file)

    assert values == {
        "LLM_NAME": "gemma-4-e4b",
        "API_KEY": "secret",
        "SINGLE": "quoted",
        "EMPTY": "",
    }


def test_load_env_file_returns_empty_for_missing_file(tmp_path):
    assert start.load_env_file(tmp_path / "missing.env") == {}


def test_apply_env_does_not_override_existing(monkeypatch):
    monkeypatch.setenv("H2_3_EXISTING", "old")
    monkeypatch.delenv("H2_3_NEW", raising=False)

    start.apply_env({"H2_3_EXISTING": "new", "H2_3_NEW": "value"})

    assert start.os.environ["H2_3_EXISTING"] == "old"
    assert start.os.environ["H2_3_NEW"] == "value"


def test_ensure_env_file_creates_from_example(tmp_path):
    (tmp_path / ".env.example").write_text("LLM_NAME=demo\n", encoding="utf-8")

    path, created = start.ensure_env_file(tmp_path)

    assert created is True
    assert path == tmp_path / ".env"
    assert path.read_text(encoding="utf-8") == "LLM_NAME=demo\n"


def test_ensure_env_file_keeps_existing_file(tmp_path):
    (tmp_path / ".env.example").write_text("LLM_NAME=demo\n", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("LLM_NAME=my-own\n", encoding="utf-8")

    path, created = start.ensure_env_file(tmp_path)

    assert created is False
    assert path == env_path
    assert path.read_text(encoding="utf-8") == "LLM_NAME=my-own\n"


def test_ensure_env_file_without_example(tmp_path):
    path, created = start.ensure_env_file(tmp_path)

    assert created is False
    assert path == tmp_path / ".env"
    assert not path.exists()


def test_ensure_env_file_survives_read_only_directory(tmp_path, monkeypatch):
    (tmp_path / ".env.example").write_text("LLM_NAME=demo\n", encoding="utf-8")

    def fail_write(self, *args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(start.Path, "write_text", fail_write)

    path, created = start.ensure_env_file(tmp_path)

    assert created is False
    assert path == tmp_path / ".env"
    assert not path.exists()


def test_decide_api_key_allows_explicit_key():
    decision = start.decide_api_key("0.0.0.0", "secret", required=True)

    assert decision.allowed is True
    assert decision.fail_open is False
    assert "задан" in decision.message


def test_decide_api_key_honours_explicit_opt_out():
    decision = start.decide_api_key("0.0.0.0", None, required=False)

    assert decision.allowed is True
    assert decision.fail_open is False
    assert "API_KEY_REQUIRED=0" in decision.message


def test_decide_api_key_allows_loopback_without_key():
    for host in ("127.0.0.1", "localhost", "::1", " LOCALHOST "):
        decision = start.decide_api_key(host, None, required=True)
        assert decision.allowed is True
        assert decision.fail_open is True


def test_decide_api_key_refuses_non_loopback_without_key():
    decision = start.decide_api_key("0.0.0.0", None, required=True)

    assert decision.allowed is False
    assert decision.fail_open is False
    assert "вне loopback" in decision.message


def test_build_uvicorn_command_uses_current_interpreter():
    command = start.build_uvicorn_command("127.0.0.1", 8123, reload=True)

    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "uvicorn"]
    assert "astro_app.web:app" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8123"
    assert command[-1] == "--reload"


def test_check_python_version_bounds():
    assert start.check_python_version((3, 9))[0] is False
    assert start.check_python_version((3, 10))[0] is True
    assert start.check_python_version((3, 12))[0] is True

    ok, message = start.check_python_version((3, 13))
    assert ok is True
    assert "проверен на 3.10–3.12" in message


def test_index_status_reports_existing_and_missing_directory(tmp_path):
    ok, message = start.index_status(str(tmp_path))
    assert ok is True
    assert str(tmp_path) in message

    missing = tmp_path / "missing"
    ok, message = start.index_status(str(missing))
    assert ok is False
    assert "не найден" in message


def prepare_main_env(monkeypatch, tmp_path, *, check_lm="0", api_key_required="0"):
    """Общие переменные для тестов main()."""
    monkeypatch.setattr(start, "ROOT", tmp_path)
    monkeypatch.setenv("STARTUP_CHECK_LM", check_lm)
    monkeypatch.setenv("API_KEY_REQUIRED", api_key_required)
    monkeypatch.setenv("SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("LLM_API_NAME", "demo")
    monkeypatch.delenv("API_KEY", raising=False)


def test_main_check_only_success(monkeypatch, tmp_path, capsys):
    prepare_main_env(monkeypatch, tmp_path)
    monkeypatch.setattr(start.importlib.util, "find_spec", lambda name: object())

    code = start.main(["--check-only", "--skip-lm-check"])

    assert code == 0
    assert "Проверки пройдены" in capsys.readouterr().out


def test_main_uses_app_host_and_port_from_env(monkeypatch, tmp_path):
    prepare_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("APP_PORT", "8124")
    monkeypatch.setattr(start.importlib.util, "find_spec", lambda name: object())
    captured = {}

    def fake_call(command, cwd=None, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        return 0

    monkeypatch.setattr(start.subprocess, "call", fake_call)

    code = start.main([])

    assert code == 0
    assert captured["command"][captured["command"].index("--host") + 1] == "127.0.0.1"
    assert captured["command"][captured["command"].index("--port") + 1] == "8124"
    assert captured["cwd"] == str(tmp_path)


def test_main_refuses_non_loopback_without_key(monkeypatch, tmp_path, capsys):
    prepare_main_env(monkeypatch, tmp_path, api_key_required="1")

    code = start.main(["--check-only", "--host", "0.0.0.0", "--skip-lm-check"])

    assert code == 1
    assert "вне loopback" in capsys.readouterr().out


def test_main_enables_fail_open_for_loopback(monkeypatch, tmp_path):
    prepare_main_env(monkeypatch, tmp_path, api_key_required="1")
    monkeypatch.setattr(start.importlib.util, "find_spec", lambda name: object())
    captured = {}

    def fake_call(command, cwd=None, env=None):
        captured["env"] = env
        return 0

    monkeypatch.setattr(start.subprocess, "call", fake_call)

    code = start.main(["--host", "127.0.0.1", "--port", "8125"])

    assert code == 0
    assert captured["env"]["API_KEY_REQUIRED"] == "0"


def test_main_reports_lm_check_failure(monkeypatch, tmp_path, capsys):
    prepare_main_env(monkeypatch, tmp_path, check_lm="1")

    def fake_check(base_url, model, *, timeout):
        return CheckResult(False, base_url, model, (), "LM Studio недоступна")

    monkeypatch.setattr(start, "check_lmstudio", fake_check)

    code = start.main(["--check-only"])

    assert code == 1
    output = capsys.readouterr().out
    assert "LM Studio недоступна" in output
    assert "--skip-lm-check" in output


def test_main_reports_missing_uvicorn(monkeypatch, tmp_path, capsys):
    prepare_main_env(monkeypatch, tmp_path)
    monkeypatch.setattr(start.importlib.util, "find_spec", lambda name: None)

    code = start.main(["--check-only"])

    assert code == 1
    assert "uvicorn не установлен" in capsys.readouterr().out


def test_main_rejects_invalid_port(monkeypatch, tmp_path, capsys):
    prepare_main_env(monkeypatch, tmp_path)

    code = start.main(["--check-only", "--port", "0"])

    assert code == 1
    assert "Некорректный порт" in capsys.readouterr().out
