"""Тесты защиты от пуша в чужой репозиторий.

Проверяется ``scripts/check_push_remote.py``: нормализация URL remote и
решение «пропустить / отклонить». Тесты не запускают git — только чистые
функции, поэтому быстрые и не зависят от окружения.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = PROJECT_ROOT / "scripts" / "check_push_remote.py"


def load_guard() -> ModuleType:
    """Загрузить скрипт-защиту как модуль (scripts/ не является пакетом)."""
    spec = importlib.util.spec_from_file_location("check_push_remote", GUARD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"не удалось загрузить {GUARD_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = load_guard()


def test_own_repository_https_allowed():
    assert guard.is_allowed("https://github.com/L-MORIA/HippoRAG.git")


def test_own_repository_ssh_allowed():
    assert guard.is_allowed("git@github.com:L-MORIA/HippoRAG.git")


def test_own_repository_without_git_suffix_allowed():
    assert guard.is_allowed("https://github.com/L-MORIA/HippoRAG")


def test_own_repository_case_insensitive():
    assert guard.is_allowed("https://github.com/l-moria/hipporag.git")


def test_own_repository_with_token_in_url_allowed():
    assert guard.is_allowed("https://gho_token@github.com/L-MORIA/HippoRAG.git")


def test_ssh_scheme_with_user_allowed():
    assert guard.is_allowed("ssh://git@github.com/L-MORIA/HippoRAG.git")


def test_upstream_rejected():
    assert not guard.is_allowed("https://github.com/osu-nlp-group/hipporag")


def test_upstream_ssh_rejected():
    assert not guard.is_allowed("git@github.com:OSU-NLP-Group/HippoRAG.git")


def test_disabled_placeholder_rejected():
    assert not guard.is_allowed("DISABLED-no-push-to-upstream")


def test_empty_and_local_paths_rejected():
    assert not guard.is_allowed("")
    assert not guard.is_allowed("C:\\repos\\somewhere")


def test_normalize_returns_none_without_two_segments():
    assert guard.normalize_slug("no-slash-at-all") is None


def test_normalize_strips_git_suffix_and_lowercases():
    assert guard.normalize_slug("https://github.com/L-MORIA/HippoRAG.git") == "l-moria/hipporag"


def test_check_rejects_foreign_remote(monkeypatch, capsys):
    monkeypatch.delenv(guard.BYPASS_ENV, raising=False)

    assert guard.check("origin", "https://github.com/osu-nlp-group/hipporag") == 1

    captured = capsys.readouterr()
    assert "ОТКАЗАНО" in captured.err
    assert guard.ALLOWED_SLUG in captured.err


def test_check_allows_own_remote(monkeypatch, capsys):
    monkeypatch.delenv(guard.BYPASS_ENV, raising=False)

    assert guard.check("fork", "https://github.com/L-MORIA/HippoRAG.git") == 0

    assert capsys.readouterr().err == ""


def test_check_honours_bypass_environment(monkeypatch, capsys):
    monkeypatch.setenv(guard.BYPASS_ENV, "1")

    assert guard.check("origin", "https://github.com/osu-nlp-group/hipporag") == 0

    assert "ВНИМАНИЕ" in capsys.readouterr().err
