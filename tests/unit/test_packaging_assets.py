"""Контрактные тесты упаковки H2-3.

Проверяется, что артефакты one-command запуска и Docker не расходятся:
CPU-only образ, публичный healthcheck, bind-mount данных, безопасный порт,
переменные ``.env.example`` и исключения ``.dockerignore``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def read(name: str) -> str:
    return (PROJECT_ROOT / name).read_text(encoding="utf-8")


def test_required_packaging_files_exist():
    for name in (
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "start.bat",
        "start.sh",
        "scripts/start.py",
        "scripts/check_lmstudio.py",
        ".env.example",
    ):
        assert (PROJECT_ROOT / name).is_file(), f"нет файла {name}"


def test_dockerfile_is_cpu_only_and_uses_public_healthcheck():
    text = read("Dockerfile")

    assert "FROM python:3.11-slim" in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "torch==${TORCH_VERSION}" in text
    assert "USER appuser" in text
    assert "HEALTHCHECK" in text
    assert "/ui/config" in text
    assert "start-period=600s" in text
    assert "scripts/start.py" in text
    assert "0.0.0.0" in text


def test_dockerfile_copies_runtime_sources():
    text = read("Dockerfile")

    for fragment in (
        "COPY pyproject.toml README_RU.md .env.example constraints.txt ./",
        "COPY src ./src",
        "COPY scripts ./scripts",
        "COPY main.py ./",
        "pip install . -c constraints.txt",
    ):
        assert fragment in text


def test_compose_binds_only_loopback_and_mounts_data():
    compose = yaml.safe_load(read("docker-compose.yml"))
    service = compose["services"]["hipporag"]

    assert service["ports"] == ["127.0.0.1:8000:8000"]
    assert service["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert service["user"] == "${DOCKER_UID:-10001}:${DOCKER_GID:-10001}"

    volumes = service["volumes"]
    for volume in (
        "./outputs:/app/outputs",
        "./data:/app/data:ro",
        "./temp_uploads:/app/temp_uploads",
        "./cache:/app/cache",
    ):
        assert volume in volumes


def test_compose_uses_docker_specific_llm_url_and_safe_defaults():
    compose = yaml.safe_load(read("docker-compose.yml"))
    environment = compose["services"]["hipporag"]["environment"]

    assert (
        environment["LLM_BASE_URL"] == "${LLM_BASE_URL_DOCKER:-http://host.docker.internal:1234/v1}"
    )
    assert environment["API_KEY_REQUIRED"] == "${API_KEY_REQUIRED:-0}"
    assert environment["STARTUP_CHECK_LM"] == "${STARTUP_CHECK_LM:-1}"
    assert environment["APP_HOST"] == "0.0.0.0"
    assert environment["APP_PORT"] == "8000"
    assert environment["HF_HOME"] == "/app/cache/huggingface"


def test_compose_healthcheck_uses_public_ui_config():
    compose = yaml.safe_load(read("docker-compose.yml"))
    healthcheck = compose["services"]["hipporag"]["healthcheck"]

    assert healthcheck["start_period"] == "600s"
    assert "/ui/config" in healthcheck["test"][1]


def test_dockerignore_excludes_secrets_data_and_heavy_artifacts():
    lines = {
        line.strip()
        for line in read(".dockerignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    for item in (
        ".git",
        ".venv",
        ".env",
        "outputs",
        "data",
        "cache",
        "temp_uploads",
        "tests",
        "eval",
        "docs",
    ):
        assert item in lines
    assert "!.env.example" in lines


def test_dockerignore_keeps_runtime_sources():
    lines = {
        line.strip()
        for line in read(".dockerignore").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    for item in ("pyproject.toml", "README_RU.md", "src", "scripts", "main.py"):
        assert item not in lines


def test_env_example_contains_startup_and_docker_settings():
    text = read(".env.example")
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()

    assert values["APP_HOST"] == "127.0.0.1"
    assert values["APP_PORT"] == "8000"
    assert values["STARTUP_CHECK_LM"] == "1"
    assert values["STARTUP_CHECK_TIMEOUT"] == "5"
    assert "LLM_BASE_URL_DOCKER" in text
    assert "host.docker.internal" in text
    assert "DOCKER_UID" in text
    assert "DOCKER_GID" in text


def test_gitignore_excludes_root_cache_directory():
    lines = {line.strip() for line in read(".gitignore").splitlines()}

    assert "/cache/" in lines


def test_start_scripts_install_venv_and_call_start_py():
    bat = read("start.bat")
    sh = read("start.sh")

    assert "scripts\\start.py" in bat
    assert "scripts/start.py" in sh
    assert ".venv" in bat
    assert ".venv" in sh
    assert "3.10" in bat and "3.12" in bat
    assert "3.10" in sh and "3.12" in sh
    assert sh.startswith("#!/usr/bin/env bash")
