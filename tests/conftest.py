"""
Pytest configuration and fixtures for HippoRAG tests.

This module provides shared fixtures and configuration for all tests.
"""

import sys
from pathlib import Path
from typing import Generator

import pytest

# tests/standalone — ручные сценарии, а не pytest-тесты (S-42): исключаем из сбора
collect_ignore_glob = ["standalone/*"]

# Add src directory to path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from hipporag.utils.env_config import HippoRAGConfig


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Get test data directory."""
    test_data = project_root / "tests" / "data"
    test_data.mkdir(parents=True, exist_ok=True)
    return test_data


@pytest.fixture(scope="session")
def test_output_dir(project_root: Path) -> Path:
    """Get test output directory."""
    test_output = project_root / "tests" / "outputs"
    test_output.mkdir(parents=True, exist_ok=True)
    return test_output


@pytest.fixture
def config() -> HippoRAGConfig:
    """Get test configuration."""
    # Create a test configuration with minimal settings
    config = HippoRAGConfig(
        data_dir="./tests/data",
        output_dir="./tests/outputs",
        cache_dir="./tests/cache",
        debug_mode=True,
        verbose=True,
    )
    return config


@pytest.fixture
def temp_env(monkeypatch, tmp_path) -> Generator[dict, None, None]:
    """Provide temporary environment variables for testing."""
    env_vars = {
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
        "VECTOR_STORE_TYPE": "faiss",
        "VECTOR_STORE_PATH": str(tmp_path / "vector_store"),
        "DATA_DIR": str(tmp_path / "data"),
        "OUTPUT_DIR": str(tmp_path / "outputs"),
        "LOG_LEVEL": "DEBUG",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    yield env_vars


@pytest.fixture
def mock_llm_response() -> dict:
    """Provide mock LLM response for testing."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the LLM.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture
def mock_embedding() -> list:
    """Provide mock embedding vector for testing."""
    return [0.1] * 384  # 384-dimensional embedding


@pytest.fixture
def sample_documents() -> list:
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "The quick brown fox jumps over the lazy dog.",
            "metadata": {"source": "test.txt", "page": 1},
        },
        {
            "id": "doc2",
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "test.txt", "page": 2},
        },
        {
            "id": "doc3",
            "content": "Python is a popular programming language for data science.",
            "metadata": {"source": "test.txt", "page": 3},
        },
    ]


@pytest.fixture
def sample_questions() -> list:
    """Provide sample questions for testing."""
    return [
        {
            "id": "q1",
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence.",
        },
        {
            "id": "q2",
            "question": "What is Python used for?",
            "answer": "Python is a popular programming language for data science.",
        },
    ]


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_api: mark test as requiring API keys")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
