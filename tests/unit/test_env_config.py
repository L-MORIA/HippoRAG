"""Unit tests for HippoRAG environment configuration module."""

from pathlib import Path

import pytest

from hipporag.utils.env_config import (
    APIConfig,
    EmbeddingConfig,
    HippoRAGConfig,
    LLMConfig,
    LoggingConfig,
    PDFConfig,
    VectorStoreConfig,
    get_env_var,
)


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_llm_config(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_invalid_llm_provider(self):
        """Test invalid LLM provider raises error."""
        with pytest.raises(ValueError):
            LLMConfig(provider="invalid_provider")

    def test_valid_llm_providers(self):
        """Test all valid LLM providers."""
        valid_providers = [
            "openai",
            "azure",
            "bedrock",
            "transformers",
            "vllm",
            "cohere",
            "gritlm",
            "lm_studio",
        ]
        for provider in valid_providers:
            config = LLMConfig(provider=provider)
            assert config.provider == provider

    def test_temperature_bounds(self):
        """Test temperature parameter bounds."""
        config = LLMConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = LLMConfig(temperature=2.0)
        assert config.temperature == 2.0

        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)

        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)


class TestEmbeddingConfig:
    """Test embedding configuration."""

    def test_default_embedding_config(self):
        """Test default embedding configuration."""
        config = EmbeddingConfig()
        assert config.provider == "sentence-transformers"
        assert config.model == "all-MiniLM-L6-v2"
        assert config.dimension == 384
        assert config.batch_size == 32

    def test_invalid_embedding_provider(self):
        """Test invalid embedding provider raises error."""
        with pytest.raises(ValueError):
            EmbeddingConfig(provider="invalid_provider")

    def test_valid_embedding_providers(self):
        """Test all valid embedding providers."""
        valid_providers = [
            "sentence-transformers",
            "gritlm",
            "contriever",
            "cohere",
            "nvembed",
        ]
        for provider in valid_providers:
            config = EmbeddingConfig(provider=provider)
            assert config.provider == provider


class TestVectorStoreConfig:
    """Test vector store configuration."""

    def test_default_vector_store_config(self):
        """Test default vector store configuration."""
        config = VectorStoreConfig()
        assert config.type == "faiss"
        assert config.collection_name == "hipporag"

    def test_invalid_vector_store_type(self):
        """Test invalid vector store type raises error."""
        with pytest.raises(ValueError):
            VectorStoreConfig(type="invalid_type")

    def test_valid_vector_store_types(self):
        """Test all valid vector store types."""
        valid_types = ["faiss", "milvus", "weaviate", "pinecone", "qdrant"]
        for store_type in valid_types:
            config = VectorStoreConfig(type=store_type)
            assert config.type == store_type


class TestPDFConfig:
    """Test PDF configuration."""

    def test_default_pdf_config(self):
        """Test default PDF configuration."""
        config = PDFConfig()
        assert config.parser == "text"
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_invalid_pdf_parser(self):
        """Test invalid PDF parser raises error."""
        with pytest.raises(ValueError):
            PDFConfig(parser="invalid_parser")

    def test_valid_pdf_parsers(self):
        """Test all valid PDF parsers."""
        valid_parsers = ["text", "ocr", "hybrid"]
        for parser in valid_parsers:
            config = PDFConfig(parser=parser)
            assert config.parser == parser


class TestLoggingConfig:
    """Test logging configuration."""

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "text"

    def test_invalid_log_level(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")

    def test_valid_log_levels(self):
        """Test all valid log levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_invalid_log_format(self):
        """Test invalid log format raises error."""
        with pytest.raises(ValueError):
            LoggingConfig(format="invalid_format")


class TestAPIConfig:
    """Test API configuration."""

    def test_default_api_config(self):
        """Test default API configuration."""
        config = APIConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 4
        assert config.reload is False

    def test_invalid_port(self):
        """Test invalid port raises error."""
        with pytest.raises(ValueError):
            APIConfig(port=0)

        with pytest.raises(ValueError):
            APIConfig(port=65536)


class TestHippoRAGConfig:
    """Test main HippoRAG configuration."""

    def test_default_hipporag_config(self):
        """Test default HippoRAG configuration."""
        config = HippoRAGConfig()
        assert config.llm.provider == "openai"
        assert config.embedding.provider == "sentence-transformers"
        assert config.vector_store.type == "faiss"
        assert config.top_k == 10
        assert config.rerank_top_k == 5

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = HippoRAGConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "llm" in config_dict
        assert "embedding" in config_dict
        assert "vector_store" in config_dict

    def test_config_get_method(self):
        """Test getting config values with dot notation."""
        config = HippoRAGConfig()
        assert config.get("llm.provider") == "openai"
        assert config.get("embedding.model") == "all-MiniLM-L6-v2"
        assert config.get("top_k") == 10
        assert config.get("nonexistent.key", "default") == "default"

    def test_config_creates_directories(self, tmp_path, monkeypatch):
        """Test that config creates necessary directories."""
        monkeypatch.chdir(tmp_path)
        config = HippoRAGConfig(
            data_dir="./test_data",
            output_dir="./test_output",
            cache_dir="./test_cache",
        )
        assert config.data_dir == "./test_data"
        assert Path("./test_data").exists()
        assert Path("./test_output").exists()
        assert Path("./test_cache").exists()


class TestGetEnvVar:
    """Test get_env_var utility function."""

    def test_get_string_env_var(self, monkeypatch):
        """Test getting string environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        value = get_env_var("TEST_VAR")
        assert value == "test_value"

    def test_get_int_env_var(self, monkeypatch):
        """Test getting integer environment variable."""
        monkeypatch.setenv("TEST_INT", "42")
        value = get_env_var("TEST_INT", var_type=int)
        assert value == 42
        assert isinstance(value, int)

    def test_get_bool_env_var(self, monkeypatch):
        """Test getting boolean environment variable."""
        monkeypatch.setenv("TEST_BOOL_TRUE", "true")
        monkeypatch.setenv("TEST_BOOL_FALSE", "false")
        assert get_env_var("TEST_BOOL_TRUE", var_type=bool) is True
        assert get_env_var("TEST_BOOL_FALSE", var_type=bool) is False

    def test_get_missing_env_var_with_default(self, monkeypatch):
        """Test getting missing environment variable with default."""
        monkeypatch.delenv("MISSING_VAR", raising=False)
        value = get_env_var("MISSING_VAR", default="default_value")
        assert value == "default_value"

    def test_get_required_missing_env_var_raises_error(self, monkeypatch):
        """Test getting required missing environment variable raises error."""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        from hipporag.exceptions import EnvironmentVariableError

        with pytest.raises(EnvironmentVariableError):
            get_env_var("REQUIRED_VAR", required=True)

    def test_get_env_var_type_conversion_error(self, monkeypatch):
        """Test environment variable type conversion error."""
        monkeypatch.setenv("TEST_INVALID_INT", "not_an_int")
        from hipporag.exceptions import EnvironmentVariableError

        with pytest.raises(EnvironmentVariableError):
            get_env_var("TEST_INVALID_INT", var_type=int)
