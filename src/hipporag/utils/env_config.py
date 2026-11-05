"""
Environment-based configuration management for HippoRAG.

This module provides utilities for loading and managing configuration
from environment variables with type validation and defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..exceptions import (
    EnvironmentVariableError,
    InvalidConfigValue,
)

T = TypeVar("T")

# Load .env file if it exists
ENV_FILE = Path(__file__).parent.parent.parent.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


class LLMConfig(BaseSettings):
    """Configuration for LLM providers."""

    model_config = SettingsConfigDict(env_prefix="LLM_", case_sensitive=False)

    provider: str = Field(default="openai", description="LLM provider type")
    model: str = Field(default="gpt-4", description="Model name/ID")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        valid_providers = {
            "openai",
            "azure",
            "bedrock",
            "transformers",
            "vllm",
            "cohere",
            "gritlm",
            "lm_studio",
        }
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {v}. Must be one of {valid_providers}")
        return v.lower()


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", case_sensitive=False)

    provider: str = Field(default="sentence-transformers")
    model: str = Field(default="all-MiniLM-L6-v2")
    api_key: Optional[str] = Field(default=None)
    dimension: int = Field(default=384, gt=0)
    batch_size: int = Field(default=32, gt=0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate embedding provider."""
        valid_providers = {
            "sentence-transformers",
            "gritlm",
            "contriever",
            "cohere",
            "nvembed",
        }
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid embedding provider: {v}. Must be one of {valid_providers}")
        return v.lower()


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", case_sensitive=False)

    type: str = Field(default="faiss", description="Vector store type")
    path: str = Field(default="./outputs/vector_store")
    host: Optional[str] = Field(default=None)
    port: Optional[int] = Field(default=None)
    collection_name: str = Field(default="hipporag")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate vector store type."""
        valid_types = {"faiss", "milvus", "weaviate", "pinecone", "qdrant"}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid vector store type: {v}. Must be one of {valid_types}")
        return v.lower()


class PDFConfig(BaseSettings):
    """Configuration for PDF processing."""

    model_config = SettingsConfigDict(env_prefix="PDF_", case_sensitive=False)

    parser: str = Field(default="text", description="PDF parser type")
    tesseract_path: Optional[str] = Field(default=None)
    ocr_language: str = Field(default="eng")
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)

    @field_validator("parser")
    @classmethod
    def validate_parser(cls, v: str) -> str:
        """Validate PDF parser type."""
        valid_parsers = {"text", "ocr", "hybrid"}
        if v.lower() not in valid_parsers:
            raise ValueError(f"Invalid PDF parser: {v}. Must be one of {valid_parsers}")
        return v.lower()


class APIConfig(BaseSettings):
    """Configuration for API server."""

    model_config = SettingsConfigDict(env_prefix="API_", case_sensitive=False)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, gt=0, lt=65536)
    workers: int = Field(default=4, gt=0)
    reload: bool = Field(default=False)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    cors_allow_credentials: bool = Field(default=True)


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    model_config = SettingsConfigDict(env_prefix="LOG_", case_sensitive=False)

    level: str = Field(default="INFO")
    file: Optional[str] = Field(default=None)
    format: str = Field(default="text")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = {"json", "text"}
        if v.lower() not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return v.lower()


class HippoRAGConfig(BaseSettings):
    """Main configuration for HippoRAG."""

    # extra="ignore": корневой .env содержит ключи приложения (LLM_NAME,
    # LLM_BASE_URL, SAVE_DIR), которые не являются полями HippoRAGConfig.
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Paths
    data_dir: str = Field(default="./data")
    output_dir: str = Field(default="./outputs")
    cache_dir: str = Field(default="./cache")

    # Processing
    top_k: int = Field(default=10, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)

    # Features
    enable_cache: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, gt=0)
    debug_mode: bool = Field(default=False)
    verbose: bool = Field(default=False)

    def __init__(self, **data: Any) -> None:
        """Initialize configuration with validation."""
        super().__init__(**data)
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validate and create necessary directories."""
        for path_attr in ["data_dir", "output_dir", "cache_dir"]:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "HippoRAGConfig":
        """Load configuration from environment variables."""
        try:
            return cls()
        except Exception as e:
            raise InvalidConfigValue(f"Failed to load configuration from environment: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split(".")
        value = self.model_dump()
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


def get_config() -> HippoRAGConfig:
    """Get global HippoRAG configuration instance."""
    global _config
    if _config is None:
        _config = HippoRAGConfig.from_env()
    return _config


def set_config(config: HippoRAGConfig) -> None:
    """Set global HippoRAG configuration instance."""
    global _config
    _config = config


# Global configuration instance
_config: Optional[HippoRAGConfig] = None


def get_env_var(
    key: str,
    default: Optional[T] = None,
    required: bool = False,
    var_type: Type[T] = str,
) -> Union[T, None]:
    """
    Get environment variable with type conversion and validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        var_type: Type to convert the value to

    Returns:
        Environment variable value or default

    Raises:
        EnvironmentVariableError: If required variable is missing
    """
    value = os.getenv(key)

    if value is None:
        if required:
            raise EnvironmentVariableError(f"Required environment variable not set: {key}")
        return default

    try:
        if var_type == bool:
            return var_type(value.lower() in ("true", "1", "yes", "on"))  # type: ignore
        elif var_type == list:
            return value.split(",")  # type: ignore
        else:
            return var_type(value)  # type: ignore
    except (ValueError, TypeError) as e:
        raise EnvironmentVariableError(
            f"Failed to convert environment variable {key} to {var_type.__name__}: {e}"
        )
