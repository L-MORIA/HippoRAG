"""Unit tests for HippoRAG exceptions module."""

import pytest

from hipporag.exceptions import (
    ConfigurationError,
    EmbeddingException,
    EmbeddingInitializationError,
    EnvironmentVariableError,
    HippoRAGException,
    InvalidConfigValue,
    LLMAuthenticationError,
    LLMException,
    LLMInitializationError,
    MissingConfigParameter,
    PDFParsingError,
    PDFProcessingException,
)


class TestExceptionHierarchy:
    """Test exception hierarchy and inheritance."""

    def test_hipporag_exception_is_base(self):
        """Test that HippoRAGException is the base exception."""
        exc = HippoRAGException("test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "test error"

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from HippoRAGException."""
        exc = ConfigurationError("config error")
        assert isinstance(exc, HippoRAGException)
        assert isinstance(exc, Exception)

    def test_llm_exception_inheritance(self):
        """Test LLM exceptions inherit correctly."""
        exc = LLMInitializationError("init failed")
        assert isinstance(exc, LLMException)
        assert isinstance(exc, HippoRAGException)

    def test_embedding_exception_inheritance(self):
        """Test embedding exceptions inherit correctly."""
        exc = EmbeddingInitializationError("init failed")
        assert isinstance(exc, EmbeddingException)
        assert isinstance(exc, HippoRAGException)

    def test_pdf_exception_inheritance(self):
        """Test PDF exceptions inherit correctly."""
        exc = PDFParsingError("parsing failed")
        assert isinstance(exc, PDFProcessingException)
        assert isinstance(exc, HippoRAGException)


class TestConfigurationExceptions:
    """Test configuration-related exceptions."""

    def test_invalid_config_value(self):
        """Test InvalidConfigValue exception."""
        with pytest.raises(InvalidConfigValue):
            raise InvalidConfigValue("Invalid value for parameter X")

    def test_missing_config_parameter(self):
        """Test MissingConfigParameter exception."""
        with pytest.raises(MissingConfigParameter):
            raise MissingConfigParameter("Required parameter Y is missing")

    def test_configuration_error_catch_all(self):
        """Test catching ConfigurationError catches all config errors."""
        with pytest.raises(ConfigurationError):
            raise InvalidConfigValue("test")


class TestLLMExceptions:
    """Test LLM-related exceptions."""

    def test_llm_initialization_error(self):
        """Test LLMInitializationError exception."""
        with pytest.raises(LLMInitializationError):
            raise LLMInitializationError("Failed to initialize LLM")

    def test_llm_authentication_error(self):
        """Test LLMAuthenticationError exception."""
        with pytest.raises(LLMAuthenticationError):
            raise LLMAuthenticationError("Invalid API key")

    def test_llm_exception_catch_all(self):
        """Test catching LLMException catches all LLM errors."""
        with pytest.raises(LLMException):
            raise LLMAuthenticationError("test")


class TestEnvironmentExceptions:
    """Test environment-related exceptions."""

    def test_environment_variable_error(self):
        """Test EnvironmentVariableError exception."""
        with pytest.raises(EnvironmentVariableError):
            raise EnvironmentVariableError("Environment variable not set")

    def test_exception_message_preservation(self):
        """Test that exception messages are preserved."""
        msg = "Specific error message"
        exc = EnvironmentVariableError(msg)
        assert str(exc) == msg
