"""
Custom exception hierarchy for HippoRAG.

This module defines domain-specific exceptions used throughout the HippoRAG framework.
"""


class HippoRAGException(Exception):
    """Base exception for all HippoRAG errors."""

    pass


# Configuration Exceptions
class ConfigurationError(HippoRAGException):
    """Raised when configuration is invalid or missing required parameters."""

    pass


class InvalidConfigValue(ConfigurationError):
    """Raised when a configuration value is invalid."""

    pass


class MissingConfigParameter(ConfigurationError):
    """Raised when a required configuration parameter is missing."""

    pass


# LLM Exceptions
class LLMException(HippoRAGException):
    """Base exception for LLM-related errors."""

    pass


class LLMInitializationError(LLMException):
    """Raised when LLM initialization fails."""

    pass


class LLMInferenceError(LLMException):
    """Raised when LLM inference fails."""

    pass


class LLMProviderNotSupported(LLMException):
    """Raised when requested LLM provider is not supported."""

    pass


class LLMAuthenticationError(LLMException):
    """Raised when LLM authentication fails (invalid API key, etc.)."""

    pass


class LLMRateLimitError(LLMException):
    """Raised when LLM API rate limit is exceeded."""

    pass


# Embedding Exceptions
class EmbeddingException(HippoRAGException):
    """Base exception for embedding-related errors."""

    pass


class EmbeddingModelError(EmbeddingException):
    """Raised when embedding model fails."""

    pass


class EmbeddingProviderNotSupported(EmbeddingException):
    """Raised when requested embedding provider is not supported."""

    pass


class EmbeddingInitializationError(EmbeddingException):
    """Raised when embedding model initialization fails."""

    pass


# PDF Processing Exceptions
class PDFProcessingException(HippoRAGException):
    """Base exception for PDF processing errors."""

    pass


class PDFParsingError(PDFProcessingException):
    """Raised when PDF parsing fails."""

    pass


class OCRError(PDFProcessingException):
    """Raised when OCR processing fails."""

    pass


class InvalidPDFFormat(PDFProcessingException):
    """Raised when PDF format is invalid or corrupted."""

    pass


# Vector Store Exceptions
class VectorStoreException(HippoRAGException):
    """Base exception for vector store errors."""

    pass


class VectorStoreInitializationError(VectorStoreException):
    """Raised when vector store initialization fails."""

    pass


class VectorStoreConnectionError(VectorStoreException):
    """Raised when connection to vector store fails."""

    pass


class VectorStoreOperationError(VectorStoreException):
    """Raised when vector store operation fails."""

    pass


# Information Extraction Exceptions
class ExtractionException(HippoRAGException):
    """Base exception for information extraction errors."""

    pass


class TripleExtractionError(ExtractionException):
    """Raised when triple extraction fails."""

    pass


class EntityExtractionError(ExtractionException):
    """Raised when entity extraction fails."""

    pass


# Retrieval Exceptions
class RetrievalException(HippoRAGException):
    """Base exception for retrieval errors."""

    pass


class RetrievalError(RetrievalException):
    """Raised when document retrieval fails."""

    pass


class RankingError(RetrievalException):
    """Raised when ranking/reranking fails."""

    pass


# Evaluation Exceptions
class EvaluationException(HippoRAGException):
    """Base exception for evaluation errors."""

    pass


class MetricCalculationError(EvaluationException):
    """Raised when metric calculation fails."""

    pass


class InvalidEvaluationDataset(EvaluationException):
    """Raised when evaluation dataset is invalid."""

    pass


# API Exceptions
class APIException(HippoRAGException):
    """Base exception for API errors."""

    pass


class InvalidAPIRequest(APIException):
    """Raised when API request is invalid."""

    pass


class APIServerError(APIException):
    """Raised when API server encounters an error."""

    pass


# Data Exceptions
class DataException(HippoRAGException):
    """Base exception for data-related errors."""

    pass


class InvalidDataFormat(DataException):
    """Raised when data format is invalid."""

    pass


class DataProcessingError(DataException):
    """Raised when data processing fails."""

    pass


class MissingDataError(DataException):
    """Raised when required data is missing."""

    pass


# Utility Exceptions
class UtilityException(HippoRAGException):
    """Base exception for utility function errors."""

    pass


class FileNotFoundError(UtilityException):
    """Raised when required file is not found."""

    pass


class DirectoryNotFoundError(UtilityException):
    """Raised when required directory is not found."""

    pass


class EnvironmentVariableError(UtilityException):
    """Raised when required environment variable is not set."""

    pass
