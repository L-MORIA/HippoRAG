"""
Astrology RAG Search API - современный API для поиска в астрологических документах.
"""

from .models import (
    Citation,
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    IngestStats,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .server import app

__version__ = "1.0.0"

__all__ = [
    "app",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "IngestRequest",
    "IngestResponse",
    "IngestStats",
    "HealthResponse",
    "ErrorResponse",
    "Citation",
]
