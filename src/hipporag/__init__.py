"""
Astrology RAG - Система поиска для астрологических документов
"""

__version__ = "2.0.0-alpha.4"

from .HippoRAG import HippoRAG
from .search_api import app
from .StandardRAG import StandardRAG
from .utils import env_config  # noqa: F401  — загрузка .env при импорте пакета (D-6)

__all__ = ["HippoRAG", "StandardRAG", "app"]
