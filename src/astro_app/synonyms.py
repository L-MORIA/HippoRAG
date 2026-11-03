"""Загрузчик словаря синонимов (Query Expansion).

Словарь — данные, а не код: хранится в data/synonyms.json ВНУТРИ пакета
(Возражение 2 рецензии: data/ корпуса в .gitignore, словарь обязан уезжать
с репозиторием и в дистрибутив). Доступ через importlib.resources — работает
и из исходников, и из wheel.

Перед чтением проверяется контракт версий package-data (H0-8): словарь и
манифест к нему должны соответствовать версии, которую ожидает код. При
расхождении бросается PackageDataError с объяснением, а не тихо читается
чужая раскладка.
"""

import json
from importlib import resources
from typing import Dict, List

from . import versions


def load_synonyms() -> Dict[str, List[str]]:
    """Возвращает SYNONYM_MAP: ключ — сущность, значение — список форм/синонимов."""
    versions.check_package_data()
    with resources.files("astro_app.data").joinpath("synonyms.json").open(encoding="utf-8") as f:
        return json.load(f)
