"""Загрузчик словаря синонимов (Query Expansion).

Словарь — данные, а не код: хранится в data/synonyms.json ВНУТРИ пакета
(Возражение 2 рецензии: data/ корпуса в .gitignore, словарь обязан уезжать
с репозиторием и в дистрибутив). Доступ через importlib.resources — работает
и из исходников, и из wheel.
"""

import json
from importlib import resources
from typing import Dict, List


def load_synonyms() -> Dict[str, List[str]]:
    """Возвращает SYNONYM_MAP: ключ — сущность, значение — список форм/синонимов."""
    with resources.files("astro_app.data").joinpath("synonyms.json").open(encoding="utf-8") as f:
        return json.load(f)
