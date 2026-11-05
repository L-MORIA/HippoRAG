"""Хранилище обратной связи H2-1: append-only JSONL без базы данных.

Одна запись — одна оценка ответа. Файл по умолчанию лежит в ``outputs/``
(каталог в ``.gitignore``), поэтому пользовательские оценки не попадают в
репозиторий. Персональные данные не собираются: нет IP, user-agent и ID
пользователя — только вопрос, ответ, режим и оценка.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import CitationOut

DEFAULT_FEEDBACK_PATH = "outputs/feedback/feedback.jsonl"


class FeedbackStore:
    """Потокобезопасная запись оценок в JSONL."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        configured = path or os.getenv("FEEDBACK_PATH") or DEFAULT_FEEDBACK_PATH
        self.path = Path(configured)
        self._lock = threading.Lock()

    def append(
        self,
        *,
        query: str,
        answer: str,
        mode: str,
        rating: str,
        comment: str = "",
        citations: Optional[List[CitationOut]] = None,
    ) -> Dict[str, Any]:
        """Добавить запись и вернуть её (с id и временем)."""
        record: Dict[str, Any] = {
            "schema_version": 1,
            "id": uuid.uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rating": rating,
            "query": query,
            "answer": answer,
            "mode": mode,
            "comment": comment,
            "citations": [citation.model_dump(mode="json") for citation in (citations or [])],
        }

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> List[Dict[str, Any]]:
        """Прочитать все записи; битые строки пропускаются."""
        if not self.path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records
