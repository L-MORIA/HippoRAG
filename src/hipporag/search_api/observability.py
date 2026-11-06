"""H2-4: структурные события и метрики в памяти.

Модуль намеренно не зависит от FastAPI: его можно использовать из API,
Web UI и тестов. Метрики хранятся в процессе и сбрасываются при рестарте —
для локального приложения этого достаточно, внешний Prometheus не нужен.

Структурные события пишутся одной JSON-строкой в логгер
``hipporag.events``. В отличие от обычных сообщений, их можно разбирать
машиной: у каждого события есть ``ts``, ``event`` и, если запрос пришёл
через HTTP, ``request_id``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import deque
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Optional

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_request_id: ContextVar[str] = ContextVar("hipporag_request_id", default="")
_event_logger = logging.getLogger("hipporag.events")
_event_logger.setLevel(logging.INFO)
# Событие — уже готовая JSON-строка. Собственный handler без префикса
# оставляет в логе чистый JSON; propagate=False не дублирует его через root.
_event_logger.propagate = False
if not _event_logger.handlers:
    _event_handler = logging.StreamHandler()
    _event_handler.setFormatter(logging.Formatter("%(message)s"))
    _event_logger.addHandler(_event_handler)


def new_request_id() -> str:
    """Новый идентификатор запроса."""
    return uuid.uuid4().hex


def get_request_id() -> str:
    """Идентификатор текущего запроса или пустая строка."""
    return _request_id.get()


def set_request_id(value: str) -> Token:
    """Привязать идентификатор к текущему контексту."""
    return _request_id.set(value)


def reset_request_id(token: Token) -> None:
    """Вернуть контекст к предыдущему значению."""
    _request_id.reset(token)


def request_id_from_header(value: Optional[str]) -> str:
    """Принять клиентский ``X-Request-ID``, если он безопасен.

    Иначе сгенерировать новый: заголовок приходит извне и не должен
    раздувать логи или ломать JSON.
    """
    candidate = (value or "").strip()
    if _VALID_REQUEST_ID.fullmatch(candidate):
        return candidate
    return new_request_id()


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Записать структурное событие одной JSON-строкой."""
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level).lower(),
        "event": event,
    }
    request_id = get_request_id()
    if request_id:
        payload["request_id"] = request_id
    payload.update(fields)
    _event_logger.log(
        level,
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":")),
    )


def is_timeout_error(error: BaseException) -> bool:
    """Распознать таймаут разных клиентов без жёсткой зависимости.

    ``httpx.TimeoutException``, ``requests.exceptions.Timeout`` и
    ``asyncio.TimeoutError`` не всегда наследуются от встроенного
    ``TimeoutError``. Поэтому проверяем имена классов в MRO.
    """
    if isinstance(error, TimeoutError):
        return True
    return any("timeout" in cls.__name__.lower() for cls in type(error).__mro__)


def classify_empty_answer(
    *,
    abstained: bool,
    citations: int,
    grounding_errors: Any = None,
) -> Optional[str]:
    """Вернуть причину пустого ответа или ``None``.

    Причина нужна и в метрике, и в логе: «пусто» без объяснения не
    позволяет отличить отказ модели от пустого контекста.
    """
    if citations:
        return None
    if not abstained:
        return "no_citations"
    errors = [str(item) for item in (grounding_errors or [])]
    for code in ("no_fragments", "llm_unavailable_or_empty", "no_inline_citations"):
        if code in errors:
            return code
    return "abstained"


class Metrics:
    """Потокобезопасные счётчики и latency последних запросов."""

    def __init__(
        self,
        *,
        max_latency_samples: int = 500,
        slow_threshold_ms: Optional[float] = None,
    ) -> None:
        self._lock = threading.Lock()
        self._latencies: Deque[float] = deque(maxlen=max_latency_samples)
        self.slow_threshold_ms = (
            float(slow_threshold_ms)
            if slow_threshold_ms is not None
            else float(os.getenv("SLOW_REQUEST_MS", "5000"))
        )
        self.reset()

    def reset(self) -> None:
        """Обнулить все счётчики (используется тестами и рестартом)."""
        with self._lock:
            self.requests_total = 0
            self.errors_total = 0
            self.client_errors_total = 0
            self.server_errors_total = 0
            self.slow_requests_total = 0
            self.timeouts_total = 0
            self.empty_answers_total = 0
            self.empty_answers_by_reason: Dict[str, int] = {}
            self._routes: Dict[str, Dict[str, float]] = {}
            self._stages: Dict[str, Dict[str, float]] = {}
            self._latencies.clear()

    def record_request(self, route: str, status_code: int, duration_ms: float) -> bool:
        """Учесть запрос; вернуть ``True``, если он медленный."""
        duration_ms = max(0.0, float(duration_ms))
        slow = duration_ms >= self.slow_threshold_ms
        with self._lock:
            self.requests_total += 1
            self._latencies.append(duration_ms)
            if status_code >= 400:
                self.errors_total += 1
            if status_code >= 500:
                self.server_errors_total += 1
            elif status_code >= 400:
                self.client_errors_total += 1
            if slow:
                self.slow_requests_total += 1
            route_stats = self._routes.setdefault(
                route, {"requests": 0, "errors": 0, "total_ms": 0.0, "max_ms": 0.0}
            )
            route_stats["requests"] += 1
            if status_code >= 400:
                route_stats["errors"] += 1
            route_stats["total_ms"] += duration_ms
            route_stats["max_ms"] = max(route_stats["max_ms"], duration_ms)
        return slow

    def record_stage(self, scope: str, stage: str, duration_ms: float) -> None:
        """Учесть длительность этапа, например ``ask.retrieval``."""
        duration_ms = max(0.0, float(duration_ms))
        key = f"{scope}.{stage}"
        with self._lock:
            stats = self._stages.setdefault(key, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
            stats["count"] += 1
            stats["total_ms"] += duration_ms
            stats["max_ms"] = max(stats["max_ms"], duration_ms)

    def record_timeout(self) -> None:
        """Учесть таймаут внешнего вызова (LLM, HTTP и т.п.)."""
        with self._lock:
            self.timeouts_total += 1

    def record_empty_answer(self, reason: str) -> None:
        """Учесть пустой ответ с причиной."""
        with self._lock:
            self.empty_answers_total += 1
            self.empty_answers_by_reason[reason] = self.empty_answers_by_reason.get(reason, 0) + 1

    @staticmethod
    def _percentile(ordered: list[float], fraction: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 1)

    def snapshot(self) -> Dict[str, Any]:
        """Снимок для ``/metrics`` и ``/health``."""
        with self._lock:
            latencies = sorted(self._latencies)
            count = len(latencies)
            average = round(sum(latencies) / count, 1) if count else 0.0
            latency = {
                "count": count,
                "avg_ms": average,
                "p50_ms": self._percentile(latencies, 0.50),
                "p95_ms": self._percentile(latencies, 0.95),
                "max_ms": round(latencies[-1], 1) if latencies else 0.0,
            }
            routes = {
                route: {
                    "requests": int(stats["requests"]),
                    "errors": int(stats["errors"]),
                    "avg_ms": (
                        round(stats["total_ms"] / stats["requests"], 1)
                        if stats["requests"]
                        else 0.0
                    ),
                    "max_ms": round(stats["max_ms"], 1),
                }
                for route, stats in sorted(self._routes.items())
            }
            stages = {
                key: {
                    "count": int(stats["count"]),
                    "avg_ms": (
                        round(stats["total_ms"] / stats["count"], 1) if stats["count"] else 0.0
                    ),
                    "max_ms": round(stats["max_ms"], 1),
                }
                for key, stats in sorted(self._stages.items())
            }
            return {
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "client_errors_total": self.client_errors_total,
                "server_errors_total": self.server_errors_total,
                "slow_requests_total": self.slow_requests_total,
                "timeouts_total": self.timeouts_total,
                "empty_answers_total": self.empty_answers_total,
                "empty_answers_by_reason": dict(sorted(self.empty_answers_by_reason.items())),
                "slow_threshold_ms": round(self.slow_threshold_ms, 1),
                "latency": latency,
                "routes": routes,
                "stages": stages,
            }


#: Общий процессный экземпляр метрик.
metrics = Metrics()


def elapsed_ms(started: float) -> float:
    """Миллисекунды от ``time.perf_counter()``."""
    return round((time.perf_counter() - started) * 1000, 1)
