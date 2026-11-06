"""FastAPI-слой веб-UI H2-1.

Модуль не создаёт второе приложение: он добавляет UI-маршруты и статику к
существующему ``hipporag.search_api.server.app``. Поэтому аутентификация,
CORS, ``/health`` и ``/search`` остаются едиными для API и браузера.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from hipporag.search_api import observability
from hipporag.search_api import server as api_server

from . import service
from .feedback import FeedbackStore
from .models import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModeInfo,
    UiConfig,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = api_server.app
feedback_store = FeedbackStore(os.getenv("FEEDBACK_PATH"))


@app.get("/ui/config", response_model=UiConfig, tags=["ui"])
async def ui_config() -> UiConfig:
    """Публичная конфигурация UI: режимы и факт обязательности ключа."""
    return UiConfig(
        auth_required=api_server.API_KEY_REQUIRED,
        version=api_server.app.version,
        default_mode=service.DEFAULT_UI_MODE,
        modes=service.mode_payload(),
        feedback_enabled=True,
    )


@app.get("/ui/modes", response_model=List[ModeInfo], tags=["ui"])
async def ui_modes() -> List[ModeInfo]:
    """Список трёх продуктовых режимов."""
    return service.mode_payload()


@app.post("/api/ask", response_model=AskResponse, tags=["ui"])
async def ask(request: AskRequest) -> AskResponse:
    """Ответ по выбранному режиму с цитатами или явным отказом."""
    if api_server.hipporag is None:
        raise HTTPException(status_code=500, detail="Система не инициализирована")

    started = time.perf_counter()
    try:
        answer = service.answer_question(
            api_server.hipporag,
            request.question,
            request.mode,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        if observability.is_timeout_error(error):
            observability.metrics.record_timeout()
        observability.log_event(
            "ask_failed",
            level=logging.ERROR,
            mode=request.mode,
            error_type=type(error).__name__,
            error=str(error)[:300],
        )
        logger.exception("Ошибка веб-поиска: mode=%s", request.mode)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка поиска")

    duration_ms = observability.elapsed_ms(started)
    retrieval_ms = round(answer.retrieval_time * 1000, 1)
    answer_ms = round(answer.answer_time * 1000, 1)
    observability.metrics.record_stage("ask", "retrieval", retrieval_ms)
    observability.metrics.record_stage("ask", "answer", answer_ms)
    observability.metrics.record_stage("ask", "total", duration_ms)
    empty_reason = observability.classify_empty_answer(
        abstained=answer.abstained,
        citations=len(answer.citations),
        grounding_errors=answer.grounding_errors,
    )
    if empty_reason:
        observability.metrics.record_empty_answer(empty_reason)

    stages = {
        "retrieval_ms": retrieval_ms,
        "answer_ms": answer_ms,
        "total_ms": duration_ms,
    }
    observability.log_event(
        "ask_completed",
        mode=answer.mode,
        latency_ms=duration_ms,
        fragments=len(answer.fragments),
        used_citations=len(answer.citations),
        abstained=answer.abstained,
        grounding_errors=answer.grounding_errors,
        empty_reason=empty_reason,
        stages=stages,
    )
    if empty_reason:
        observability.log_event(
            "answer_empty",
            path="/api/ask",
            reason=empty_reason,
            mode=answer.mode,
            fragments=len(answer.fragments),
            used_citations=len(answer.citations),
            stages=stages,
        )
    return answer


@app.post("/api/feedback", response_model=FeedbackResponse, tags=["ui"])
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Сохранить оценку ответа в append-only JSONL."""
    try:
        record = feedback_store.append(
            query=request.query,
            answer=request.answer,
            mode=request.mode,
            rating=request.rating,
            comment=request.comment,
            citations=request.citations,
        )
    except OSError:
        logger.exception("Не удалось сохранить обратную связь")
        raise HTTPException(status_code=500, detail="Не удалось сохранить обратную связь")

    return FeedbackResponse(
        id=record["id"],
        timestamp=datetime.fromisoformat(record["created_at"]),
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Главная страница UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/ui/dashboard", include_in_schema=False)
async def dashboard() -> FileResponse:
    """H2-4: простая страница наблюдаемости."""
    return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")


# Mount добавляется последним: маршруты /ui/config и /ui/modes должны
# матчиться раньше статики. Статика публична, данные — под guard.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="astro-ui")
