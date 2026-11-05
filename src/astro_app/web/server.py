"""FastAPI-слой веб-UI H2-1.

Модуль не создаёт второе приложение: он добавляет UI-маршруты и статику к
существующему ``hipporag.search_api.server.app``. Поэтому аутентификация,
CORS, ``/health`` и ``/search`` остаются едиными для API и браузера.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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

    try:
        return service.answer_question(
            api_server.hipporag,
            request.question,
            request.mode,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception:
        logger.exception("Ошибка веб-поиска: mode=%s", request.mode)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка поиска")


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


# Mount добавляется последним: маршруты /ui/config и /ui/modes должны
# матчиться раньше статики. Статика публична, данные — под guard.
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="astro-ui")
