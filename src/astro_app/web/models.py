"""Модели веб-UI H2-1: запросы, ответы, режимы и обратная связь."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ModeInfo(BaseModel):
    """Один режим, доступный в веб-интерфейсе."""

    id: str = Field(..., description="Номер режима из реестра astro_app.modes")
    label: str = Field(..., description="Короткое название для пользователя")
    description: str = Field(..., description="Когда режим уместен")
    expert: bool = Field(..., description="Экспертный режим (скрыт под «Экспертные»)")
    latency: str = Field(..., description="Ориентировочная latency по замерам")
    needs_llm: bool = Field(..., description="Нужен ли LLM уже на этапе поиска")


class UiConfig(BaseModel):
    """Публичная конфигурация UI: без секретов и без данных индекса."""

    auth_required: bool = Field(..., description="Требуется ли X-API-Key")
    version: str = Field(..., description="Версия API")
    default_mode: str = Field(..., description="Режим по умолчанию")
    modes: List[ModeInfo] = Field(..., description="Три продуктовых режима")
    feedback_enabled: bool = Field(default=True, description="Принимается ли обратная связь")


class AskRequest(BaseModel):
    """Вопрос пользователя к выбранному режиму."""

    question: str = Field(..., min_length=1, max_length=4000, description="Вопрос по корпусу")
    mode: str = Field(default="5", max_length=8, description="Номер режима")
    top_k: int = Field(default=5, ge=1, le=10, description="Сколько фрагментов идёт в ответ")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Вопрос не должен быть пустым")
        return value


class CitationOut(BaseModel):
    """Использованная цитата: только то, на что ответ реально сослался."""

    id: int = Field(..., ge=1, description="Номер фрагмента в промпте, начиная с 1")
    source_file: str = Field(..., description="Имя файла источника")
    doc_title: str = Field(..., description="Название документа")
    page_num: int = Field(..., description="Номер страницы")
    section: Optional[str] = Field(None, description="Секция документа")
    text_snippet: str = Field(..., description="Фрагмент текста источника")
    sources: List[str] = Field(default_factory=list, description="Все документы с фрагментом")


class FragmentOut(BaseModel):
    """Найденный фрагмент для диагностики и показа «что нашлось»."""

    id: int = Field(..., ge=1, description="Номер фрагмента в промпте, начиная с 1")
    text: str = Field(..., description="Текст фрагмента")
    doc_title: str = Field(..., description="Название документа")
    page_num: int = Field(..., description="Номер страницы")
    section: Optional[str] = Field(None, description="Секция документа")
    source_file: str = Field(..., description="Имя файла источника")
    score: float = Field(default=0.0, description="Скор режима (не сравним между режимами)")


class AskResponse(BaseModel):
    """Ответ веб-UI: текст, использованные цитаты и диагностика."""

    status: str = Field(default="success", description="Статус выполнения")
    query: str = Field(..., description="Вопрос пользователя")
    mode: str = Field(..., description="Номер режима")
    mode_label: str = Field(..., description="Название режима")
    expert: bool = Field(..., description="Выбран экспертный режим")
    answer: str = Field(..., description="Ответ с inline-ссылками [n] или общий отказ")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Оценка заземления: 0 — отказ или нет ссылок, 0.5 — есть ошибки, "
            "1.0 — чистое заземление. Не вероятность правильности ответа."
        ),
    )
    citations: List[CitationOut] = Field(
        default_factory=list, description="Только использованные цитаты"
    )
    fragments: List[FragmentOut] = Field(
        default_factory=list, description="Найденные фрагменты (диагностика)"
    )
    retrieved_docs: List[str] = Field(
        default_factory=list, description="Тексты найденных фрагментов"
    )
    abstained: bool = Field(default=False, description="Модель отказалась отвечать")
    grounding_errors: List[str] = Field(
        default_factory=list, description="Ошибки заземления или диагностика"
    )
    reformulated_query: Optional[str] = Field(
        None, description="Переформулировка запроса, если режим её делает"
    )
    processing_time: float = Field(..., description="Полное время ответа, секунды")
    retrieval_time: float = Field(..., description="Время поиска, секунды")
    answer_time: float = Field(..., description="Время LLM и проверки ссылок, секунды")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Время ответа",
    )


class FeedbackRequest(BaseModel):
    """Оценка ответа пользователем."""

    rating: Literal["up", "down"] = Field(..., description="Полезно / бесполезно")
    query: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(default="", max_length=20000)
    mode: str = Field(default="5", max_length=8)
    comment: str = Field(default="", max_length=2000)
    citations: List[CitationOut] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    """Подтверждение записи обратной связи."""

    status: str = Field(default="ok")
    id: str = Field(..., description="Идентификатор записи")
    timestamp: datetime = Field(..., description="Время записи")
