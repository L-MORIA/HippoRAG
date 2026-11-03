from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    question: str = Field(..., description="Вопрос по астрологии для поиска")
    top_k: int = Field(default=5, ge=1, le=20, description="Количество результатов для возврата")
    include_full_text: bool = Field(
        default=False, description="Включать ли полный текст документов"
    )


class Citation(BaseModel):
    source_file: str = Field(..., description="Имя файла источника")
    doc_title: str = Field(..., description="Название документа")
    page_num: int = Field(..., description="Номер страницы")
    section: Optional[str] = Field(None, description="Секция документа")
    text_snippet: str = Field(..., description="Фрагмент текста из источника")
    sources: Optional[List[str]] = Field(
        None, description="Все документы, содержащие этот фрагмент (S-12/D-1)"
    )


class SearchResult(BaseModel):
    answer: str = Field(..., description="Ответ на вопрос")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели в ответе")
    citations: List[Citation] = Field(..., description="Цитаты из источников")
    retrieved_docs: List[str] = Field(..., description="Релевантные фрагменты документов")
    processing_time: float = Field(..., description="Время обработки запроса в секундах")


class SearchResponse(BaseModel):
    status: str = Field(default="success", description="Статус выполнения запроса")
    data: SearchResult = Field(..., description="Результаты поиска")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Время выполнения запроса"
    )


class IngestRequest(BaseModel):
    pdf_paths: List[str] = Field(..., description="Список путей к PDF файлам для индексации")
    enable_ocr: bool = Field(default=False, description="Включить OCR для сканированных документов")
    ocr_language: str = Field(default="eng+rus", description="Язык OCR")


class IngestStats(BaseModel):
    total_files: int = Field(..., description="Общее количество файлов")
    processed_files: int = Field(..., description="Количество успешно обработанных файлов")
    failed_files: List[str] = Field(..., description="Список файлов с ошибками")
    chunks_extracted: int = Field(..., description="Количество извлеченных фрагментов")


class IngestResponse(BaseModel):
    status: str = Field(default="success", description="Статус выполнения запроса")
    stats: IngestStats = Field(..., description="Статистика индексации")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Время выполнения запроса"
    )


class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Статус сервиса")
    version: str = Field(default="1.0.0", description="Версия API")
    uptime: str = Field(..., description="Время работы сервиса")
    chunks: int = Field(default=0, description="Количество чанков в индексе")
    facts: int = Field(default=0, description="Количество фактов в индексе")
    auth_required: bool = Field(
        default=True, description="Требуется ли заголовок X-API-Key для запросов"
    )
    auth_failures: int = Field(
        default=0, description="Счётчик отказов авторизации с момента запуска"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Текущее время"
    )


class ErrorResponse(BaseModel):
    status: str = Field(default="error", description="Статус выполнения запроса")
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Детали ошибки")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Время возникновения ошибки"
    )
