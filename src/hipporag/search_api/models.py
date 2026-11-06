from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Грубая доля подтверждённых цитат (0 — отказ или нет ссылок, "
            "0.5 — есть ошибки заземления, 1.0 — чистое заземление). "
            "Не вероятность правильности ответа."
        ),
    )
    citations: List[Citation] = Field(
        ..., description="Только те цитаты, на которые ответ реально сослался"
    )
    retrieved_docs: List[str] = Field(..., description="Релевантные фрагменты документов")
    abstained: bool = Field(
        default=False, description="Модель отказалась отвечать: в контексте нет ответа"
    )
    grounding_errors: List[str] = Field(
        default_factory=list, description="Ошибки заземления: невалидные или неполные ссылки"
    )
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
    index_version: Optional[int] = Field(
        default=None, description="Версия формата индекса из index_manifest.json"
    )
    index_state: str = Field(
        default="unknown", description="Состояние формата индекса (ok/missing/stale/...)"
    )
    index_operation: str = Field(
        default="idle", description="Текущая операция с индексом (idle/running/error)"
    )
    documents: int = Field(default=0, description="Количество документов в индексе")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Текущее время"
    )


# ── H2-2: жизненный цикл индекса ────────────────────────────────────────────


class IndexStatusResponse(BaseModel):
    state: str = Field(..., description="Состояние операции: idle/running/error")
    operation: Optional[str] = Field(None, description="Имя операции")
    started_at: Optional[str] = Field(None, description="Начало операции")
    finished_at: Optional[str] = Field(None, description="Конец операции")
    error: Optional[str] = Field(None, description="Ошибка последней операции")
    detail: Dict[str, Any] = Field(default_factory=dict, description="Детали операции")
    index: Dict[str, Any] = Field(default_factory=dict, description="Сведения об индексе")
    backups: int = Field(default=0, description="Количество резервных копий")


class IndexDocument(BaseModel):
    source_file: str = Field(..., description="Имя файла источника")
    doc_title: str = Field(default="", description="Название документа")
    pages: int = Field(default=0, description="Количество страниц")
    chunks: int = Field(default=0, description="Количество чанков")


class IndexDocumentsResponse(BaseModel):
    documents: List[IndexDocument] = Field(default_factory=list)
    total: int = Field(default=0, description="Всего документов")


class BackupRequest(BaseModel):
    label: Optional[str] = Field(None, description="Метка резервной копии")


class BackupInfo(BaseModel):
    name: str = Field(..., description="Имя zip-файла")
    path: str = Field(..., description="Полный путь")
    created_at: str = Field(..., description="Время создания")
    files: int = Field(default=0, description="Количество файлов")
    bytes: int = Field(default=0, description="Размер в байтах")
    label: str = Field(default="", description="Метка")
    source_dir: str = Field(default="", description="Каталог индекса")


class BackupListResponse(BaseModel):
    backups: List[BackupInfo] = Field(default_factory=list)


class RestoreRequest(BaseModel):
    name: str = Field(..., description="Имя резервной копии (basename .zip)")


class RestoreResponse(BaseModel):
    restored: str = Field(..., description="Имя восстановленной копии")
    working_dir: str = Field(..., description="Каталог индекса")
    format_state: str = Field(..., description="Состояние формата после восстановления")


class DeleteDocumentsRequest(BaseModel):
    source_files: List[str] = Field(..., description="Имена документов (source_file)")


class DeleteDocumentsResponse(BaseModel):
    deleted_documents: List[str] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    deleted_chunks: int = Field(default=0)
    shared_chunks_updated: int = Field(default=0)
    skipped_shared_chunks: int = Field(default=0)


class ReindexRequest(BaseModel):
    pdf_paths: List[str] = Field(..., description="Пути к PDF для переиндексации")


class ReindexResponse(BaseModel):
    status: str = Field(default="accepted", description="Статус: accepted, success или warning")
    message: str = Field(..., description="Пояснение")
    detail: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    status: str = Field(default="error", description="Статус выполнения запроса")
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Детали ошибки")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Время возникновения ошибки"
    )
