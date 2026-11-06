import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from astro_app.index_lifecycle import IndexLifecycle

from ..HippoRAG import HippoRAG
from ..utils.config_utils import BaseConfig
from .models import (
    BackupInfo,
    BackupListResponse,
    BackupRequest,
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    HealthResponse,
    IndexDocumentsResponse,
    IndexStatusResponse,
    IngestRequest,
    IngestResponse,
    IngestStats,
    MetricsResponse,
    MetricsSummary,
    ReindexRequest,
    ReindexResponse,
    RestoreRequest,
    RestoreResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from .observability import (
    REQUEST_ID_HEADER,
    classify_empty_answer,
    elapsed_ms,
    get_request_id,
    is_timeout_error,
    log_event,
    metrics,
    request_id_from_header,
    reset_request_id,
    set_request_id,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
hipporag: Optional[HippoRAG] = None
index_lifecycle: Optional[IndexLifecycle] = None
_app_config: Optional[BaseConfig] = None
start_time: float = time.time()

# D-11: последние задержки /search для p95 в логах. Храним только числа,
# без вопросов и ответов, чтобы не раздувать память и не логировать PII.
_GROUNDING_LATENCIES_MS: deque = deque(maxlen=200)


def grounding_p95_latency_ms() -> float:
    """p95 по последним запросам; 0.0, пока нет данных."""
    if not _GROUNDING_LATENCIES_MS:
        return 0.0
    ordered = sorted(_GROUNDING_LATENCIES_MS)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return round(ordered[index], 1)


MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
API_KEY = os.getenv("API_KEY")
# D3/D7: по умолчанию ключ обязателен — сервер не поднимается без API_KEY.
# Явный opt-out (API_KEY_REQUIRED=0) оставлен только для локальной разработки.
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# N2/N5: единый источник правды для CORS. Guard не дублирует список origin,
# а CORS-слой обрамляет guard и добавляет заголовки ко всем ответам, включая 401.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "API_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000"
    ).split(",")
    if origin.strip()
]

# D5: счётчик отказов авторизации с момента запуска (виден в /health).
auth_failures = 0


def _close_hipporag() -> None:
    """Закрыть ресурсы текущего индекса перед подменой каталога (H2-2)."""
    if hipporag is None:
        return
    for attribute in (
        "chunk_embedding_store",
        "entity_embedding_store",
        "fact_embedding_store",
    ):
        store = getattr(hipporag, attribute, None)
        close = getattr(store, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:  # noqa: BLE001 — закрытие не должно мешать restore
                logger.warning("Не удалось закрыть %s: %s", attribute, error)


def _reload_hipporag() -> HippoRAG:
    """Пересоздать HippoRAG после восстановления индекса (H2-2)."""
    global hipporag
    if _app_config is None:
        raise RuntimeError("Конфигурация сервера не сохранена — reload невозможен")
    hipporag = HippoRAG(global_config=_app_config)
    return hipporag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при запуске сервера: конфиг из env, проверка непустоты индекса."""
    global hipporag, index_lifecycle, _app_config
    if API_KEY_REQUIRED and not API_KEY:
        raise RuntimeError(
            "API_KEY обязателен (API_KEY_REQUIRED=1 по умолчанию), но не задан. "
            "Задайте API_KEY в окружении или в .env. Для локальной разработки на "
            "loopback отключение задаётся явно: API_KEY_REQUIRED=0."
        )
    try:
        logger.info("Инициализация HippoRAG...")
        config = BaseConfig()
        config.save_dir = os.getenv("SAVE_DIR", "outputs/astrology_test")
        config.llm_name = os.getenv("LLM_NAME", "gemma-4-e4b")
        config.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        config.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        )
        _app_config = config

        hipporag = HippoRAG(global_config=config)
        index_lifecycle = IndexLifecycle(
            index_dir=hipporag.working_dir,
            get_hipporag=lambda: hipporag,
            close_hipporag=_close_hipporag,
            reload_hipporag=_reload_hipporag,
        )
        chunk_count = len(hipporag.chunk_embedding_store.get_all_ids())
        fact_count = len(hipporag.fact_embedding_store.get_all_ids())
        if chunk_count == 0:
            logger.warning(
                "Индекс пуст: проверьте SAVE_DIR и EMBEDDING_MODEL_NAME — "
                "API не видит данные и /search вернёт пустые результаты"
            )
        else:
            logger.info(f"Инициализация завершена: {chunk_count} чанков, {fact_count} фактов")
        if fact_count <= 1:
            logger.warning(
                f"Факт-стор почти пуст ({fact_count} записей) — графовая часть поиска "
                "вырождена, поиск идёт преимущественно по dense-retrieval. Требуется переиндексация."
            )
    except Exception as e:
        logger.error(f"Ошибка инициализации HippoRAG: {str(e)}")
        raise
    yield


app = FastAPI(
    title="Astrology RAG Search API",
    description="API для поиска информации в астрологических документах с цитированием источников",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


def is_public_ui_path(path: str) -> bool:
    """Публичная статика UI: HTML/CSS/JS без данных корпуса.

    Страница должна открываться в браузере до ввода API-ключа, иначе
    пользователь не увидит даже поле для ключа. Все маршруты с данными
    (``/api/ask``, ``/api/feedback``, ``/search``) остаются под guard.
    """
    return path == "/" or path == "/ui" or path.startswith("/ui/")


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """Аутентификация по заголовку X-API-Key.

    Требование ключа управляется API_KEY_REQUIRED (по умолчанию включено, D3/D7):
    без API_KEY сервер не стартует, а при API_KEY_REQUIRED=0 остаётся fail-open
    для локальной разработки.

    CORS-preflight (OPTIONS) пропускается без проверки ключа: браузер не отправляет
    ключ в preflight-запросе (D2). Отказы считаются и логируются (D5).
    """
    global auth_failures
    if request.method != "OPTIONS" and not is_public_ui_path(request.url.path):
        if not API_KEY:
            if API_KEY_REQUIRED:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "error": (
                            "Сервер запущен без API_KEY при API_KEY_REQUIRED=1 — "
                            "запросы не обслуживаются"
                        ),
                    },
                )
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "").encode("utf-8")
        if not secrets.compare_digest(provided, API_KEY.encode("utf-8")):
            auth_failures += 1
            logger.warning(
                "Отказ авторизации: неверный X-API-Key (path=%s, origin=%s, всего отказов: %d)",
                request.url.path,
                request.headers.get("origin") or "-",
                auth_failures,
            )
            return JSONResponse(
                status_code=401,
                content={"status": "error", "error": "Требуется заголовок X-API-Key"},
            )
    return await call_next(request)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """H2-4: request ID, длительность и структурное событие запроса.

    Регистрируется после guard, но до CORS: CORS остаётся внешним слоем
    (N2/N5), а observability видит ответы guard, включая 401, и ответы
    маршрутов. 5xx от необработанных исключений учитываются в ``except``.
    """
    request_id = request_id_from_header(request.headers.get(REQUEST_ID_HEADER))
    token = set_request_id(request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    except Exception as error:
        duration_ms = elapsed_ms(started)
        log_event(
            "request_failed",
            level=logging.ERROR,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error_type=type(error).__name__,
            error=str(error)[:300],
        )
        if is_timeout_error(error):
            metrics.record_timeout()
        raise
    finally:
        duration_ms = elapsed_ms(started)
        route = request.scope.get("route")
        route_path = getattr(route, "path", None) or request.url.path
        slow = metrics.record_request(route_path, status_code, duration_ms)
        level = logging.DEBUG if request.url.path in ("/health", "/metrics") else logging.INFO
        if slow:
            level = logging.WARNING
        log_event(
            "request_completed",
            level=level,
            method=request.method,
            path=request.url.path,
            route=route_path,
            status_code=status_code,
            duration_ms=duration_ms,
            slow=slow,
        )
        reset_request_id(token)


# CORS middleware для веб-приложений.
# Регистрируется ПОСЛЕ guard и observability: в Starlette последний добавленный
# middleware становится внешним, поэтому CORS обрамляет их и добавляет свои
# заголовки ко всем ответам, включая 401 — без дублирования списка origin
# внутри guard (N2/N5).
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    # X-API-Key нужен браузерному клиенту для передачи ключа (D1 из вердикта Mavis)
    allow_headers=["Content-Type", "Authorization", "X-API-Key", REQUEST_ID_HEADER],
    expose_headers=[REQUEST_ID_HEADER],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса."""
    uptime_seconds = time.time() - start_time
    uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"

    chunk_count = len(hipporag.chunk_embedding_store.get_all_ids()) if hipporag else 0
    fact_count = len(hipporag.fact_embedding_store.get_all_ids()) if hipporag else 0
    index_info = index_lifecycle.status() if index_lifecycle is not None else {}
    index_meta = index_info.get("index") or {}

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime_str,
        chunks=chunk_count,
        facts=fact_count,
        auth_required=API_KEY_REQUIRED,
        auth_failures=auth_failures,
        index_version=index_meta.get("format_version"),
        index_state=index_meta.get("format_state", "unknown"),
        index_operation=index_info.get("state", "idle"),
        documents=index_meta.get("documents", 0),
        llm_model=index_meta.get("llm_name") or getattr(_app_config, "llm_name", None),
        embedding_model=index_meta.get("embedding_model")
        or getattr(_app_config, "embedding_model_name", None),
        index_created_at=index_meta.get("created_at"),
        metrics=MetricsSummary(**metrics.snapshot()),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["observability"])
async def get_metrics() -> MetricsResponse:
    """H2-4: счётчики запросов, ошибок, пустых ответов и per-stage timings."""
    return MetricsResponse(
        uptime_seconds=round(time.time() - start_time, 1),
        **metrics.snapshot(),
    )


def _require_index_lifecycle() -> IndexLifecycle:
    """Вернуть менеджер индекса или 503, если сервер ещё не инициализирован."""
    if index_lifecycle is None:
        raise HTTPException(status_code=503, detail="Индекс не инициализирован")
    return index_lifecycle


@app.get("/index/status", response_model=IndexStatusResponse, tags=["index"])
async def index_status() -> IndexStatusResponse:
    """Статус индекса: версия, счётчики, документы и текущая операция (H2-2)."""
    return IndexStatusResponse(**_require_index_lifecycle().status())


@app.get("/index/documents", response_model=IndexDocumentsResponse, tags=["index"])
async def index_documents() -> IndexDocumentsResponse:
    """Список документов, попавших в индекс (H2-2)."""
    lifecycle = _require_index_lifecycle()
    try:
        documents = lifecycle.documents()
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return IndexDocumentsResponse(documents=documents, total=len(documents))


@app.post("/index/backup", response_model=BackupInfo, tags=["index"])
async def index_backup(request: Optional[BackupRequest] = None) -> BackupInfo:
    """Создать резервную копию индекса (H2-2)."""
    lifecycle = _require_index_lifecycle()
    try:
        record = lifecycle.backup(request.label if request else None)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return BackupInfo(**record)


@app.get("/index/backups", response_model=BackupListResponse, tags=["index"])
async def index_backups() -> BackupListResponse:
    """Список резервных копий, новые сверху (H2-2)."""
    return BackupListResponse(backups=_require_index_lifecycle().list_backups())


@app.post("/index/restore", response_model=RestoreResponse, tags=["index"])
async def index_restore(request: RestoreRequest) -> RestoreResponse:
    """Восстановить индекс из резервной копии (H2-2)."""
    lifecycle = _require_index_lifecycle()
    try:
        result = lifecycle.restore(request.name)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return RestoreResponse(**result)


@app.post("/index/delete", response_model=DeleteDocumentsResponse, tags=["index"])
async def index_delete(request: DeleteDocumentsRequest) -> DeleteDocumentsResponse:
    """Удалить документы из индекса по ``source_file`` (H2-2)."""
    lifecycle = _require_index_lifecycle()
    try:
        result = lifecycle.delete_documents(request.source_files)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001 — наружу отдаём понятный 500
        logger.exception("Ошибка удаления документов")
        raise HTTPException(status_code=500, detail="Не удалось удалить документы") from error
    return DeleteDocumentsResponse(**result)


@app.post("/index/reindex", response_model=ReindexResponse, tags=["index"])
async def index_reindex(
    request: ReindexRequest,
    sync: bool = False,
    background_tasks: BackgroundTasks = None,
) -> ReindexResponse:
    """Переиндексировать PDF, предварительно сделав бэкап (H2-2).

    По умолчанию операция выполняется в фоне, а её результат и
    предупреждения доступны через ``GET /index/status``. С ``?sync=true``
    запрос ждёт окончания и сразу возвращает ``status=warning``, если из
    какого-то PDF не извлечено ни одного чанка.
    """
    lifecycle = _require_index_lifecycle()
    if lifecycle.status().get("state") == "running":
        raise HTTPException(status_code=409, detail="Операция с индексом уже выполняется")

    missing = [path for path in request.pdf_paths if not os.path.exists(path)]
    if missing:
        raise HTTPException(status_code=400, detail="Файлы не найдены: " + ", ".join(missing))

    if not sync and background_tasks is not None:
        background_tasks.add_task(lifecycle.reindex, request.pdf_paths)
        return ReindexResponse(
            status="accepted",
            message=(
                "Переиндексация запущена в фоновом режиме; результат и "
                "предупреждения — в GET /index/status"
            ),
            detail={"pdf_paths": request.pdf_paths},
        )

    try:
        detail = lifecycle.reindex(request.pdf_paths)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    warnings = list(detail.get("warnings") or [])
    status = detail.get("status")
    if status not in ("success", "warning"):
        status = "warning" if warnings else "success"
    message = "Переиндексация завершена"
    if warnings:
        message += f" с предупреждениями ({len(warnings)})"
    return ReindexResponse(status=status, message=message, detail=detail)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Поиск ответов на астрологические вопросы."""
    if hipporag is None:
        raise HTTPException(status_code=500, detail="Система не инициализирована")

    started = time.perf_counter()
    try:
        rag_started = time.perf_counter()
        # Выполнение поиска
        results = hipporag.rag_qa(queries=[request.question])
        result = results[0][
            0
        ]  # results[0] = tuple, [0] = queries_solutions list, [0] = first QuerySolution
        rag_qa_ms = elapsed_ms(rag_started)

        build_started = time.perf_counter()
        # Подготовка цитат: только те, на которые ответ реально сослался.
        # Полный список найденных фрагментов остаётся в retrieved_docs.
        citations = []
        for citation in result.used_citations:
            citations.append(
                {
                    "source_file": citation.source_file,
                    "doc_title": citation.doc_title,
                    "page_num": citation.page_num,
                    "section": citation.section,
                    "text_snippet": citation.text_snippet,
                    "sources": citation.sources or [],
                }
            )

        # Грубая оценка заземления, а не вероятность правильности ответа.
        if result.abstained or not citations:
            confidence = 0.0
        elif result.grounding_errors:
            confidence = 0.5
        else:
            confidence = 1.0

        build_ms = elapsed_ms(build_started)
        processing_time = time.perf_counter() - started
        search_result = SearchResult(
            answer=result.answer,
            confidence=confidence,
            citations=citations,
            retrieved_docs=(
                result.docs[: request.top_k] if not request.include_full_text else result.docs
            ),
            abstained=result.abstained,
            grounding_errors=result.grounding_errors,
            processing_time=processing_time,
        )

        metrics.record_stage("search", "rag_qa", rag_qa_ms)
        metrics.record_stage("search", "build_response", build_ms)
        metrics.record_stage("search", "total", processing_time * 1000)
        empty_reason = classify_empty_answer(
            abstained=result.abstained,
            citations=len(citations),
            grounding_errors=result.grounding_errors,
        )
        if empty_reason:
            metrics.record_empty_answer(empty_reason)

        _GROUNDING_LATENCIES_MS.append(processing_time * 1000)
        logger.info(
            "search: abstained=%s grounding_errors=%s used_citations=%d "
            "retrieved_docs=%d latency_ms=%.1f p95_ms=%.1f",
            result.abstained,
            result.grounding_errors,
            len(result.used_citations),
            len(result.docs),
            processing_time * 1000,
            grounding_p95_latency_ms(),
        )
        stages = {
            "rag_qa_ms": rag_qa_ms,
            "build_response_ms": build_ms,
            "total_ms": round(processing_time * 1000, 1),
        }
        log_event(
            "search_completed",
            abstained=result.abstained,
            grounding_errors=result.grounding_errors,
            used_citations=len(citations),
            retrieved_docs=len(result.docs),
            latency_ms=round(processing_time * 1000, 1),
            p95_ms=grounding_p95_latency_ms(),
            empty_reason=empty_reason,
            stages=stages,
        )
        if empty_reason:
            log_event(
                "answer_empty",
                path="/search",
                reason=empty_reason,
                abstained=result.abstained,
                retrieved_docs=len(result.docs),
                used_citations=len(citations),
                stages=stages,
            )

        return SearchResponse(data=search_result)

    except Exception as e:
        if is_timeout_error(e):
            metrics.record_timeout()
        logger.error(f"Ошибка поиска: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка поиска")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Индексация PDF-документов."""
    if hipporag is None:
        raise HTTPException(status_code=500, detail="Система не инициализирована")

    try:
        # Настройка OCR при необходимости
        ocr_config = None
        if request.enable_ocr:
            ocr_config = {
                "enabled": True,
                "language": request.ocr_language,
                "dpi": 300,
                "engine_mode": 1,
            }

        # Проверка существования файлов
        failed_files = []
        valid_files = []

        for pdf_path in request.pdf_paths:
            if not os.path.exists(pdf_path):
                failed_files.append(pdf_path)
                logger.warning(f"Файл не найден: {pdf_path}")
            else:
                valid_files.append(pdf_path)

        if not valid_files and request.pdf_paths:
            raise HTTPException(status_code=400, detail="Не найдено ни одного существующего файла")

        # Индексация документов
        if valid_files:
            chunks_before = len(hipporag.chunk_embedding_store.get_all_ids())
            hipporag.index_pdfs(valid_files, ocr_config=ocr_config)
            chunks_extracted = len(hipporag.chunk_embedding_store.get_all_ids()) - chunks_before
            writer = getattr(hipporag, "_write_index_manifest", None)
            if callable(writer):
                writer()
        else:
            chunks_extracted = 0

        stats = IngestStats(
            total_files=len(request.pdf_paths),
            processed_files=len(valid_files),
            failed_files=failed_files,
            chunks_extracted=chunks_extracted,
        )

        return IngestResponse(stats=stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка индексации: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка индексации")


@app.post("/upload-and-ingest")
async def upload_and_ingest(
    files: List[UploadFile] = File(...),
    sync: bool = False,
    background_tasks: BackgroundTasks = None,
):
    """Загрузка и индексация PDF-файлов.

    По умолчанию индексация идёт в фоне; её результат и предупреждения
    доступны через ``GET /index/status``. С ``?sync=true`` запрос ждёт
    окончания и возвращает ``status=warning``, если из файла не извлечено
    ни одного чанка.
    """
    if hipporag is None:
        raise HTTPException(status_code=500, detail="Система не инициализирована")

    upload_dir = Path("temp_uploads")
    upload_dir.mkdir(exist_ok=True)
    saved_files = []
    failed_files = []

    for file in files:
        try:
            from pathlib import PureWindowsPath

            # Имя с "\" или "/" не должно позволять выйти за пределы каталога загрузки
            safe_name = PureWindowsPath(file.filename or "").name
            if not safe_name.lower().endswith(".pdf"):
                continue
            file_path = upload_dir / safe_name
            if not file_path.resolve().is_relative_to(upload_dir.resolve()):
                failed_files.append(file.filename)
                logger.warning(f"Отклонено небезопасное имя файла: {file.filename!r}")
                continue
            content = await file.read()
            if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
                failed_files.append(file.filename)
                logger.warning(
                    f"Файл {file.filename!r} превышает лимит {MAX_UPLOAD_MB} МБ — отклонён"
                )
                continue
            with open(file_path, "wb") as buffer:
                buffer.write(content)
                saved_files.append(str(file_path))
        except Exception as e:
            failed_files.append(file.filename)
            logger.error(f"Ошибка сохранения файла {file.filename}: {str(e)}")
    if not saved_files and files:
        raise HTTPException(status_code=400, detail="Не удалось сохранить ни один файл")

    # Индексация через lifecycle — с бэкапом, статусом и диагностикой (H2-2).
    def index_task() -> dict:
        try:
            if index_lifecycle is not None:
                return index_lifecycle.reindex(saved_files)
            hipporag.index_pdfs(saved_files)
            return {}
        except Exception as error:
            logger.error("Ошибка индексации загруженных файлов: %s", error)
            raise
        finally:
            # Очистка временных файлов после индексации, в том числе при ошибке.
            for file_path in saved_files:
                try:
                    os.remove(file_path)
                except Exception as error:
                    logger.warning("Не удалось удалить временный файл %s: %s", file_path, error)
            try:
                upload_dir.rmdir()
            except OSError:
                logger.warning("Каталог temp_uploads не пуст — оставлен на месте")

    if not sync and background_tasks:
        background_tasks.add_task(index_task)
        return {
            "status": "accepted",
            "message": (
                "Файлы загружены, индексация запущена в фоновом режиме; "
                "результат и предупреждения — в GET /index/status"
            ),
            "processed_files": len(saved_files),
            "failed_files": len(failed_files),
            "warnings": [],
            "zero_chunk_files": [],
        }

    try:
        result = index_task()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    warnings = list(result.get("warnings") or [])
    zero_chunk_files = list(result.get("zero_chunk_files") or [])
    status = result.get("status")
    if status not in ("success", "warning"):
        status = "warning" if warnings else "success"
    message = "Файлы загружены и проиндексированы"
    if warnings:
        message += f" с предупреждениями ({len(warnings)})"
    return {
        "status": status,
        "message": message,
        "processed_files": len(saved_files),
        "failed_files": len(failed_files),
        "warnings": warnings,
        "zero_chunk_files": zero_chunk_files,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений."""
    logger.error(f"Необработанное исключение: {str(exc)}", exc_info=True)
    log_event(
        "unhandled_error",
        level=logging.ERROR,
        path=request.url.path,
        method=request.method,
        error_type=type(exc).__name__,
        error=str(exc)[:300],
    )
    headers = {}
    request_id = get_request_id()
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Внутренняя ошибка сервера",
            "detail": None,
        },
        headers=headers or None,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail, "detail": getattr(exc, "detail", None)},
    )
