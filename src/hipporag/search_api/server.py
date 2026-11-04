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

from ..HippoRAG import HippoRAG
from ..utils.config_utils import BaseConfig
from .models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    IngestStats,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
hipporag: Optional[HippoRAG] = None
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при запуске сервера: конфиг из env, проверка непустоты индекса."""
    global hipporag
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

        hipporag = HippoRAG(global_config=config)
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
    if request.method != "OPTIONS":
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


# CORS middleware для веб-приложений.
# Регистрируется ПОСЛЕ guard: в Starlette последний добавленный middleware становится
# внешним, поэтому CORS обрамляет guard и добавляет свои заголовки ко всем ответам,
# включая 401 — без дублирования списка origin внутри guard (N2/N5).
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    # X-API-Key нужен браузерному клиенту для передачи ключа (D1 из вердикта Mavis)
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса."""
    uptime_seconds = time.time() - start_time
    uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"

    chunk_count = len(hipporag.chunk_embedding_store.get_all_ids()) if hipporag else 0
    fact_count = len(hipporag.fact_embedding_store.get_all_ids()) if hipporag else 0

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime_str,
        chunks=chunk_count,
        facts=fact_count,
        auth_required=API_KEY_REQUIRED,
        auth_failures=auth_failures,
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Поиск ответов на астрологические вопросы."""
    if hipporag is None:
        raise HTTPException(status_code=500, detail="Система не инициализирована")

    try:
        start_time = time.time()

        # Выполнение поиска
        results = hipporag.rag_qa(queries=[request.question])
        result = results[0][
            0
        ]  # results[0] = tuple, [0] = queries_solutions list, [0] = first QuerySolution

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

        # Вычисление времени обработки
        processing_time = time.time() - start_time

        # Грубая оценка заземления, а не вероятность правильности ответа.
        if result.abstained or not citations:
            confidence = 0.0
        elif result.grounding_errors:
            confidence = 0.5
        else:
            confidence = 1.0

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

        return SearchResponse(data=search_result)

    except Exception as e:
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
    files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None
):
    """Загрузка и индексация PDF-файлов."""
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

    # Индексация в фоне
    def index_task():
        try:
            hipporag.index_pdfs(saved_files)
            # Очистка временных файлов после индексации
            for file_path in saved_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл {file_path}: {e}")
            try:
                upload_dir.rmdir()
            except OSError:
                logger.warning("Каталог temp_uploads не пуст — оставлен на месте")
        except Exception as e:
            logger.error(f"Ошибка фоновой индексации: {str(e)}")

    if background_tasks:
        background_tasks.add_task(index_task)
        message = "Файлы загружены и индексация запущена в фоновом режиме"
    else:
        index_task()
        message = "Файлы загружены и проиндексированы"

    return {
        "status": "success",
        "message": message,
        "processed_files": len(saved_files),
        "failed_files": len(failed_files),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Глобальный обработчик исключений."""
    logger.error(f"Необработанное исключение: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Внутренняя ошибка сервера",
            "detail": None,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Обработчик HTTP исключений."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.detail, "detail": getattr(exc, "detail", None)},
    )
