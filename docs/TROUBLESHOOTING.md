# Устранение неполадок HippoRAG

> Сначала три быстрые проверки:
>
> 1. LM Studio отвечает: `curl http://localhost:1234/v1/models`.
> 2. Индекс виден: `python scripts/show_index_stats.py --save_dir outputs/astrology_test`.
> 3. Сервер жив: `curl -H "X-API-Key: ..." http://127.0.0.1:8000/health`.
>
> Если ошибка не описана здесь — смотрите [HOW_TO_RUN.md](./HOW_TO_RUN.md),
> [ADMIN_GUIDE.md](./ADMIN_GUIDE.md) и логи в консоли сервера.

## 1. Установка и запуск

| Симптом | Причина | Решение |
|---|---|---|
| `ModuleNotFoundError: No module named 'astro_app'` | Не выполнена editable-установка | `pip install -e ".[dev]"` |
| `ModuleNotFoundError: rank_bm25` / `igraph` | Не все зависимости | Повторите `pip install -e ".[dev]"` |
| Ошибки на Python 3.13/3.14 | Версия не поддерживается | Используйте Python 3.10–3.12 |
| `OutlinesUnavailableError: Guided-декодирование … требует пакет outlines` | Не установлен `outlines` | `pip install -e ".[offline]"` или `pip install "outlines>=1.0,<2"` |
| `PackageDataError: Package-data несовместимы с кодом` | Данные и код разошлись | `python scripts/update_package_data_manifest.py` или переустановите пакет |

### 1.1. One-command start и Docker (H2-3)

| Симптом | Причина | Решение |
|---|---|---|
| `Python не найден` или `Нужен Python 3.10-3.12` | Нет поддерживаемой версии в PATH/`py` | Установите Python 3.10–3.12; `start.bat` ищет `py -3.12/3.11/3.10` |
| `start.sh: Permission denied` | Потерян бит исполнения | `chmod +x start.sh` |
| `LM Studio недоступна по ...` при старте | LM Studio не запущена или неверный `LLM_BASE_URL` | Developer → Start Server; проверьте `curl http://localhost:1234/v1/models`; временно `--skip-lm-check` или `STARTUP_CHECK_LM=0` |
| `Модель '...' не найдена` | `LLM_API_NAME` не совпадает с id в LM Studio | Укажите id из `/v1/models`, например `google/gemma-4-e4b` |
| `Для адреса вне loopback нужен API_KEY` | `scripts/start.py` не разрешает fail-open на `0.0.0.0` | Задайте `API_KEY` или запускайте на `127.0.0.1` |
| `uvicorn не установлен` | Пакет не установлен в текущее окружение | `python -m pip install -e .` или запустите `start.bat`/`./start.sh` |
| `.env` не создаётся | Каталог только для чтения (контейнер под non-root) | Передайте переменные окружения снаружи; это не ошибка запуска |
| `docker: command not found` | Docker не установлен | Установите Docker Engine/Desktop |
| Docker не видит LM Studio | `localhost` внутри контейнера — это контейнер | Compose уже задаёт `host.docker.internal`; для другого хоста — `LLM_BASE_URL_DOCKER` |
| `permission denied` на `outputs/` или `cache/` | UID/GID bind-mount не совпадает | `DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) docker compose up` или `chown` каталоги |
| `port is already allocated` | Порт 8000 занят | Измените `APP_PORT`/`ports` в compose или остановите старый контейнер |
| Контейнер `unhealthy` | Первый старт скачивает модель; `/health` закрыт ключом | Смотрите `docker compose logs`; healthcheck использует `/ui/config`, `start_period` 600 с |
| Docker ходит не на тот LLM | В контейнер попал хостовый `LLM_BASE_URL` | Задайте `LLM_BASE_URL_DOCKER=http://host.docker.internal:1234/v1` |

## 2. LM Studio и LLM

| Симптом | Причина | Решение |
|---|---|---|
| `Connection refused` к `localhost:1234` | LM Studio не запущен или другой порт | Запустите сервер в LM Studio, проверьте `LLM_BASE_URL` |
| `400 Invalid model identifier` | `LLM_API_NAME` не совпадает с id модели | Укажите id из `GET /v1/models`, например `google/gemma-4-e4b` |
| Ответ — отказ с `llm_unavailable_or_empty` | LM Studio вернул пустой ответ или недоступен | Проверьте модель, повторите запрос |
| Ответ — отказ с `no_fragments` | Поиск не нашёл фрагментов | Проверьте индекс, `SAVE_DIR` и вопрос |
| Ответ без ссылок заменён отказом | Модель не поставила валидные `[n]` | Это строгий режим H2-6; повторите или смените модель |
| `grounding_errors: invalid_citations` | Ссылка вне диапазона фрагментов | Проверьте ответ; при повторе — сообщите разработчику |
| `grounding_errors: source_mismatch` | Цитата не соответствует фрагменту | Проверьте метаданные и разметку; сообщите разработчику |
| Медленный первый ответ | Загрузка модели/эмбеддингов | Первый запуск всегда дольше; см. [PERFORMANCE.md](./PERFORMANCE.md) |

## 3. Индекс

| Симптом | Причина | Решение |
|---|---|---|
| Индекс «пустой», хотя PDF есть | `SAVE_DIR` при индексации и поиске различаются | Сверьте `.env`, `--output_dir` и `show_index_stats.py` |
| В логе `Формат индекса не подтверждён` | Индекс собран до H0-8 | `python scripts/index_manifest.py --adopt` или переиндексация |
| В логе `Индекс собран другим форматом` | Индекс и код несовместимы | Переиндексируйте или откатитесь на версию кода под индекс |
| `QA-кэш: N записей не прочитаны` | Старая схема или другая модель | Записи будут пересчитаны; действие не требуется |
| OCR не работает | Нет Tesseract или языка `rus` | Установите Tesseract и языковой пакет, затем `--enable_ocr --ocr_language rus` |
| Индексация падает на PDF | Повреждённый файл или OCR | Проверьте файл, попробуйте без OCR, смотрите `indexing.log` |
| Долгая индексация | LLM извлекает факты | Это ожидаемо; ускоряет GPU/меньший корпус |

## 4. Веб-интерфейс

| Симптом | Причина | Решение |
|---|---|---|
| Страница не открывается | Сервер не запущен или другой порт | `uvicorn astro_app.web:app --host 127.0.0.1 --port 8000` |
| `404` на `/ui/app.js` или `/ui/styles.css` | Статика не попала в установку | Переустановите пакет (`pip install -e ".[dev]"`); static входит в package-data |
| Диалог ключа не принимает ключ | Неверный `API_KEY` | Сверьте ключ с `.env`; смена ключа требует перезапуска сервера |
| `401` на `/api/ask` | Нет или неверный `X-API-Key` | Введите ключ в диалоге; проверьте заголовок |
| `503` на любом data-маршруте | Сервер запущен без `API_KEY` при `API_KEY_REQUIRED=1` | Задайте `API_KEY` или явно `API_KEY_REQUIRED=0` для loopback |
| `400` «Неизвестный режим» | В запросе режим вне `5/2/1` | Используйте только три режима UI |
| `500` «Система не инициализирована» | Индекс не загрузился | Проверьте `SAVE_DIR`, логи старта, `show_index_stats.py` |
| Ответ есть, но цитат нет | Модель не сослалась на фрагменты | Строгий режим заменяет такой ответ отказом; повторите запрос |
| История пуста | Другой браузер/профиль или очищены данные сайта | История хранится в `localStorage`, сервер её не знает |
| Экспорт пустой | Нет текущего ответа или истории | Сначала задайте вопрос; история — только в этом браузере |
| Оценка не сохраняется | Нет прав на `FEEDBACK_PATH` | Проверьте каталог и права; по умолчанию `outputs/feedback/feedback.jsonl` |

## 5. API и CORS

| Симптом | Причина | Решение |
|---|---|---|
| `401` с корректным ключом | Ключ читается при импорте; сервер не перезапущен | Перезапустите сервер после смены `.env` |
| Браузер блокирует запрос по CORS | Origin не в `API_ALLOWED_ORIGINS` | Добавьте origin через запятую и перезапустите |
| `404` на `/docs` | Не тот процесс/порт | `/docs` есть у `search_api`; проверьте URL |
| `400` или файл в `failed_files` при загрузке PDF | Файл больше `MAX_UPLOAD_MB`, не PDF или небезопасное имя | Увеличьте лимит, проверьте формат и имя файла |
| `/search` возвращает пусто | Индекс пуст или вопрос вне корпуса | Проверьте индекс; это может быть корректный отказ |

## 6. Качество и метрики

| Симптом | Причина | Решение |
|---|---|---|
| `page recall@5` ниже 80 % | M8 (ablation упрощённых ретриверов) даёт максимум 73.9 %; полный H1-6-путь в M7 — 87.0 % | Проверьте, что запущен полный путь, а не ablation; см. [MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md) и [H1_6_PAGE_RETRIEVAL_2026-11-05.md](./H1_6_PAGE_RETRIEVAL_2026-11-05.md) |
| Answer rate/accuracy ниже порогов | Baseline H2-6 до H1-6; после M7 — 100 % и 73.9 % | См. [H2_6_GROUNDING_REPORT_2026-11-05.md](./H2_6_GROUNDING_REPORT_2026-11-05.md) |
| Citation precision ниже 0.80 | M7: 0.621 — модель добавляет лишние цитаты (recall 0.848) | Сузить промпт цитирования; см. [MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md) |
| Метрики не воспроизводятся | Нет индекса/PDF/LM Studio | Нужны `outputs/astrology_test` и LM Studio; команды в HOW_TO_RUN §7.2 |
| Предупреждение «ПЕРЕРАНК НЕ ВЫПОЛНЕН» | RU-CE не скачался/недоступен | Проверьте сеть; строгий режим — `HIPPORAG_REQUIRE_CROSSENCODER=1` |
| Нужен чанковый baseline вместо page-level | H1-6 включает страницы | `ASTRO_RETRIEVER_PAGE_LEVEL=0` |

## 7. Диагностика

```bash
# версия Python и пакетов
python --version
pip show hipporag

# LM Studio
curl http://localhost:1234/v1/models

# индекс
python scripts/show_index_stats.py --save_dir outputs/astrology_test

# здоровье и метрики API (ключ обязателен)
curl -H "X-API-Key: your-secret-key" http://127.0.0.1:8000/health
curl -H "X-API-Key: your-secret-key" http://127.0.0.1:8000/metrics

# unit-тесты без индекса и LLM
pytest tests/unit -m unit -q

# ссылки в документации
python scripts/check_docs_links.py
```

### 7.1. Почему запрос медленный

1. Возьмите `X-Request-ID` из заголовка ответа (в браузере — DevTools →
   Network).
2. Найдите событие `request_completed` с этим `request_id` и полем
   `slow=true`: там есть `duration_ms` и маршрут.
3. Для `/search` откройте `search_completed`, для `/api/ask` —
   `ask_completed` с тем же `request_id` и посмотрите `stages`:
   - `ask.retrieval` — поиск и переранжирование;
   - `ask.answer` — вызов LLM, разбор цитат и repair-retry;
   - `search.rag_qa` — полный вызов `HippoRAG.rag_qa`;
   - `search.build_response` — сборка цитат и ответа.
4. `/metrics` или `/ui/dashboard` покажет, какой маршрут и этап
   накапливает задержку: `routes[].avg_ms/max_ms`,
   `stages[].avg_ms/max_ms`, `latency.p95_ms`.

Типичные причины: первый запрос после старта (загрузка модели),
`ask.answer` заметно больше `ask.retrieval` (медленная LM Studio или
повтор из-за цитат), `search.rag_qa` больше `build_response`
(поиск/переранжирование). Порог `slow=true` задаётся `SLOW_REQUEST_MS`
(по умолчанию 5000 мс).

### 7.2. Почему ответ пустой

Событие `answer_empty` содержит `reason`, `path`, число найденных
фрагментов и этапы. Причины:

| `reason` | Что означает | Что делать |
|---|---|---|
| `no_fragments` | Поиск не нашёл фрагментов, LLM не вызывался | Проверьте индекс, `SAVE_DIR`, режим и вопрос |
| `llm_unavailable_or_empty` | LM Studio недоступен или вернул пустую строку | Проверьте `/v1/models`, модель и `LLM_API_NAME` |
| `no_inline_citations` | Модель ответила без валидных `[n]` | Строгий режим H2-6; повторите или смените модель |
| `no_citations` | В ответе нет использованных цитат | Проверьте grounding-контракт и ответ модели |
| `abstained` | Явный отказ без более конкретной причины | Посмотрите `grounding_errors` в том же событии |

Сводка по причинам — `empty_answers_by_reason` в `/metrics` и на
`/ui/dashboard`. Вопросы, ответы и тексты фрагментов в события не
пишутся.

Если проблема воспроизводится, приложите:

1. команду запуска и версию Python;
2. фрагмент лога без секретов и содержимого корпуса;
3. ответ `/health` и, если запрос медленный или пустой,
   `X-Request-ID` и события `request_completed` / `answer_empty`;
4. вывод `show_index_stats.py`;
5. режим и вопрос, на котором воспроизводится ошибка.
