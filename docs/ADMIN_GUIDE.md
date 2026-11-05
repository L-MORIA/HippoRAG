# Руководство администратора HippoRAG

> **Кому:** человеку, который разворачивает систему, следит за индексом,
> ключами и сохранностью данных.
>
> **Связанные документы:** [HOW_TO_RUN.md](./HOW_TO_RUN.md) (установка и
> команды), [INDEXING_GUIDE.md](./INDEXING_GUIDE.md) (индексация),
> [ARCHITECTURE.md](./ARCHITECTURE.md) (устройство),
> [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) (ошибки).

## 1. Что администрируется

| Компонент | Что это | Где |
|---|---|---|
| Индекс | Чанки, эмбеддинги, граф знаний, кэш LLM | `SAVE_DIR` (по умолчанию `outputs/astrology_test`) |
| API | FastAPI-приложение поиска и ингеста | `src/hipporag/search_api/server.py` |
| Web UI | Статика и `/api/ask`, `/api/feedback` | `src/astro_app/web/` |
| Обратная связь | Append-only JSONL | `outputs/feedback/feedback.jsonl` |
| QA-кэш | Кэш готовых ответов CLI | `outputs/qa_cache/qa_cache.json` |
| Конфигурация | `.env` | корень репозитория (не коммитится) |

## 2. Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `LLM_NAME` | `gemma-4-e4b` | Имя каталога индекса и кэша |
| `LLM_API_NAME` | — | id модели в LM Studio, если отличается от каталога |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | Адрес OpenAI-совместимого API |
| `SAVE_DIR` | `outputs/astrology_test` | Каталог индекса |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Модель эмбеддингов |
| `API_KEY` | — | Ключ доступа к data-API |
| `API_KEY_REQUIRED` | `1` | `0/false/no/off` — отключить ключ (только loopback) |
| `API_ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Разрешённые CORS-origin |
| `MAX_UPLOAD_MB` | `200` | Лимит загружаемого PDF |
| `FEEDBACK_PATH` | `outputs/feedback/feedback.jsonl` | Путь журнала оценок |
| `HIPPORAG_REQUIRE_CROSSENCODER` | `0` | `1` — падать, если RU-CE недоступен |
| `ASTRO_RETRIEVER_PAGE_LEVEL` | `1` | `0` — вернуть чанковый retrieval H1-6 |

Полный шаблон — `.env.example`. После изменения `.env` перезапустите сервер:
часть настроек читается при импорте модуля.

## 3. Жизненный цикл индекса

### 3.1. Сборка

```bash
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology_test
python scripts/show_index_stats.py --save_dir outputs/astrology_test
```

`--output_dir` и `SAVE_DIR` должны совпадать. Индекс лежит во вложенном
каталоге вида:

```text
outputs/astrology_test/
└── gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2/
    ├── graph.pickle
    ├── index_manifest.json
    ├── chunk_embeddings/vdb_chunk.parquet
    ├── entity_embeddings/vdb_entity.parquet
    ├── fact_embeddings/vdb_fact.parquet
    └── llm_cache/
```

### 3.2. Проверка версии

При загрузке проверяется `index_manifest.json`. Если в логе написано
«Формат индекса не подтверждён» или «Индекс собран другим форматом»:

- **предпочтительно** переиндексировать корпус;
- для старого индекса можно выполнить `python scripts/index_manifest.py --adopt`,
  но это подтверждает формат без пересборки — используйте осознанно;
- откат к предыдущей версии кода допустим, если индекс ей соответствует.

Версии форматов собраны в `src/astro_app/versions.py`:
`INDEX_FORMAT_VERSION`, `QA_CACHE_VERSION`, `PACKAGE_DATA_VERSION`.

### 3.3. Бэкап

Остановите сервер и скопируйте:

1. каталог индекса `SAVE_DIR`;
2. `.env` (без публикации в репозиторий);
3. `outputs/feedback/feedback.jsonl`, если оценки нужны;
4. `outputs/qa_cache/qa_cache.json`, если нужен кэш CLI.

Индекс можно копировать как обычный каталог: это parquet-файлы, `graph.pickle`
и манифест. Не копируйте `temp_uploads/` — это временные загрузки.

### 3.4. Восстановление

1. Остановите сервер.
2. Замените каталог индекса резервной копией.
3. Проверьте `index_manifest.json` и запустите
   `python scripts/show_index_stats.py --save_dir ...`.
4. Запустите сервер и проверьте `/health`.

### 3.5. Переиндексация

1. Остановите сервер.
2. Сохраните бэкап текущего индекса.
3. Запустите `index_pdfs.py` с тем же `--output_dir`.
4. Проверьте статистику и версию.
5. Запустите сервер и задайте контрольный вопрос.

> **Ограничение:** добавление, удаление и переиндексация **из UI/API** пока не
> реализованы — это задача H2-2. Маршруты `/ingest` и `/upload-and-ingest`
> существуют в API, но в интерфейсе не выведены; для штатного обновления
> используйте `index_pdfs.py`.

## 4. API-сервер

### 4.1. Запуск

```bash
export API_KEY=your-secret-key
uvicorn src.hipporag.search_api.server:app --host 127.0.0.1 --port 8000
```

По умолчанию сервер слушает только loopback. Наружу порт не выставляйте без
reverse-proxy с TLS.

### 4.2. Аутентификация

Все data-маршруты требуют заголовок:

```text
X-API-Key: ваш-секретный-ключ
```

| `API_KEY` | `API_KEY_REQUIRED` | Поведение |
|---|---|---|
| задан | `1` (по умолчанию) | без ключа → `401`, с ключом → `200` |
| не задан | `1` | сервер **не стартует** (fail-closed) |
| не задан | `0` | запросы без ключа (только локальная разработка) |

Смена ключа требует перезапуска: ключ читается один раз при импорте.
`API_ALLOWED_ORIGINS` — единственный источник правды для CORS; ответы `401`
тоже получают корректные CORS-заголовки.

### 4.3. Маршруты

| Маршрут | Назначение | Защита |
|---|---|---|
| `GET /health` | Статус, число чанков/фактов, `auth_failures` | `X-API-Key` |
| `POST /search` | Поиск через `HippoRAG.rag_qa` | `X-API-Key` |
| `POST /ingest` | Индексация PDF по путям | `X-API-Key` |
| `POST /upload-and-ingest` | Загрузка PDF и индексация | `X-API-Key` |
| `GET /docs` | Swagger UI | `X-API-Key` |
| `GET /` , `GET /ui/*` | Статика Web UI | без ключа |
| `POST /api/ask` | Вопрос из Web UI | `X-API-Key` |
| `POST /api/feedback` | Оценка ответа | `X-API-Key` |

> `/docs` защищён ключом: в браузере без заголовка `X-API-Key` он вернёт `401`.
> Для локальной разработки можно временно запустить с `API_KEY_REQUIRED=0`.

### 4.4. Web UI

```powershell
$env:API_KEY = "your-secret-key"
uvicorn astro_app.web:app --host 127.0.0.1 --port 8000
```

`astro_app.web:app` — то же приложение, что и `search_api`, с добавленными
UI-маршрутами. Отдельный процесс и отдельный порт не нужны.

## 5. Обратная связь

Каждая оценка — одна строка JSONL:

```json
{"schema_version": 1, "id": "...", "created_at": "...", "rating": "up", "query": "...", "answer": "...", "mode": "5", "comment": "...", "citations": []}
```

- Файл append-only; приложение только дописывает строки.
- Персональных данных нет: IP, user-agent и ID пользователя не сохраняются.
- Путь задаётся `FEEDBACK_PATH`; каталог создаётся автоматически.
- Для выгрузки/анализа читайте файл вручную; UI-просмотра пока нет.
- Ротация не автоматизирована: остановите сервер, переименуйте файл и
  запустите снова. Следите за правами на каталог.

## 6. Логи и мониторинг

- Логи пишутся в stdout с уровнем `INFO`.
- `/search` логирует `abstained`, `grounding_errors`, число цитат, число
  фрагментов, latency и p95 по последним запросам.
- `/health` возвращает `auth_required` и `auth_failures` — счётчик отказов
  авторизации с запуска.
- Структурных логов, per-stage timings и дашборда пока нет — это H2-4.
  Для базовой диагностики используйте `grep` по логам и `/health`.

## 7. Чек-лист безопасности

- [ ] `API_KEY` задан, `API_KEY_REQUIRED=1`.
- [ ] Сервер слушает `127.0.0.1`, наружу — только через reverse-proxy с TLS.
- [ ] `API_ALLOWED_ORIGINS` содержит только доверенные origin.
- [ ] `.env` не коммитится и недоступен другим пользователям машины.
- [ ] `outputs/feedback/` и каталог индекса доступны только эксплуататору.
- [ ] Бэкап индекса и `.env` снят и проверен на восстановление.
- [ ] `API_KEY_REQUIRED=0` не используется на публичном интерфейсе.

## 8. Что ещё не реализовано

| Задача | Статус |
|---|---|
| H2-2: add/delete/reindex из UI/API, backup/restore, статус индексации | ⏳ |
| H2-3: Docker, Windows-установщик, one-command start | ⏳ |
| H2-4: структурные логи, метрики, дашборд | ⏳ |
| H2-5: полный набор документации | ✅ этот документ и связанные гайды |
| H2-6: перезамер grounding-метрик | ⏳ ждёт индекс и LM Studio |

## 9. Быстрые команды

```bash
# индекс
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology_test
python scripts/show_index_stats.py --save_dir outputs/astrology_test

# API
uvicorn src.hipporag.search_api.server:app --host 127.0.0.1 --port 8000

# Web UI
uvicorn astro_app.web:app --host 127.0.0.1 --port 8000

# тесты
pytest tests/unit -m unit -q
```
