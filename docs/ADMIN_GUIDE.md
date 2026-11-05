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

**Вручную (без остановки не обойтись):**

1. каталог индекса `SAVE_DIR`;
2. `.env` (без публикации в репозиторий);
3. `outputs/feedback/feedback.jsonl`, если оценки нужны;
4. `outputs/qa_cache/qa_cache.json`, если нужен кэш CLI.

Индекс можно копировать как обычный каталог: это parquet-файлы, `graph.pickle`
и манифест. Не копируйте `temp_uploads/` — это временные загрузки.

**Через API (сервер можно не останавливать):**

```bash
curl -X POST http://127.0.0.1:8000/index/backup \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label": "before-update"}'
```

`POST /index/backup` создаёт zip рядом с каталогом индекса — в
`<SAVE_DIR>/backups/`, и возвращает имя, путь, число файлов и размер.
`GET /index/backups` перечисляет копии, новые сверху. К каждой копии
пишется sidecar `<имя>.zip.json` с метаданными. Во время другой операции
с индексом бэкап вернёт `409`: нельзя копировать наполовину записанный
parquet.

> Бэкап включает `llm_cache/`, поэтому может занимать сотни мегабайт.
> Автоматической ротации нет: старые копии удаляйте вручную.

### 3.4. Восстановление

**Через API:**

```bash
curl -X POST http://127.0.0.1:8000/index/restore \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "index-20261105-193000-before-update.zip"}'
```

Сервер закрывает текущий `HippoRAG`, распаковывает копию во временный
каталог, проверяет `graph.pickle` и `index_manifest.json`, подменяет каталог
и пересоздаёт `HippoRAG`. Если проверка или перезагрузка падает, старый
каталог возвращается на место. Ошибки: `404` — копия не найдена, `400` —
плохой архив или имя, `409` — идёт другая операция.

**Вручную:**

1. Остановите сервер.
2. Замените каталог индекса резервной копией.
3. Проверьте `index_manifest.json` и запустите
   `python scripts/show_index_stats.py --save_dir ...`.
4. Запустите сервер и проверьте `/health`.

### 3.5. Переиндексация

**Через API по путям:**

```bash
curl -X POST http://127.0.0.1:8000/index/reindex \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"pdf_paths": ["data/pdfs/new.pdf"]}'
```

Сервер сначала делает бэкап `before-reindex`, затем удаляет старые чанки
этих же файлов (по `source_file`) и индексирует PDF заново. Повторная
загрузка того же файла не оставляет дубликатов. Если чанк общий для двух
документов, удаляется только metadata-запись переиндексируемого источника,
а сам чанк остаётся.

**Через загрузку файла:** `POST /upload-and-ingest` сохраняет PDF в
`temp_uploads/` и запускает ту же переиндексацию в фоне — с бэкапом и
статусом. Это основной путь «загрузил PDF → индекс обновился».

**Наблюдение:**

```bash
curl http://127.0.0.1:8000/index/status -H "X-API-Key: $API_KEY"
curl http://127.0.0.1:8000/index/documents -H "X-API-Key: $API_KEY"
```

`/index/status` показывает `state` (`idle/running/error`), текущую операцию,
время начала/окончания, ошибку, версию формата, счётчики чанков/фактов/
документов и число бэкапов. `/index/documents` группирует чанки по
`source_file` и отдаёт число страниц и чанков.

**Вручную:** остановите сервер, сохраните бэкап, запустите `index_pdfs.py`
с тем же `--output_dir`, проверьте статистику и версию, запустите сервер.

> Панель администратора в Web UI пока не выведена: операции доступны через
> API и Swagger (`/docs`). Это осознанное ограничение первой версии H2-2.

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
| `POST /upload-and-ingest` | Загрузка PDF и индексация с бэкапом | `X-API-Key` |
| `GET /index/status` | Статус индекса и операции | `X-API-Key` |
| `GET /index/documents` | Документы в индексе | `X-API-Key` |
| `POST /index/backup` | Создать бэкап | `X-API-Key` |
| `GET /index/backups` | Список бэкапов | `X-API-Key` |
| `POST /index/restore` | Восстановить из бэкапа | `X-API-Key` |
| `POST /index/delete` | Удалить документы по `source_file` | `X-API-Key` |
| `POST /index/reindex` | Переиндексировать PDF по путям | `X-API-Key` |
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
- `/health` возвращает `auth_required`, `auth_failures`, `index_version`,
  `index_state` (`ok/missing/stale/...`), `index_operation`
  (`idle/running/error`) и число документов `documents`.
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
| H2-2: add/delete/reindex, backup/restore, статус индексации | ✅ API и `/upload-and-ingest`; панель в Web UI — ⏳ |
| H2-3: Docker, Windows-установщик, one-command start | ⏳ |
| H2-4: структурные логи, метрики, дашборд | ⏳ |
| H2-5: полный набор документации | ✅ этот документ и связанные гайды |
| H2-6: перезамер grounding-метрик | ✅ M7: 7/8 PASS; citation precision 0.621 ниже цели |

## 9. Быстрые команды

```bash
# индекс
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology_test
python scripts/show_index_stats.py --save_dir outputs/astrology_test

# API
uvicorn src.hipporag.search_api.server:app --host 127.0.0.1 --port 8000

# индекс через API
curl http://127.0.0.1:8000/index/status -H "X-API-Key: $API_KEY"
curl -X POST http://127.0.0.1:8000/index/backup -H "X-API-Key: $API_KEY"
curl -X POST http://127.0.0.1:8000/index/restore -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" -d '{"name": "index-....zip"}'

# Web UI
uvicorn astro_app.web:app --host 127.0.0.1 --port 8000

# тесты
pytest tests/unit -m unit -q
```
