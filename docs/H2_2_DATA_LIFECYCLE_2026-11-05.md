# H2-2: жизненный цикл индекса

**Дата:** 2026-11-05
**Задача:** [ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md), H2-2
**Код:** `src/astro_app/index_lifecycle.py`, `src/hipporag/search_api/server.py`
**Тесты:** `tests/unit/test_index_lifecycle.py`, `tests/unit/test_search_api_index.py`

## 1. Что сделано

Раньше индекс обновлялся только вручную: остановить сервер, скопировать
каталог, запустить `index_pdfs.py`. Теперь те же операции доступны через
API, а `/upload-and-ingest` делает бэкап и переиндексацию автоматически.

| Возможность | Как |
|---|---|
| Статус индекса и операции | `GET /index/status` |
| Список документов | `GET /index/documents` |
| Бэкап | `POST /index/backup`, `GET /index/backups` |
| Восстановление | `POST /index/restore` |
| Удаление документов | `POST /index/delete` |
| Переиндексация PDF | `POST /index/reindex` |
| Загрузка PDF из UI/API | `POST /upload-and-ingest` |
| Версия индекса в health | `GET /health` → `index_version`, `index_state`, `index_operation`, `documents` |

Все маршруты, кроме статики, требуют `X-API-Key` (см.
[ADMIN_GUIDE.md](./ADMIN_GUIDE.md) §4.2).

## 2. Как устроено

`IndexLifecycle` не импортирует FastAPI: он получает каталог индекса и
колбэки `get_hipporag` / `close_hipporag` / `reload_hipporag`. Поэтому его
можно использовать из API, CLI и тестов.

### 2.1. Бэкап

- zip-архив каталога индекса, включая `graph.pickle`, манифест и parquet;
- каталог по умолчанию — `<каталог индекса>/../backups`;
- имя вида `index-20261105-193000-<label>.zip`;
- sidecar `<имя>.zip.json` с путём, размером, числом файлов и меткой;
- архив проверяется на `graph.pickle` и `index_manifest.json`;
- во время другой операции бэкап запрещён (`409`), чтобы не поймать
  наполовину записанный parquet.

### 2.2. Восстановление

1. Проверяется имя: только basename с `.zip`, без выхода за `backups/`.
2. Архив распаковывается во временный каталог рядом с индексом.
3. Проверяются `graph.pickle` и формат манифеста.
4. Текущий `HippoRAG` закрывается, каталог индекса переименовывается,
   на его место встаёт распакованный.
5. `HippoRAG` создаётся заново.
6. Если перезагрузка упала — старый каталог возвращается на место, и
   выполняется попытка поднять его снова.

Ошибки: `404` — копия не найдена, `400` — плохой архив/имя, `409` — идёт
другая операция.

### 2.3. Удаление документов

Документы адресуются по `source_file` из metadata чанков.

- чанк, у которого все источники входят в удаляемый набор, удаляется
  обычным `HippoRAG.delete`;
- чанк, общий для удаляемого и остающегося документа, **не удаляется**:
  у него убирается только metadata-запись удаляемого источника
  (`EmbeddingStore.remove_metadata_entries`), иначе вместе с одним PDF
  пропали бы данные другого;
- отсутствующие документы возвращаются в `missing_documents`, а не
  считаются ошибкой.

Ответ содержит `deleted_documents`, `missing_documents`, `deleted_chunks`,
`shared_chunks_updated`, `skipped_shared_chunks`.

### 2.4. Переиндексация

1. Проверяется, что все PDF существуют.
2. Создаётся бэкап `before-reindex`.
3. Старые чанки этих же `source_file` удаляются (кроме общих).
4. Вызывается `HippoRAG.index_pdfs`.
5. Обновляется `index_manifest.json`.

`/upload-and-ingest` сохраняет файл под исходным basename в
`temp_uploads/` и вызывает тот же `reindex` в фоне, поэтому повторная
загрузка не оставляет дубликатов.

## 3. Проверка

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe -m pytest tests/unit/test_index_lifecycle.py tests/unit/test_search_api_index.py -q
```

Ожидаемо: **21 passed**.

Полный unit-набор:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -m unit -q
```

Ожидаемо: **457 passed, 1 skipped** (2026-11-05).

## 4. Живая приёмка

Приёмка H2-2: «загрузка PDF → индекс обновился → можно откатиться».
На реальном индексе она ещё не проводилась — нужен запущенный сервер с
`outputs/astrology_test`. Порядок:

1. Запустить сервер и проверить `GET /index/status`.
2. `POST /index/backup` — запомнить имя копии.
3. `POST /upload-and-ingest` с новым PDF.
4. Дождаться `state = idle` в `/index/status`, проверить
   `GET /index/documents` и контрольный поиск.
5. `POST /index/restore` с именем копии из шага 2.
6. Проверить, что документ из шага 3 исчез, а поиск работает.

### 4.1. Результат приёмки — PASS (проведена 2026-11-05/06)

Условия: код `95df0af` (+ офлайн-скрипты H2-6), индекс
`gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2`
(1369 чанков, 3 PDF, манифест `ok`), LM Studio `google/gemma-4-e4b`,
сервер `uvicorn src.hipporag.search_api.server:app` на `127.0.0.1:8000` с
inline `API_KEY` (fail-closed подтвердился: без ключа сервер не стартует).

| Шаг | Операция | Результат |
|---|---|---|
| 1 | `GET /health`, `GET /index/status` | 1369 чанков, `state=idle`, `format_state=ok`, `auth_required=true` |
| 2 | `POST /index/backup` label=acceptance-h2-2 | `index-20261105-205941-acceptance-h2-2.zip`, 65 МБ, 5 файлов |
| 3 | `POST /upload-and-ingest` с `h2_2_acceptance.pdf` (1 страница) | `processed_files=1`, фоновая индексация, авто-бэкап `before-reindex` |
| 4 | `state=idle`; `GET /index/documents` | 4-й документ: `h2_2_acceptance.pdf`, 1 стр, 1 чанк |
| 4 | Контрольный поиск `POST /search` «unique token XYZZY…» | тестовый чанк — **№ 1** в выдаче, ответ `Unique token: XYZZY [1].` с цитатой |
| 5 | `POST /index/restore` из копии шага 2 | `format_state=ok`, HippoRAG перезагружен |
| 6 | `GET /index/documents`, `GET /health` | снова 3 документа, 1369 чанков, `state=ok` |
| 6 | Поиск XYZZY после отката | тестовый документ **исчез** из выдачи |
| 6 | Контрольный поиск по старому корпусу («Шен-ци гуа 1») | корректный ответ «юго-восток [1, 5]» с цитатой из `Эзотерический фэншуй.pdf` |

Тестовый PDF генерируется служебным скриптом
`scripts/make_h2_2_test_pdf.py` (1 страница, уникальный токен `XYZZY`,
извлечение проверено тем же PyMuPDF, что в ingestion).

**Находка (не блокирует приёмку).** Первая попытка с почти пустой страницей
(160 символов): парсер извлёк 0 чанков, `reindex` завершился «успехом» за 3 с,
манифест перезаписан без изменений, API вернул `success`. Рекомендация:
`reindex`/`upload-and-ingest` должны возвращать предупреждение (или не-2xx),
когда из файла извлечено 0 чанков, иначе тихий no-op выглядит как успех.
Зафиксировано в §5 как ограничение первой версии.

После приёмки сервер остановлен, `temp_uploads/` удалён; в
`outputs/astrology_test/backups/` остались три копии (приёмочная и две
автоматические `before-reindex`) — ротации нет, это известное ограничение §5.

## 5. Ограничения

- Панели администратора в Web UI нет: операции доступны через API и
  Swagger (`/docs`). Это первая версия H2-2.
- Автоматической ротации бэкапов нет; `llm_cache/` входит в архив, поэтому
  копии могут быть большими.
- `reindex`/`upload-and-ingest` при 0 извлечённых чанков завершаются
  «успехом» без предупреждения в ответе (находка живой приёмки §4.1);
  сервер пишет WARNING в лог. Требуется явная диагностика в ответе API.
- Удаление документа не пересчитывает граф заново: удаляются чанки и
  связанные с ними сущности/факты силами `HippoRAG.delete`; общие чанки
  сохраняются.
- `restore` и `reindex` не запускаются параллельно: вторая операция
  получает `409`.
