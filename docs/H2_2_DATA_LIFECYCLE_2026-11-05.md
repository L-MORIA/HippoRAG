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

## 5. Ограничения

- Панели администратора в Web UI нет: операции доступны через API и
  Swagger (`/docs`). Это первая версия H2-2.
- Автоматической ротации бэкапов нет; `llm_cache/` входит в архив, поэтому
  копии могут быть большими.
- Удаление документа не пересчитывает граф заново: удаляются чанки и
  связанные с ними сущности/факты силами `HippoRAG.delete`; общие чанки
  сохраняются.
- `restore` и `reindex` не запускаются параллельно: вторая операция
  получает `409`.
