# Архитектура HippoRAG

> **Назначение:** локальная RAG-система по корпусу астрологических PDF:
> поиск фрагментов, генерация ответа со ссылками `[n]`, явный отказ, если
> ответа нет в источниках.
>
> **Связанные документы:** [USER_GUIDE.md](./USER_GUIDE.md),
> [ADMIN_GUIDE.md](./ADMIN_GUIDE.md), [H2_1_WEB_UI_2026-11-05.md](./H2_1_WEB_UI_2026-11-05.md),
> [H2_6_GROUNDING_REPORT_2026-11-05.md](./H2_6_GROUNDING_REPORT_2026-11-05.md).

## 1. Общая схема

```text
┌──────────────────────────────┐
│ Браузер: HTML/CSS/JS         │
│ история — localStorage       │
└──────────────┬───────────────┘
               │ X-API-Key для /api/*
┌──────────────▼───────────────┐
│ FastAPI (одно приложение)    │
│ hipporag.search_api.server   │
│  ├─ /, /ui/*      (статика)  │
│  ├─ /api/ask      (Web UI)   │
│  ├─ /api/feedback (JSONL)    │
│  ├─ /search       (rag_qa)   │
│  └─ /health                  │
└──────┬───────────────┬───────┘
       │               │
┌──────▼──────┐  ┌─────▼────────────────┐
│ astro_app   │  │ HippoRAG core        │
│ 23 режима   │  │ граф, эмбеддинги,    │
│ retrieval   │  │ PDF-ингест, QA       │
└──────┬──────┘  └─────┬────────────────┘
       │               │
┌──────▼───────────────▼───────────────┐
│ Индекс: parquet + graph.pickle       │
│ + llm_cache, QA cache, feedback.jsonl│
└──────────────────────────────────────┘
       │
┌──────▼───────────────────────────────┐
│ LM Studio (OpenAI-compatible API)    │
│ LLM_BASE_URL, по умолчанию :1234     │
└──────────────────────────────────────┘
```

Всё работает локально. Наружу уходят только запросы к `LLM_BASE_URL`;
по умолчанию это LM Studio на loopback.

## 2. Компоненты

| Слой | Путь | Ответственность |
|---|---|---|
| Точка входа CLI | `main.py`, `src/astro_app/cli.py`, `menu.py` | Интерактивное меню, выбор режима, вывод |
| Режимы поиска | `src/astro_app/modes/` | 23 `ModeSpec`: BM25, граф, dense, RRF, LLM-режимы |
| Page-level retrieval | `src/astro_app/page_retrieval.py` | Группировка чанков по странице, RRF, RU-CE (H1-6) |
| Grounding | `src/astro_app/grounded_qa.py`, `src/hipporag/qa/grounding.py` | Контракт `ОТВЕТ / ИСТОЧНИКИ / ОТКАЗ`, парсер `[n]` |
| Web UI | `src/astro_app/web/` | Три режима, `/api/ask`, `/api/feedback`, статика |
| API | `src/hipporag/search_api/server.py` | FastAPI, auth guard, `/search`, `/health`, ингест |
| Ядро | `src/hipporag/HippoRAG.py`, `StandardRAG.py` | Индекс, граф, retrieval, QA |
| Хранилища | `src/hipporag/embedding_store.py`, `vector_stores/` | Parquet-эмбеддинги, метаданные |
| PDF-ингест | `src/hipporag/pdf_ingestion/` | Извлечение текста, OCR, чанкинг |
| Кэш | `src/astro_app/qa_cache.py`, `llm_cache/` | QA-кэш CLI и кэш LLM |
| Версии | `src/astro_app/versions.py` | `INDEX_FORMAT_VERSION`, `QA_CACHE_VERSION`, `PACKAGE_DATA_VERSION` |
| Eval | `scripts/eval_*.py`, `scripts/grounding_*.py`, `eval/` | Golden set, recall, grounding-метрики |

## 3. Путь запроса Web UI

1. Браузер отправляет `POST /api/ask` с `X-API-Key`.
2. `api_key_guard` проверяет ключ (кроме публичной статики и `OPTIONS`).
3. `astro_app.web.server.ask` проверяет, что `hipporag` инициализирован.
4. `astro_app.web.service.answer_question`:
   - выбирает `ModeSpec` из `astro_app.modes`;
   - вызывает режим и нормализует выдачу (5- и 6-элементные кортежи,
     `PageHit`, строки);
   - дедуплицирует фрагменты и обрезает текст до 1600 символов;
   - передаёт фрагменты в `answer_grounded`.
5. `answer_grounded` формирует промпт, вызывает LLM через `lmstudio_chat`,
   парсит `[n]`, при необходимости делает один repair-retry.
6. Сервис собирает `citations` **только по использованным** номерам,
   `fragments`, `retrieved_docs`, `abstained`, `grounding_errors`,
   `confidence` и тайминги.
7. Браузер отображает ответ, цитаты и диагностику; история пишется в
   `localStorage`.

Если фрагментов нет, LLM не вызывается: возвращается общий отказ с
`no_fragments`. Если LLM вернул пустую строку — отказ с
`llm_unavailable_or_empty`.

## 4. Режимы и единицы поиска

| Режим | Путь | Единица выдачи | LLM на этапе поиска |
|---|---|---|---|
| `5` «Быстрый» | `search_bm25_hybrid_ru` | Чанк | Нет |
| `2` «Граф» | графовый PPR | Чанк | Нет |
| `1` «Максимальный охват» | multi-vector + переформулировка | Чанк | Да |

В CLI доступны все 23 режима. В Web UI — только три измеренных, чтобы не
обещать невыверенную скорость.

H1-6 добавил page-level путь (`search_bm25_hybrid_ru_pages`) и адаптер
`scripts/grounding_astro_retriever.py`. Он прошёл приёмку в M7: **87.0 %**
page recall@5 при цели 80 %. В UI пока не подключён — это следующий шаг.
Чанковый baseline H2-6 остаётся доступен через
`ASTRO_RETRIEVER_PAGE_LEVEL=0`.

## 5. Контракт заземления

```text
ОТВЕТ: Овну соответствуют прихожая и плита [1]. Управитель — Марс [1].
ИСТОЧНИКИ: 1
ОТКАЗ: нет
```

Правила:

- каждая ссылка `[n]` должна входить в диапазон переданных фрагментов;
- ответ без валидных ссылок заменяется общим отказом
  `В предоставленных фрагментах нет ответа на этот вопрос.`;
- допускается один repair-retry, повторный ответ снова проверяется;
- `confidence` — грубая оценка заземления (`0.0 / 0.5 / 1.0`), а не
  вероятность правильности;
- `citations` содержит только использованные цитаты, полный retrieval —
  в `retrieved_docs` / `fragments`.

## 6. Данные и хранилища

| Данные | Формат | Где | Коммитится |
|---|---|---|---|
| Индекс | parquet + `graph.pickle` + manifest | `SAVE_DIR` | Нет |
| Кэш LLM | SQLite | `<index>/llm_cache/` | Нет |
| QA-кэш | JSON | `outputs/qa_cache/qa_cache.json` | Нет |
| Обратная связь | JSONL, append-only | `outputs/feedback/feedback.jsonl` | Нет |
| История UI | `localStorage` | браузер пользователя | Нет |
| Golden set | JSONL | `eval/grounding/`, `eval/dataset/` | Да |
| Конфигурация | `.env` | корень | Нет |

`outputs/` целиком в `.gitignore`: корпус, индекс и пользовательские оценки
не попадают в репозиторий.

## 7. Безопасность

- Публично: `/`, `/ui/*` — только статика и список режимов.
- Под `X-API-Key`: `/api/ask`, `/api/feedback`, `/search`, `/health`,
  `/ingest`, `/upload-and-ingest`, `/docs`.
- `API_KEY_REQUIRED=1` по умолчанию: без ключа сервер не стартует.
- CORS ограничен `API_ALLOWED_ORIGINS`; `401` тоже получает CORS-заголовки.
- Ключ UI хранится в `sessionStorage` только на время вкладки.
- Персональные данные в feedback не собираются.
- Нет ролей, пользователей и rate limiting: это общий ключ на инстанс.

## 8. Версионирование

Три версии в `src/astro_app/versions.py`:

- `INDEX_FORMAT_VERSION` — раскладка индекса; пишется в
  `index_manifest.json` и проверяется при загрузке;
- `QA_CACHE_VERSION` — схема записей QA-кэша;
- `PACKAGE_DATA_VERSION` — формат встроенных данных, сверяется с
  `src/astro_app/data/data_manifest.json`.

При изменении раскладки индекса, схемы кэша или встроенных данных
соответствующая версия поднимается; старые записи пересчитываются или
индекс переиндексируется.

## 9. Тесты и замеры

- `pytest tests/unit -m unit -q` — быстрый unit-набор без индекса и LLM.
- `scripts/eval_search_quality.py` — recall@k, MRR, nDCG, latency на
  `eval/dataset/questions.jsonl`.
- `scripts/eval_grounding.py` — метрики заземления на golden set.
- `scripts/grounding_retriever.py` — прогон через `HippoRAG.rag_qa`.
- `scripts/grounding_astro_retriever.py` — прогон через режим 5.
- `scripts/measure_page_ceiling.py` — потолок page recall (H1-6).

Метрики без индекса и LM Studio не воспроизводятся: в CI проверяется
контракт датасета и корректность формул, а не качество на реальном корпусе.

## 10. Ограничения и направления

- H2-2: нет add/delete/reindex из UI, а также backup/restore и статуса индексации.
- H2-3: нет Docker, Windows-установщика и one-command start.
- H2-4: нет структурных логов, per-stage timings, метрик и дашборда.
- H1-6: page-level retrieval реализован и прошёл приёмку (M7: **87.0 %** page recall@5); в UI не подключён.
- H2-6: после H1-6 7 из 8 метрик PASS; citation precision **0.621** < 0.80 — остаётся.
- H2-1: приёмка с 5 внешними пользователями ещё не проведена.
