# H2-4. Наблюдаемость: структурные логи, метрики и дашборд

> **Дата:** 2026-11-06
> **Статус:** реализовано, unit- и TestClient-проверки зелёные; живая
> приёмка на корпусе с LM Studio не требуется для контракта метрик.
> **Связанные документы:** [ADMIN_GUIDE.md](./ADMIN_GUIDE.md),
> [TROUBLESHOOTING.md](./TROUBLESHOOTING.md),
> [ARCHITECTURE.md](./ARCHITECTURE.md),
> [ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md).

## 1. Задача и приёмка

По роадмапу H2-4 требует:

- структурные логи;
- per-stage timings;
- метрики запросов, ошибок, таймаутов и пустых ответов;
- `/health` с версией индекса и модели;
- простой дашборд.

**Приёмка:** по логам можно ответить, почему запрос был медленным или
вернул пустой ответ.

Реализация отвечает на это так:

- каждый HTTP-запрос получает `request_id` и событие
  `request_completed` с `duration_ms`, `status_code` и `slow`;
- `/search` и `/api/ask` дополнительно пишут `search_completed` /
  `ask_completed` с разбивкой по этапам;
- пустой ответ пишет отдельное событие `answer_empty` с причиной
  (`no_fragments`, `llm_unavailable_or_empty`, `no_inline_citations`,
  `no_citations`, `abstained`);
- те же данные агрегируются в `/metrics` и на странице
  `/ui/dashboard`.

## 2. Структурные события

Модуль: `src/hipporag/search_api/observability.py`.

События пишутся в логгер `hipporag.events` одной JSON-строкой без
префикса. Пример:

```json
{"ts":"2026-11-06T18:46:52.400332+00:00","level":"info","event":"request_completed","request_id":"733dc8bbb8934e98a8894edb61f30816","method":"POST","path":"/api/ask","route":"/api/ask","status_code":200,"duration_ms":842.3,"slow":false}
```

| Событие | Когда | Ключевые поля |
|---|---|---|
| `request_completed` | конец любого HTTP-запроса | `request_id`, `method`, `path`, `route`, `status_code`, `duration_ms`, `slow` |
| `request_failed` | исключение прошло через middleware | `request_id`, `error_type`, `error`, `duration_ms` |
| `unhandled_error` | глобальный обработчик 500 | `request_id`, `path`, `error_type`, `error` |
| `search_completed` | успешный `/search` | `abstained`, `grounding_errors`, `used_citations`, `retrieved_docs`, `latency_ms`, `p95_ms`, `empty_reason`, `stages` |
| `ask_completed` | успешный `/api/ask` | `mode`, `latency_ms`, `fragments`, `used_citations`, `abstained`, `empty_reason`, `stages` |
| `answer_empty` | `/search` или `/api/ask` без цитат | `path`, `reason`, `stages`, счётчики контекста |
| `ask_failed` | ошибка Web UI-поиска | `mode`, `error_type`, `error` |

`request_id` берётся из заголовка `X-Request-ID`, если он безопасен
(`[A-Za-z0-9._-]{1,64}`), иначе генерируется. Сервер возвращает его в
ответе, поэтому запрос можно связать с логом.

`/health` и `/metrics` не пишут `request_completed` на уровне `INFO`,
чтобы периодический опрос не засорял лог; медленные запросы к ним всё
равно попадают в `WARNING`.

## 3. Метрики

`Metrics` — потокобезопасные счётчики в памяти процесса:

- `requests_total` — все HTTP-запросы;
- `errors_total` — ответы 4xx/5xx, отдельно `client_errors_total` и
  `server_errors_total`;
- `slow_requests_total` — запросы медленнее `SLOW_REQUEST_MS`
  (по умолчанию 5000 мс);
- `timeouts_total` — таймауты, распознанные по `TimeoutError` и именам
  классов `httpx`/`requests` (`is_timeout_error`);
- `empty_answers_total` и `empty_answers_by_reason` — пустые ответы по
  причинам;
- `latency` — `count`, `avg_ms`, `p50_ms`, `p95_ms`, `max_ms` по
  последним 500 запросам;
- `routes` — запросы/ошибки/среднее/максимум по каждому маршруту;
- `stages` — `count`, `avg_ms`, `max_ms` по этапам.

Per-stage timings:

| Этап | Где измеряется | Что означает |
|---|---|---|
| `search.rag_qa` | `/search` | вызов `HippoRAG.rag_qa` |
| `search.build_response` | `/search` | сборка цитат и ответа |
| `search.total` | `/search` | полное время обработки |
| `ask.retrieval` | `/api/ask` | поиск фрагментов выбранным режимом |
| `ask.answer` | `/api/ask` | LLM, парсинг цитат и repair-retry |
| `ask.total` | `/api/ask` | полное время endpoint |

Метрики сбрасываются при рестарте процесса. Это осознанное ограничение:
внешний Prometheus/OTLP для локального приложения не требуется.

## 4. Эндпоинты

### `GET /health`

Дополнен полями:

- `llm_model` — `llm_name` из `index_manifest.json` (fallback — конфиг);
- `embedding_model` — `embedding_model` из манифеста (fallback — конфиг);
- `index_created_at` — время создания индекса;
- `metrics` — компактная сводка `MetricsSummary` (счётчики, latency,
  маршруты и этапы).

Существующие поля (`status`, `uptime`, `chunks`, `facts`, `auth_*`,
`index_version`, `index_state`, `index_operation`, `documents`) сохранены.

### `GET /metrics`

Полный снимок `MetricsResponse`: те же поля плюс `uptime_seconds` и
`timestamp`. Требует `X-API-Key`, как и остальные данные.

### `GET /ui/dashboard`

Публичная HTML-страница (данные на ней — только после ввода ключа).
Показывает:

- здоровье, uptime, версию/состояние индекса, модели, документы;
- запросы, ошибки, медленные запросы, таймауты, пустые ответы;
- p50/p95/max latency;
- таблицы маршрутов и этапов;
- причины пустых ответов.

Страница обновляется каждые 5 секунд; ключ хранится в `sessionStorage`
вкладки и не пишется на сервер.

## 5. Как разбирать медленный и пустой запрос

1. Найти `request_completed` по `request_id` (заголовок ответа
   `X-Request-ID`).
2. Если `slow=true`, открыть `search_completed`/`ask_completed` с тем же
   `request_id` и посмотреть `stages`.
3. Если `empty_reason` не `null`, открыть `answer_empty`: там причина,
   число найденных фрагментов и этапы.
4. Сверить с `/metrics`: `routes` покажет, какой маршрут накапливает
   задержку, `stages` — какой этап, `empty_answers_by_reason` — долю
   отказов по причинам.

Примеры причин:

- `no_fragments` — поиск ничего не нашёл, LLM не вызывался;
- `llm_unavailable_or_empty` — LM Studio недоступен или вернул пустую
  строку;
- `no_inline_citations` — модель ответила без валидных `[n]`, сработал
  контракт заземления;
- `no_citations` — ответ не содержит использованных цитат;
- `abstained` — явный отказ без более конкретной причины.

## 6. Проверка

Добавлены 11 тестов в `tests/unit/test_observability.py`:

- счётчики запросов/ошибок/медленных/таймаутов/пустых ответов;
- агрегация этапов;
- классификация причин пустого ответа;
- распознавание таймаутов клиентов;
- JSON-событие с `request_id`;
- генерация и эхо `X-Request-ID`;
- поля `/health` (`llm_model`, `embedding_model`, `index_created_at`,
  `metrics`);
- `/metrics`;
- события и этапы `/search`;
- события и этапы `/api/ask`;
- маршрут и статика дашборда.

Команды:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_observability.py -q
# 11 passed

.\.venv\Scripts\python.exe -m pytest tests/unit -q
# 541 passed, 1 skipped (2026-11-06, Python 3.11.9)
```

Дополнительно проверено через `TestClient`: `/health` отдаёт
`llm_model`, `embedding_model`, `index_created_at` и `metrics`;
`/metrics` — `requests_total`, `routes`, `stages`; `/ui/dashboard` —
`200 text/html`; ответы содержат `X-Request-ID`.

## 7. Ограничения

- Метрики только в памяти: рестарт обнуляет счётчики.
- Нет экспорта в Prometheus/OTLP и нет персистентной истории.
- `timeouts_total` учитывает таймауты, которые дошли до API. Внутренний
  `lmstudio_chat` по-прежнему превращает недоступность LLM в пустой
  ответ; такой случай виден как `llm_unavailable_or_empty`, а не как
  `timeout`.
- Дашборд не требует ключ для загрузки HTML, но все данные (`/health`,
  `/metrics`) закрыты guard-ом.
- Вопросы, ответы и тексты фрагментов в события не пишутся — только
  счётчики и причины.
