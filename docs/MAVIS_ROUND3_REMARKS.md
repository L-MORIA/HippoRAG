# Замечание Mavis к раунду 3 (аутентификация fail-closed, CORS-обрамление, Eval Smoke)

**Дата:** 2026-11-03
**Автор замечаний:** Mavis (mavis)
**Проверяемый коммит:** `HEAD` после раунда 3 (`ff41ed8`, `124c967`, `caed15e`)
**Основание:** `docs/AUDIT_FIX_VERIFICATION_R2_2026-11-03.md` (замечания N1–N4, N6, N7, D3/D5/D7, §7.1–§7.6)
**Метод:** чтение кода, 228/228 unit-тестов, black/isort/ruff, YAML-валидация

---

## 1. Вердикт

## ✅ РАУНД 3 ПРИНЯТ — 1 замечание (низкий приоритет)

Все пять замечаний раунда 2 (D3/D7 fail-closed, N2/N5 CORS-обрамление, D5 аудит 401, §7.4 docs-only PR, §7.1–§7.3 частично) закрыты корректно. Архитектура аутентификации изменилась качественно: fail-open → **fail-closed** с явным opt-out — это единственно верное решение для публичного репозитория. Порядок middleware исправлен архитектурно (CORS — внешний слой), а не костылём.

---

## 2. Что проверено и подтверждено

### 2.1. D3/D7 — Fail-closed аутентификация

**Было:** без `API_KEY` API полностью открыт (`/search`, `/ingest`, `/docs`, `/openapi.json` → 200). Единственный барьер между публичным репозиторием и открытым API.

**Стало:** `API_KEY_REQUIRED=1` по умолчанию. Без `API_KEY` сервер **не стартует** (`RuntimeError` в lifespan с инструкцией). Явный opt-out `API_KEY_REQUIRED=0` — только для локальной разработки.

```
server.py:37  API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "1")...
server.py:61  if API_KEY_REQUIRED and not API_KEY:
server.py:62      raise RuntimeError("API_KEY обязателен...")
```

Проверено: с `API_KEY` сервер стартует; без `API_KEY` и без `API_KEY_REQUIRED=0` — процесс падает с понятной ошибкой. Fail-closed для публичного репозитория — правильно.

### 2.2. N2/N5 — CORS как внешний слой (архитектурный фикс)

**Было:** origin захардкожен в 401-ответе guard (`localhost:3000`) — второй разрешённый origin (`localhost:8000`) не мог прочитать отказ. Список origin дублировался в двух местах.

**Стало:** CORS middleware регистрируется **после** guard → в Starlette последний middleware — внешний → CORS добавляет заголовки ко всем ответам, включая 401, без дублирования. Список origin — один (`API_ALLOWED_ORIGINS` из env).

```
server.py:110  @app.middleware("http")     ← guard (внутренний)
server.py:155  app.add_middleware(CORSMiddleware, ...)  ← CORS (внешний)
```

Проверено: 401-ответ содержит `access-control-allow-origin` без хардкода в guard.

### 2.3. D5 — Аудит отказов авторизации

`auth_failures` — счётчик с момента запуска; `logger.warning` на каждый 401 (path, origin, накопленный счётчик); поля `auth_required`/`auth_failures` в `/health`. Проверено: поле присутствует в ответе.

### 2.4. §7.4 — paths-ignore убран из pull_request

Docs-only PR получают статус-checks и могут пройти branch protection. Для `push` в main `paths-ignore` сохранён — осознанно: владелец пушит напрямую, docs-only push не нуждается в CI.

### 2.5. H1-5 (частично) — Eval Smoke job

Новый job `eval-smoke` в CI (10 вопросов, без LLM и индекса). Покрывает: контракт датасета (парсинг, схема, уникальность qid, статус verified), корректность метрик на синтетической выдаче, согласованность разметки. Не требует индекса, моделей и сети.

### 2.6. Прочее

- `.env.example`: честное описание fail-closed/fail-open с инструкцией для выставления наружу
- `HOW_TO_RUN.md`: раздел «Аутентификация API» расширен
- `MAIN_SPLIT_PLAN.md` §7: таблица восстановлена (строка 2 целая, строка 3 удалена — recall-пояснение перенесено в ROADMAP_REVISION)
- 228/228 тестов, black/isort/ruff — все зелёные

---

## 3. Единственное замечание

### ⚠️ Z-1. Нет unit-теста, фиксирующего порядок middleware

**Суть.** Порядок «CORS — внешний, guard — внутренний» реализован корректно, но не зафиксирован тестом. Если кто-то переставит `add_middleware` — CORS перестанет оборачивать 401-ответы, и браузерный клиент снова увидит «CORS error» вместо «401: нужен ключ». Функциональные тесты (`test_search_api_auth.py`) проверяют эффект, но не источник: они пройдут, пока сервер запущен с одним origin, и сломаются при втором — только если тест проверяет именно CORS-заголовки на 401.

**Рекомендация (10 мин).** Добавить в `test_search_api_auth.py` тест:

```python
def test_401_has_cors_allow_origin(server_module):
    """Если порядок middleware изменится, CORS перестанет обрамлять 401."""
    from fastapi.testclient import TestClient
    client = TestClient(server_module.app)
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 401
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
```

Тест падает при (а) перестановке middleware, (б) удалении CORS из allow_origins, (в) переименовании заголовка. Дёшево и покрывает конкретный риск.

**Не блокирует:** это защита от будущих регрессий, не от текущих.

---

## 4. Что осталось открытым (по всем раундам, сводка)

| Замечание | Статус | Приоритет |
|-----------|--------|-----------|
| **D3** fail-closed | ✅ закрыт | — |
| **D5** аудит 401 + счётчик | ✅ закрыт | — |
| **N1** не-ASCII ключ → 500 | ✅ закрыт | — |
| **N2/N5** CORS-обрамление 401 | ✅ закрыт | — |
| **D6** документация | ✅ закрыт | — |
| **D7** публичный репо + fail-open | ✅ закрыт (fail-closed устраняет сценарий) | — |
| **§7.1** recall@5 (27.5% vs 60%) | ⚠️ пояснение в ROADMAP_REVISION; агрегированный отчёт в репозитории — не сделано | P2 |
| **§7.2** §14 про publish-job | ⬜ не исправлено (косметика текста) | P2 |
| **§7.3** устаревшие счётчики | ⬜ не исправлено (косметика) | P2 |
| **§7.4** paths-ignore для push | ✅ осознанно сохранён для push, убран для PR | — |
| **§7.6** артефакты замера в outputs/ | ⬜ агрегированный отчёт в репозитории — не сделано | P2 |
| **§7.7** приватность | ⬜ в допустимых пределах | — |
| **H1-2** рекомендация режимов | ⬜ не начата | P1 |
| **H1-3** performance-бюджет | ⬜ не начат | P1 |
| **H1-4** ablation графа | ⬜ не начат | P2 |
| **H1-5** eval smoke | ⚠️ частично: job в CI есть, retrieval-прогон на фиксированном индексе — не воспроизводим в CI (индекс вне репо) | P1 |
| **Режим 12** дисплей | ✅ исправлен | — |
| **Режим 18** верификация | ✅ EXACT с базлайном | — |

---

## 5. Рекомендуемый порядок (после раунда 3)

1. **Z-1** (10 мин) — тест порядка middleware
2. **H1-2** (0,5 дня) — рекомендация 1–2 режимов по данным §8a: default = режим 5 (BM25+CE-ru, 8/8 адекватности, 2 с), экспертный = режим 1 (97.5% recall@5), LLM-ответ = режим 21 (adequacy 0.812)
3. **H1-3** (2–3 дня) — performance-бюджет
4. **H1-4** (2–3 дня) — ablation графа
5. **§7.6** (30 мин) — агрегированный отчёт замера рядом с eval/dataset/
6. **§7.2–§7.3** (20 мин) — косметика плана

---

## 6. Приложение: команды проверки

```bash
# Тесты
python -m pytest tests/unit -m unit -q                     # 228 passed
python -m pytest tests/unit/test_search_api_auth.py -q     # 15 passed
python -m pytest tests/unit/test_eval_smoke_contract.py -q # 9 passed

# Формат и линт
python -m black --check src/astro_app/ src/hipporag/search_api/ main.py
python -m isort --check-only src/astro_app/ main.py
python -m ruff check src/astro_app/ src/hipporag/search_api/ main.py

# Fail-closed: сервер не стартует без ключа
API_KEY_REQUIRED=1 python -c "
from astro_app.cli import main"  # ← должен работать (main не поднимает сервер)
# но:
# python -c "
# import os; os.environ['API_KEY_REQUIRED']='1'
# import hipporag.search_api.server as srv
# importlib.reload(srv)
# "  # ← RuntimeError: API_KEY обязателен
```
