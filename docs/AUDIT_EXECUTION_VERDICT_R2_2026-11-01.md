# Проверка раунда 2 сводного плана аудита

**Дата проверки:** 2026-11-01
**Проверяющий:** Mavis (mavis)
**Проверяемый документ:** `docs/AUDIT_SYNTHESIS_2026-11-01.md` (строка «Раунд 2» в разделе 10)
**Чек-лист:** приложение раздела 9 + `pytest tests/unit` + воспроизведение D-1…D-8 из вердикта по выполнению

---

## 1. Вердикт

## ✅ РАУНД 2 ВЫПОЛНЕН — ПРИНЯТЬ

Все восемь расхождений (D-1…D-8) закрыты **по существу, а не формально**. Каждый проверен по коду и воспроизведён эмпирически — я не полагался на журнал.

Инструментальные проверки, которые в прошлый раз падали, теперь зелёные:

| Проверка | Было | Стало |
|----------|------|-------|
| `pytest tests/unit` | 62 passed | **62 passed** ✅ |
| `black --check src/ tests/` | exit 1, 57 файлов | **exit 0, 107 файлов unchanged** ✅ |
| `isort --check-only src/ tests/` | exit 1, Traceback | **exit 0** ✅ |
| `ruff check src tests` | не было в CI | **All checks passed** ✅ |
| `ruff check src --select F` (моя строгая проверка) | 19 замечаний | **All checks passed** ✅ |
| `compileall src/hipporag` | exit 0 | **exit 0** ✅ |
| mypy `python_version` | `"3.10"` | **`"3.12"`** ✅ |
| mypy — запуск | падал на `numpy` stub | **запускается**, exit 1, 208 замечаний в 34 файлах (не блокирует CI — advisory) |

Осталась **одна мелкая правка** (R-1 ниже) — она не влияет на код, только на pre-commit-хук, и делается за минуту.

**Отдельно отмечаю: заявление про D-7 принято и перенесено в критерии приёмки.** Исполнитель верно переформулировал смысл переиндексации: индекс функционально неполон (1 факт вместо тысяч), а не «грязные хеши». И предложенные критерии (facts > 100, synonymy-рёбра > 0, `/health.facts` ненулевой) — ровно те, что нужны.

---

## 2. Проверка D-1…D-8

### ⛔→✅ D-1. Цитата теперь несёт **все** источники фрагмента

**Было:** `get_metadata()` возвращал только первый источник → цитата указывала на `A.pdf` вместо `B.pdf`; `get_all_metadata_entries()` не вызывался в проде.

**Стало — три независимых изменения, все работают:**

1. `Citation` получил поле `sources` (`src/hipporag/utils/custom_types.py:16`) с `__post_init__`-фолбэком на `[source_file]`.
2. Новая функция `build_citation_from_metadata(entries, content)` (`src/hipporag/utils/misc_utils.py:158-176`) строит цитату из **всех** записей.
3. `HippoRAG.retrieve` вызывает её (`src/hipporag/HippoRAG.py:576-580`) — то есть потребитель, который я и указывал, переведён на новый API.

Эмпирическая проверка:

```
metadata entries      : ['A.pdf', 'B.pdf']
Citation.source_file  : A.pdf          ← первичный (первый известный)
Citation.sources      : ['A.pdf', 'B.pdf']
D-1 FIXED?            : YES
after reload sources  : ['A.pdf', 'B.pdf']   ← переживает parquet-цикл
```

**API отдаёт новые источники:** `src/hipporag/search_api/server.py:144` → `"sources": citation.sources or []`, и поле объявлено в `search_api/models.py:21-23`.

### ✅ D-2. CI починен

| Пункт | Проверка | Итог |
|-------|----------|------|
| `multi_line_mode` → `multi_line_output` | `isort --check-only` → exit 0 | ✅ |
| `black` по коду | exit 0, 107 файлов | ✅ |
| mypy `python_version` | `pyproject.toml:142` → `"3.12"` | ✅ |
| flake8 → ruff в CI | `ci-cd.yml:40` → `ruff check src/ tests/` | ✅ |
| mypy — advisory | `ci-cd.yml:44` → `continue-on-error: true` | ✅ |
| сломанный docs-job удалён | в workflow остались jobs: `lint`, `security`, `test`, `build`, `publish`; совпадений по `docs|make html|sphinx` → 0 | ✅ |
| `.pre-commit-config.yaml` в корне | `Test-Path` → True | ✅ |
| `ruff` в dev-extras | `pyproject.toml:75` → `"ruff>=0.5.0"` | ✅ |
| `[tool.ruff]` с `select`/`ignore` | `pyproject.toml:97-107` | ✅ |

Конфиг ruff осмысленный: `select = ["E9", "F", "B"]` (синтаксис, pyflakes, bugbear) и явный `ignore` стилистического долга с комментарием, что он не влияет на корректность. `extend-exclude` исключает `tests/standalone`, `archive`, `outputs`, `htmlcov`.

### ✅ D-3. `EmbeddingCache` удалён, аннотации исправлены

- Поиск `EmbeddingCache|multiprocessing\.|threading\.Lock` по `src/` → **0 совпадений**. Класс действительно удалён (`embedding_model/base.py`).
- `base.py:78,85` — аннотации теперь `"EmbeddingConfig"` вместо `"LLMConfig"` (копипаста устранена).

### ✅ D-4. Ruff F-класс = 0

Проверил **своей** строгой командой, а не только той, что в CI:

```
ruff check src --select F   →  All checks passed!   (было 19)
```

То есть закрыты не только «шумовые» F401/F841, но и настоящие F821 — модульные импорты `ChunkWithCitation` и `TYPE_CHECKING` для `Image` сделаны правильно.

### ✅ D-5. Контракт `infer` доведён до всех потребителей

`src/hipporag/utils/qa_utils.py:53-54`:

```python
# infer возвращает (content, metadata, cache_hit) — единый контракт (D-5)
response_content, _metadata, _cache_hit = llm_client.infer(messages=messages)
```

Тихого `""` больше нет.

### ✅ D-6. `.env` грузится библиотекой

`src/hipporag/__init__.py:10`:

```python
from .utils import env_config  # noqa: F401  — загрузка .env при импорте пакета (D-6)
```

Проверено:

```
env_config loaded : True
LLM_NAME env      : gemma-4-e4b
LLM_BASE_URL env  : http://localhost:1234/v1
BaseConfig()      : gemma-4-e4b | http://localhost:1234/v1
```

`import hipporag` → LM Studio из `.env`, а не `gpt-4o-mini` / `None`.

### ✅ D-7. Факт-стор теперь виден и честно репортится

- `/health` отдаёт `facts` (`server.py:108,115`).
- lifespan при `fact_count <= 1` пишет предупреждение: «Факт-стор почти пуст … графовая часть поиска вырождена … Требуется переиндексация» (`server.py:61-65`).

Живой прогон API:

```
HEALTH: 200 {"status":"healthy","version":"1.0.0","uptime":"0h 1m 27s","chunks":1422,"facts":1,...}
chunks  : 1422
entities: 6979
facts   : 1
graph   : 8401
```

Раньше `/health` отдавал `chunks: 1422` и **выглядел** здоровым при вырожденном факт-сторе. Теперь `facts: 1` видно сразу — ложная уверенность устранена.

### ✅ D-8. Тест S-12 переписан и теперь ловит исходный дефект

`tests/unit/test_stage0_fixes.py:32-38`:

```python
# D-1/D-8: цитата обязана показывать ВСЕ источники фрагмента, а не только первый —
# иначе исходный дефект «цитата указывает не на тот документ» сохраняется
citation = build_citation_from_metadata(entries, text)
assert set(citation.sources) == {"A.pdf", "B.pdf"}
assert citation.source_file == "A.pdf"  # первичный (первый известный) источник
```

Тест **упал бы** на старом поведении — ровно тот сценарий, который я предупреждал. Плюс добавлены тесты на дедуп одинаковых метаданных, на пустые метаданные и на roundtrip через parquet.

### ✅ Приёмка S-12 в приложении скорректирована

Строка 263 теперь гласит: «два PDF с одинаковым колонтитулом → **одна запись со списком источников** (вариант «б»); цитата (`Citation.sources`) называет **оба** документа» — соответствует реализации.

---

## 3. Единственная оставшаяся правка

### ⚠️ R-1. pre-commit-хук ruff упадёт на каждом коммите

`.pre-commit-config.yaml:28-32`:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.5.0
  hooks:
    - id: ruff
      args: ['--select', 'E9,F,B']      # ← CLI-флаг перекрывает ignore из pyproject
```

**Что происходит:** CLI-флаг `--select` сбрасывает выбор правил целиком, вместе с `ignore` из `pyproject.toml`. Проверено той же командой, что запускает хук:

```
ruff check src tests --select E9,F,B   →  Found 38 errors  (exit 1)

B905: 20   B904: 9   B007: 4   B027: 3   B006: 1   B008: 1
```

Все 38 — ровно те коды, которые `ignore = ["B905","B904","B027","B007","B006","B008"]` подавляет. То есть хук ругается ровно на то, что сам же проект решил игнорировать.

**Почему это важно:** хук в `pre-commit` с `autofix_commit_msg` и `autofix_prs: true` будет валить **каждый** коммит, пока флаг не убрать. Это не «косметика» — это заблокированный рабочий процесс.

**Правка (1 минута):** удалить строку `args: ['--select', 'E9,F,B']` — `select` уже задан в `[tool.ruff.lint]` в `pyproject.toml`, и CI-шаг `ruff check src/ tests/` работает именно так (exit 0). Либо, если флаг нужен для ясности, добавить рядом `--ignore` с тем же списком.

*Ирония: в журнале раунда 2 сказано «CI: flake8 → `ruff check` (select в pyproject; CLI `--select` перекрывал config-ignore)» — проблема была осознана и исправлена в CI, но тот же флаг остался в pre-commit-хуке.*

### Мелочи в документации (на 5 минут)

| ID | Что | Где |
|----|-----|-----|
| R-2 | Пункт 0.8 в разделе 6 всё ещё требует «2 записи с разными `source_file`», а в приложении (стр. 263) уже «одна запись со списком источников» — два разных критерия в одном документе | `docs/AUDIT_SYNTHESIS_2026-11-01.md:164` |
| R-3 | Приёмка S-30 (стр. 264) говорит «чанков > 0»; теперь `/health` отдаёт ещё и `facts`, и при `facts <= 1` граф вырожден — критерий стоит дополнить | `docs/AUDIT_SYNTHESIS_2026-11-01.md:264` |

---

## 4. Что осталось за рамками Раунда 2 (не дефект, а следующий шаг)

**Переиндексация — обязательный шаг, а не опция.** Текущий индекс функционально неполон: `facts = 1` при `chunks = 1422` и `entities = 6979`. Графовая часть HippoRAG (PPR по фактам) на таком индексе вырождена, поиск опирается преимущественно на dense-retrieval.

Критерии приёмки, которые я считаю корректными (и предлагаю зафиксировать):

1. `facts > 100` в новом `vdb_fact.parquet` (не 1);
2. synonymy-рёбра > 0 в графе (проверка: `graph.es()` содержит рёбра с `source`/`target` из entity-узлов, а не только entity→passage);
3. `/health` → `facts` ненулевой **и** без предупреждения «Факт-стор почти пуст» в логе lifespan;
4. `entities` и `facts` сопоставимы по порядку величины с `chunks` (как сейчас 6979 / 1 / 1422 — перекос в 7000x).

Также после переиндексации имеет смысл перепроверить S-15/S-16 в новых данных: «ё» и единая нормализация меняют hash-ключи сущностей, поэтому старые 6979 entity-узлов не сопоставимы с новыми.

---

## 5. Сводка

| | |
|---|---|
| **Вердикт** | ✅ Раунд 2 выполнен и принят |
| D-1…D-8 | 8 из 8 закрыты по существу, проверены эмпирически |
| Инструменты | black / isort / ruff / pytest / compileall — все зелёные; mypy переведён на 3.12 и сделан advisory (запускается, но находит 208 замечаний — потому и advisory) |
| Осталось | **R-1** (1 минута) + R-2/R-3 (документация, 5 минут) |
| Блокер | только R-1 — из-за него pre-commit валит каждый коммит |
| Следующий шаг | переиндексация + приёмка по 4 критериям из раздела 4 |

---

## 6. Приложение: воспроизведение проверок Раунда 2

```bash
# Инструменты — все должны быть зелёными
python -m pytest tests/unit -q                      # 62 passed
python -m black --check src/ tests/                  # exit 0, 107 unchanged
python -m isort --check-only src/ tests/             # exit 0
python -m ruff check src tests                       # All checks passed
python -m ruff check src --select F                  # All checks passed (было 19)
python -m compileall -q src/hipporag                 # exit 0

# R-1 — единственная оставшаяся правка
python -m ruff check src tests --select E9,F,B       # exit 1, 38 errors → хук pre-commit падает
python -m mypy src/hipporag --ignore-missing-imports # exit 1, 208 errors (advisory, CI не блокирует)

# D-6 — .env грузится библиотекой
python -c "import os, hipporag; print(os.environ.get('LLM_NAME'))"   # gemma-4-e4b

# D-7 — факт-стор виден
# поднять API → GET /health → {"chunks":1422,"facts":1}
```

---

## 7. Подтверждено чистым

- 62/62 юнит-теста, включая `-m unit` (как в CI) и `pytest tests/unit` целиком.
- `compileall` — 0 синтаксических ошибок.
- `black`/`isort` — 0 расхождений на 107 файлах.
- `ruff` — 0 замечаний в `src` и `tests` по правилам E9/F/B, включая строгую проверку `--select F`.
- Namespace: ни одного импорта `src.hipporag`; `hipporag` и `src/hipporag` — один модуль.
- `setup.py` отсутствует, `requirements.txt` → `-e .`, зависимости в `pyproject.toml` (33 шт. + extras `ru`/`offline`/`dev`).
- Отладочных вставок нет: `ipdb`/`set_trace`/`assert False`/bare `except:` → 0; `print()` → 0; `utcnow` → 0.
- `EmbeddingCache` удалён; `multiprocessing`/`threading` в `src/` не встречаются.
- Кириллица: «ё/Ё» сохранена, `normalize_entity_name` == `text_processing`, synonymy-регулярка даёт ненулевую длину для кириллицы, NER не матчит подстроки, `dataset` по умолчанию `astrology`.
- `Citation.sources` + `build_citation_from_metadata()` + проброс в API — работают, переживают parquet-цикл.
- `/health` отдаёт `chunks` и `facts`; lifespan предупреждает о вырожденном факт-сторе.
- `docs/` — 47 файлов в `archive/docs/`, 11 активных; `.gitignore` расширен; `git rm --cached` для 2 файлов `outputs/` выполнен.
- `.pre-commit-config.yaml` в корне, хуки black/isort/ruff/mypy/bandit; bandit использует `.bandit`.
