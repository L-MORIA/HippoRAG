# Полный аудит проекта HippoRAG (Astrology RAG)

**Дата:** 2026-11-01
**Объём:** исходный код пакета `src/hipporag` (~8 000 строк), корневые скрипты (`main.py`, `menu.py` и др.), конфигурация сборки, CI, тесты, зависимости.

---

## 1. Резюме

Проект — глубоко переработанный форк HippoRAG 2 под задачу поиска по русскоязычным PDF (астрология/фэншуй) с локальным LLM через LM Studio. Ядро работоспособно в текущем «боевом» сценарии (онлайн-режим + локальный LLM + sentence-transformers), но аудит выявил **7 критических дефектов, которые ломают целые режимы работы** (offline-индексация, StandardRAG, удалённый OpenAI/Azure, API-эндпоинт загрузки файлов, кэш LLM), плюс системные проблемы русскоязычной обработки («ё», synonymy-рёбра, двойная нормализация сущностей), три несогласованных списка зависимостей и почти полное отсутствие тестов на ядро.

**Ключевые цифры:**

| Показатель | Значение |
|---|---|
| Критических ошибок (P0) | 7 |
| Высоких (P1) | 8 |
| Средних (P2) | 9 |
| Низких / гигиена (P3) | 6 |
| Замечаний pyflakes | 176 |
| Покрытие ядра тестами | ~0 % (42 теста только по env_config/exceptions) |

---

## 2. Что проверялось

- Полное чтение: `HippoRAG.py`, `StandardRAG.py`, `embedding_store.py`, `rerank.py`, `llm/*` (openai_gpt, base, __init__), `embedding_model/*` (base, OpenAI, Transformers, __init__), `information_extraction/*` (openie_openai, ner_russian), `prompts/*` (manager, templates), `utils/*` (config, env, misc, llm, embed, logging, lm_studio_manager), `evaluation/*`, `pdf_ingestion/*`, `search_api/*`, `exceptions.py`.
- Обзор: `main.py` (3139 строк — структура), корневые скрипты, `scripts/`, CI-workflow, `.gitignore`, setup.py/pyproject.toml/requirements.txt, `.env` (имена переменных).
- Инструменты: `compileall` (0 синтаксических ошибок), `pyflakes` (176 замечаний), `pytest tests/unit tests/integration` (42 passed), проверка импортируемости зависимостей, ручное исполнение подозрительных регулярных выражений.

**Не проверялось глубоко:** `reproduce/`, `eval/`, скрипты `tests/standalone/*` (40+ ручных скриптов), пути Bedrock/Transformers LLM, фактическое качество поиска (нужен запуск с моделями).

---

## 3. Критические ошибки (P0) — блокируют целые режимы

### P0-1. Отладчик в production-коде: `ipdb.set_trace()` в конструкторе StandardRAG
**Файл:** `src/hipporag/StandardRAG.py:101-103`

```python
import ipdb
ipdb.set_trace()
```

Класс **всегда** останавливается в интерактивном отладчике при создании. Без установленного `ipdb` — `ImportError`. Класс непригоден в принципе.
Аналогичная точка: `src/hipporag/embedding_model/OpenAI.py:111-114` — bare `except:` с `ipdb.set_trace()`: любая ошибка API эмбеддингов уводит процесс в отладчик вместо повтора/ошибки (bare except к тому же ловит KeyboardInterrupt/SystemExit).

### P0-2. Offline-индексация всегда падает: `assert False` в `pre_openie`
**Файл:** `src/hipporag/HippoRAG.py:249`

```python
assert False, logger.info("Done with OpenIE, run online indexing ...")
```

`pre_openie` вызывается из `index()` при `openie_mode == "offline"` и **всегда** бросает `AssertionError`. Режим сломан и другими способами: при offline `self.embedding_model = None` (HippoRAG.py:152-159), но `EmbeddingStore` и `get_query_embeddings` требуют модель. При запуске с `python -O` assert исчезает — поведение меняется непредсказуемо.

### P0-3. Кэш LLM не подключён; контракт возврата `infer` разный для локального и удалённого пути
**Файл:** `src/hipporag/llm/openai_gpt.py`

Декоратор `cache_response` (строка 25) определён, но **не применён ни к одному методу** (проверено grep по всему проекту). Следствия:

1. SQLite-кэш (`llm_cache/*.sqlite`) — мёртвый код, каждый запрос повторно бьёт в LLM.
2. Удалённый путь (OpenAI/Azure) возвращает **2 значения** `(message, metadata)` (строка 239), а локальный путь — **3 значения** `(message, metadata, False)` (строка 300). Все вызывающие распаковывают 3: `HippoRAG.qa` (HippoRAG.py:973), `OpenIE.ner`/`triple_extraction` (openie_openai.py:65, 135). **Работа через облачный OpenAI/Azure падает с ValueError на первом же вызове.**

### P0-4. Вызов несуществующего метода в StandardRAG
**Файл:** `src/hipporag/StandardRAG.py:285`

`queries = self.retrieve_dpr(queries=queries)` — у класса нет метода `retrieve_dpr` (есть `retrieve`). `rag_qa` со строковыми запросами без `gold_docs` → `AttributeError`.

### P0-5. Сломанный эндпоинт `/upload-and-ingest` в API
**Файл:** `src/hipporag/search_api/server.py:182-245`

Вся логика индексации (`def index_task`, `background_tasks.add_task`, `return {...})` находится **внутри блока `except` после `raise`** — недостижимый код. Успешный путь (файлы сохранены) доходит до конца функции и возвращает `None`: файлы сохраняются на диск, но **никогда не индексируются**, клиент получает пустой ответ. Побочно: `upload_dir.rmdir()` удалил бы только пустую папку.

### P0-6. Используемые зависимости отсутствуют во всех файлах зависимостей
Проверено импортом и grep:

| Пакет | Кто использует | Отсутствует в |
|---|---|---|
| `pandas` | embedding_store.py (parquet-хранилище — ядро!) | requirements.txt, pyproject.toml, setup.py |
| `pyarrow` | parquet-движок pandas | везде |
| `requests` | openai_gpt._infer_local_llm, lm_studio_manager | везде |
| `rank_bm25` | main.py (строка 6) | везде |
| `spacy` + `ru_core_news_sm` | information_extraction/ner_russian.py + 4 скрипта | везде |

Чистая установка из requirements.txt ломает ядро (embedding_store). Дополнительно: **три несогласованных источника зависимостей** (setup.py, pyproject.toml, requirements.txt) с разными версиями одних и тех же пакетов (например, boto3>=1.26.0 vs >=1.34.0), `tenacity`/`python-dotenv`/`pydantic-settings` есть в setup.py, но отсутствуют в pyproject.toml.

### P0-7. `pydantic-settings` не объявлен, fallback в env_config несовместим с pydantic v2
**Файл:** `src/hipporag/utils/env_config.py:14-17`

```python
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings   # в pydantic v2 этого класса больше нет → ImportError
```

В requirements.txt/pyproject.toml `pydantic-settings` не указан (только в setup.py). При чистой установке `env_config` падает на импорте. Также используются удалённые в v2 конструкции: `@validator`, `class Config`, `.dict()`.

---

## 4. Высокие ошибки (P1) — некорректное поведение и данные

### P1-1. Synonymy-рёбра не строятся для русского текста
**Файл:** `src/hipporag/HippoRAG.py:1145`

```python
if len(re.sub("[^A-Za-z0-9]", "", entity)) > 2:
```

Регулярка оставляет только латиницу/цифры: для русской сущности длина всегда 0 → ветка не выполняется. Для данного корпуса synonymy-рёбра (одна из двух ключевых идей HippoRAG 2) **отключены полностью**. Унаследовано из upstream (там было `[^a-zA-Z0-9 ]`), но для русскоязычного проекта это критично.

### P1-2. Двойной буст в `hybrid_retrieval`
**Файл:** `src/hipporag/HippoRAG.py:1774-1775`

```python
boosted_scores[i] = boosted_scores[i] * boost_multiplier + exact_match_bonus
boosted_scores[i] *= boost_multiplier        # множитель применяется второй раз
```

Множитель применяется дважды (до ~2.25x вместо 1.5x), бонус тоже умножается. Комментарий («Add 1.0 to score») не соответствует коду.

### P1-3. Буква «ё» искажается в `text_processing`
**Файл:** `src/hipporag/utils/misc_utils.py:99`

```python
re.sub("[^A-Za-zА-Яа-я0-9 ]", " ", text.lower())
```

«ё/Ё» (U+0451/U+0401) не входит в диапазон А-Яа-я: «ёжик» → «жик», «чёрный» → «ч рный». Все факты/сущности с «ё» попадают в граф искажёнными (худшее качество эмбеддингов и генерации ответов с цитатами). Forк добавил А-Яа-я к upstream-регулярке, но забыл «ё».

### P1-4. Две несогласованные нормализации сущностей → дубли узлов графа
- `normalize_entity_name()` (misc_utils.py:40): lower + обрезка краевой пунктуации + схлопывание `\s+`.
- `text_processing()` (misc_utils.py:94): замена запрещённых символов на пробелы **внутри** строки, схлопывания пробелов нет.

Пример: «рак (знак)» → `normalize` = «рак знак», `text_processing` = «рак  знак» (двойной пробел). NER-сущности идут в entity-хранилище в normalize-виде, а узлы графа/факты — в text_processing-виде → одна сущность размножается на разные hash-ключи, PPR размывает вес.

### P1-5. `assert` как управление потоком в retrieval
**Файлы:** `src/hipporag/HippoRAG.py:1943-1945` (`assert sum(node_weights) > 0`), `HippoRAG.py:1828-1829`, `embedding_model/__init__.py:38` (`assert False, Unknown embedding model`).

Если факты найдены, но их сущности отсутствуют в графе — retrieval падает AssertionError вместо предусмотренного fallback на DPR ( ветка «No facts found after reranking» срабатывает только при пустом `top_k_facts`).

### P1-6. Два разных класса `QuerySolution`
- `utils/misc_utils.py:71` — с `gold_answers`/`gold_docs`/`to_dict()`, без `citations`.
- `utils/custom_types.py:17` — с `citations`, без `gold_answers`.

`HippoRAG.py` импортирует custom_types-версию (после star-import, строка 41 — она побеждает), `rag_qa` присваивает `q.gold_answers` как посторонний атрибут (dataclass это допускает — работает молча). `StandardRAG` использует misc_utils-версию. Одноимённые классы с разными полями — источник трудноуловимых багов.

### P1-7. `run_ppr` при пустом маппинге вершин возвращает пустоту
**Файл:** `src/hipporag/HippoRAG.py:2041-2042`

`vertices=range(len(self.node_name_to_vertex_idx))` — fallback в `prepare_retrieval_objects` (строка 1521) при ошибке маппинга ставит `{}`, после чего PPR молча возвращает пустой результат (тихая деградация вместо ошибки). При расхождении числа вершин графа и маппинга — смещённые вероятности.

### P1-8. Конфигурация: Literal-типы не соответствуют фактическим значениям
**Файлы:** `src/hipporag/utils/config_utils.py:115,196` и `HippoRAG.py:146-147`

- `openie_mode: Literal["offline", "online"]`, но код проверяет `"Transformers-offline"`; при любом третьем значении `self.openie` просто не создаётся (AttributeError позже, непонятно где).
- `dataset: Literal[...]` не содержит `"astrology"`, хотя шаблон `rag_qa_astrology` зарегистрирован и используется (спасает только runtime-проверка `is_template_name_valid`).
- `__post_init__` читает из env только `LLM_NAME/LLM_BASE_URL/SAVE_DIR` — `EMBEDDING_*` не читаются, в отличие от декларируемой в README «env-first» конфигурации.

---

## 5. Средние проблемы (P2)

1. **API-сервер** (`search_api/server.py`): конфигурация зашита в код (строки 58-61: `outputs/astrology_test`, `gemma-4-e4b`, английская `all-MiniLM-L6-v2` вместо используемой в проекте мультиязычной); фиктивная статистика `chunks_extracted = len(valid_files) * 10` (строка 166); тексты внутренних исключений отдаются клиенту; deprecated `@app.on_event` (строка 51); нет лимита размера загрузки (DoS) и аутентификации; версии API «1.0.0» не совпадают с версией пакета.
2. **Retrieval без графа работает «случайно»**: `get_fact_scores`/`dense_passage_retrieval` вызывают `batch_encode(query)` со строкой вместо списка — `OpenAIEmbeddingModel` конвертирует сам (OpenAI.py:86-87), `TransformersEmbeddingModel` — нет (код работает из-за особенности `SentenceTransformer.encode`).
3. **Мёртвый код**: `synonym_candidates` и `num_synonym_triple` (HippoRAG.py:1136-1165); `all_cache_hit` (HippoRAG.py:973); `EmbeddingCache` (embedding_model/base.py:220, не используется); `dynamic_retry` (llm_utils.py:101 — ссылается на несуществующие атрибуты конфига, упал бы при первом использовании); `sync_render` (prompt_template_manager.py:153 — `asyncio.run()` над не-корутиной, упал бы); `make_cache_embed` с опечаткой `"promps"` в ключе кэша (base.py:114) и без имени модели в ключе.
4. **OpenIE-парсинг хрупкий** (information_extraction/openie_openai.py:33-41, 109-117): regex-based JSON-парсинг ломается, если `]` встречается внутри текста сущности; ретрай-подсказка `RETRY_JSON_STRICT.replace("\\[...\\]", ...)` ничего не заменяет (в строке нет экранированных скобок) — модель не получает корректный пример формата `[[...]]` на повторной попытке.
5. **rerank.py**: `print()` вместо logger (строки 100, 144, 159); при ошибке LLM-фильтра молча используются первые N фактов в исходном порядке (снижение качества без лога уровня warning).
6. **EmbeddingStore** (`embedding_store.py`): нет потокобезопасности — актуально для FastAPI-сервера; `assert` при загрузке parquet (строка 129) вместо обработки битого файла; непоследовательные возвращаемые значения `insert_strings` (None / {}); неиспользуемый параметр `dtype` в `get_rows`.
7. **Azure**: `azure_endpoint.split("api-version=")[1]` → IndexError без внятного сообщения при неверном URL (openai_gpt.py:185, embedding_model/OpenAI.py:42).
8. **Хардкоды**: модель `"gemma-4-e4b"` в openai_gpt.py:252 и lm_studio_manager.py:193; `top_p=0.9`, `temperature` fallback 0.7 в локальном LLM-пути не берутся из конфига; `max_workers=15` в openie_openai.py:199.
9. **Пакет**: `src/hipporag/__init__.py` безусловно импортирует `search_api.app` — любой `import hipporag` тянет FastAPI/uvicorn; версия в `__init__` — «2.0.0», в setup.py/pyproject — «2.0.0-alpha.4»; URL проекта в pyproject.toml указывают на чужой репозиторий (`anomalyco/hipporag`).

---

## 6. Низкие проблемы и гигиена (P3)

1. **176 замечаний pyflakes**: неиспользуемые импорты (в т.ч. тяжёлый `transformers.HfArgumentParser` на уровне модуля в HippoRAG.py и StandardRAG.py), дубли `import numpy as np` / `defaultdict` (HippoRAG.py:7/15, 9/16), f-строки без плейсхолдеров, star-import (`from .utils.misc_utils import *`) со всеми вытекающими «may be undefined».
2. **Репозиторий**: в git остались 2 файла `outputs/musique/*.json` (статус D — нужно подтвердить удаление); `htmlcov/`, `.mypy_cache/`, `.pytest_cache/`, `models/`, `archive/`, `src/hipporag.egg-info/` не добавлены в `.gitignore`; **~40 файлов (scripts/, eval/, search_api, pdf_ingestion, utils/env_config, tests/unit и др.) не закоммичены** — большой объём рабочего кода вне контроля версий.
3. **docs/**: 60+ файлов-отчётов, многие дублируют друг друга (FINAL_REPORT, FINAL_SOLUTION_REPORT, COMPLETE_SOLUTION_REPORT, README_FINAL…). Нужна консолидация.
4. **Тесты**: 42 теста покрывают только env_config и exceptions; ядро (индексация, хранилище, PPR, парсинг OpenIE) — 0 %; `tests/integration/` пуст; 40+ standalone-скриптов в tests/ не являются тестами pytest.
5. **CI** (`.github/workflows/ci-cd.yml`): запускает black/isort/flake8/mypy — при 176 замечаниях pyflakes и текущем формате кода lint-джоб будет красным; flake8 не читает `[tool.flake8]` из pyproject.toml (нужен `.flake8` или setup.cfg); установка `pip install -e ".[dev]"` упрётся в неполные зависимости (см. P0-6).
6. **Конфигурация сборки**: setup.py и pyproject.toml дублируют метаданные с разными списками зависимостей (setuptools объединяет их с union — итоговые зависимости непредсказуемы); в `TransformersEmbeddingModel` аннотации `-> None` у возвращающих значение методов; `logging_utils` — 60 строк закомментированного кода.

---

## 7. Безопасность

- ✅ `.env` в `.gitignore`, в git не трекается; секретов в репозитории нет (в `.env` только `LLM_NAME`, `LLM_BASE_URL`).
- ⚠️ Мутация глобального `os.environ["OPENAI_API_KEY"] = "sk-dummy"` в двух местах (llm/__init__.py:19, openai_gpt.py:150) — лучше передавать ключ клиенту явно.
- ⚠️ API без аутентификации, детальные ошибки возвращаются клиенту, приём файлов без лимита размера; CORS ограничен localhost (приемлемо для локального инструмента, недопустимо для внешнего хоста).
- ✅ Path traversal в `/upload-and-ingest` обработан (`PurePosixPath(...).name`).

---

## 8. Архитектурные наблюдения

1. **Две параллельные системы конфигурации**: `BaseConfig` (dataclass + env в __post_init__) и `HippoRAGConfig` (env_config.py, pydantic-settings) — не связаны между собой, разные префиксы env (`LLM_NAME` vs `LLM_MODEL`). Нужен выбор одной.
2. **main.py — монолит 3139 строк** с ~40 поисковыми стратегиями (BM25, ColBERT, HyDE, Self-RAG, E5, RRF, DAT, LLM-фильтры…) и доменными словарями синонимов в коде. Это исследовательская песочница: её стоит разрезать на модули `search/` и вынести словари в данные.
3. **Промпты англоязычные** (ner.py, triple_extraction.py — one-shot примеры про Radio City/Индия) при русскоязычном корпусе: качество NER через локальную модель ограничено; проект компенсирует это словарным RussianNER в скриптах. Локализация промптов — прямой путь к качеству.
4. **Кириллица в парсере synonymy/normalization** — системная тема: весь пайплайн писался под латиницу. Рекомендуется единый модуль `text_norm.py` с одной функцией нормализации, которую используют все компоненты.

---

## 9. План исправлений

### Этап 0 — фиксация текущего состояния (0,5 часа)
- Закоммитить все несмерженные файлы (scripts/, eval/, search_api, pdf_ingestion, tests/unit, utils/env_config и др.) в отдельную ветку.
- Дополнить `.gitignore`: `htmlcov/`, `.mypy_cache/`, `.pytest_cache/`, `models/`, `archive/`, `*.egg-info/`, `outputs/`, `temp_uploads/`; выполнить `git rm --cached` для двух файлов `outputs/musique/*.json`.

### Этап 1 — критические дефекты P0 (0,5–1 день)
1. Удалить `import ipdb; ipdb.set_trace()` из StandardRAG.py:101-103 и OpenAI.py:111-114; в OpenAI.py заменить bare `except:` на `except Exception` с retry/логом.
2. HippoRAG.py:249 — убрать `assert False`; для offline-режима либо реализовать полный цикл (сохранение OpenIE → отдельная команда построения графа), либо честно отключить режим с понятным `raise NotImplementedError`.
3. openai_gpt.py: применить `@cache_response` к `infer` **или** явно отказаться от кэша и привести оба пути (локальный/удалённый) к единому контракту возврата `(message, metadata, cache_hit)`. Прогнать grep по вызывающим (`HippoRAG.qa`, `OpenIE.ner`, `OpenIE.triple_extraction`, DSPyFilter) на соответствие контракту.
4. StandardRAG.py:285 — заменить `self.retrieve_dpr(...)` на `self.retrieve(...)`; удалить мёртвый unguarded-импорт VLLMOfflineOpenIE (строка 24).
5. server.py `/upload-and-ingest` — вынести логику индексации из `except`, вернуть корректный ответ, заменить `rmdir` на `shutil.rmtree` в `finally`.
6. Зависимости: добавить `pandas`, `pyarrow`, `requests`, `rank_bm25` в requirements.txt **и** pyproject.toml; `spacy` + модель — в отдельный extra (`[ru]`); удалить setup.py ИЛИ pyproject.toml (оставить один источник истины — рекомендую pyproject.toml); синхронизировать версии; добавить `pydantic-settings`, `tenacity`, `python-dotenv` в pyproject.
7. env_config.py — убрать нерабочий fallback (`from pydantic import BaseSettings`), перевести на `pydantic-settings` v2 (`SettingsConfigDict`, `field_validator`).

### Этап 2 — корректность данных и retrieval (1 день)
1. misc_utils.text_processing — включить «ёЁ» (`[^A-Za-zА-Яа-яЁё0-9 ]`) + схлопывание пробелов.
2. HippoRAG.py:1145 — заменить регулярку на Unicode-вариант (например, `len(re.sub(r"[\W_]+", "", entity, flags=re.UNICODE)) > 2`).
3. Унифицировать нормализацию сущностей: один модуль, одна функция; перестроить индекс после изменения (старые hash-ключи инвалидируются).
4. hybrid_retrieval — убрать второе умножение (HippoRAG.py:1775), привести к задокументированной формуле.
5. Заменить assert'ы в retrieval-пути на graceful fallback к DPR + warning (HippoRAG.py:1943, 1828; embedding_model/__init__.py:38 → `ValueError`).
6. run_ppr — при пустом/рассинхронизированном маппинге вершин бросать осмысленное исключение вместо тихого пустого результата.

### Этап 3 — API и конфигурация (0,5 дня)
1. server.py: конфигурацию из env/файла (не хардкод), `lifespan` вместо `on_event`, честная статистика индексации, лимит размера загружаемых файлов, обобщение текстов ошибок для клиента.
2. Прописать `openie_mode`/`dataset` Literal в соответствии с реальностью («astrology», «Transformers-offline»), добавить валидацию неизвестных значений с внятной ошибкой.
3. BaseConfig.__post_init__ — читать `EMBEDDING_MODEL_NAME`/`EMBEDDING_BASE_URL` из env.

### Этап 4 — гигиена кода (1 день)
1. Прогнать `pyflakes`/`ruff check --fix` — убрать 176 замечаний (неиспользуемые импорты, дубли, f-строки); отказаться от star-import (явные импорты из misc_utils).
2. Удалить мёртвый код: `EmbeddingCache`, `dynamic_retry`, `sync_render`, `synonym_candidates`, закомментированный блок в logging_utils.
3. `print()` → `logger` (rerank.py, prompt_template_manager.py).
4. Слить два `QuerySolution` в один (в custom_types добавить gold_answers/gold_docs/to_dict); обновить импорты.
5. Выровнять версии пакета (2.0.0-alpha.4), поправить URL в pyproject.toml, вынести `[tool.flake8]` в `.flake8`.
6. Настроить pre-commit (ruff + black) — у проекта уже есть архивный `.pre-commit-config.yaml` в archive/, вернуть в работу.

### Этап 5 — тесты и CI (непрерывно, начать сразу)
1. Юнит-тесты на чистые функции: `compute_mdhash_id`, `text_processing` (включая «ё»), `normalize_entity_name`, `fix_broken_generated_json`, regex-парсеры OpenIE, `min_max_normalize`.
2. Smoke-тест инстанцирования `StandardRAG`/`HippoRAG` на тестовом каталоге (поймал бы ipdb-подобные вставки), тест контракта `infer` (2 или 3 значения — одним тестом).
3. Интеграционный тест индексации 1 документа + 1 запроса на mock-LLM (заглушка вместо OpenAI-клиента).
4. Починить lint-джобу CI (после этапа 4), добавить джобу pytest.
5. (Опционально) Локализовать NER/RE-промпты под русский корпус — вероятно, наибольший прирост качества поиска.

### Оценка суммарных трудозатрат: ~3–4 рабочих дня до полного закрытия P0–P2 и гигиены.

---

## 10. Что работает хорошо

- Аккуратный механизм инкрементальной индексации (`get_missing_string_hash_ids`, resume после обрыва — HippoRAG.index:306-316).
- Обработка битого JSON от LLM (`fix_broken_generated_json` с учётом строк и вложенности).
- Ретраи NER/триплетов с усилением формата (RETRY_JSON_STRICT).
- Пайплайн PDF → текст/OCR → семантический чанкер с overlap и метаданными цитат (ChunkWithCitation) — редкая для форков HippoRAG функциональность.
- Метаданные цитат доходят до ответов API и промптов QA.
- lm_studio_manager — продуманный Windows-специфичный авто-старт LM Studio с ручным фолбэком.
