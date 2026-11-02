# Проверка выполнения сводного плана аудита

**Дата проверки:** 2026-11-01
**Проверяющий:** Mavis (mavis)
**Проверяемый документ:** `docs/AUDIT_SYNTHESIS_2026-11-01.md` (49 271 байт), раздел 10 «Журнал выполнения»
**Чек-лист:** приложение раздела 9 (воспроизводимость находок) + `pytest tests/unit`

---

## 1. Вердикт

## ⚠️ ВЫПОЛНЕНО НА 85–90 % — ОТЧИТ ПРИНЯТЬ С 6 ПОПРАВКАМИ

План выполнен **по-настоящему, а не декларативно**: я проверил каждый этап по коду и воспроизвёл приёмку эмпирически. Это не «отчёт о проделанной работе» — код действительно изменён, и ключевые дефекты безопасности закрыты.

Но журнал содержит **7 расхождений с фактическим состоянием**, из которых 2 существенны:

- **S-12 (моё блокирующее условие) формально исправлен, но пользовательский симптом остался** — цитаты по-прежнему указывают на первый документ.
- **CI lint-job красный** во всех четырёх шагах, при этом в журнале написано «CI зелёный».

Обе правки — по ~30 минут.

---

## 2. Что подтверждено (проверено, а не принято на веру)

### Baseline

| Проверка | Результат | Совпадает с журналом |
|----------|-----------|---------------------|
| `pytest tests/unit` | **62 passed** (0.88s) | ✅ да |
| `pytest tests/unit -m unit` (как в CI) | **62 passed**, exit 0 | ✅ маркер проставляется в `tests/conftest.py:158-166` |
| `compileall src/hipporag` | exit 0 | ✅ да |
| `grep set_trace\|import ipdb\|assert False\|except:$` по `src/` | **0 совпадений** | ✅ да |
| `print(` в `src/` | **0** | ✅ да (S-34) |
| `utcnow` в `src/` | **0** | ✅ да (S-27) |
| Импорты `src.hipporag` | **0** во всём проекте | ✅ да (S-13) |
| `setup.py` | удалён, `requirements.txt` → `-e .` | ✅ да (S-09) |
| `docs/`: активных файлов / в `archive/docs/` | **11 / 47** | ✅ да (S-41) |
| `git rm --cached` для `outputs/musique/*.json` | 2 файла в staged-удалении | ✅ да (S-40) |

### Приёмка S-30 — выполнена (это главное)

Поднял API через `TestClient` с реальным индексом:

```
HEALTH status: 200
HEALTH body  : {"status":"healthy","version":"1.0.0","uptime":"0h 1m 26s","chunks":1422,...}
working_dir  : outputs/astrology_test\gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2
chunks       : 1422
entities     : 6979
graph vcount : 8401
```

API теперь видит **тот самый** индекс, который не видел раньше. Дефект «API поднимает пустой индекс» закрыт по-настоящему: конфиг из env, дефолт — мультиязычная модель, `working_dir` совпадает с `index_pdfs.py`, при старте проверяется непустота.

`/search` вернул 500 с обобщённым текстом (`"Внутренняя ошибка поиска"`) — LM Studio в этой среде не запущен, `ConnectionError` корректно не пробросился клиенту. Это подтверждает S-30 в части «обобщённые тексты ошибок».

### Этап 0 — все 8 правок (7 подтверждены полностью, 1 частично)

| ID | Проверка | Итог |
|----|----------|------|
| S-08 | `_validate_inside_base()` (`clean_database.py:30-34`) + эмпирический прогон | ✅ `clean_dataset("..")` → `False`, `IMPORTANT.txt` **выжил**; `clean_dataset("../..")` → `False`; легитимный `ds1` удалён |
| S-07 | `PureWindowsPath(...).name` + `is_relative_to` (`server.py:229-233`) | ✅ `'..\..\evil.pdf'` → `'evil.pdf'`, `'C:\Windows\evil.pdf'` → `'evil.pdf'`, всё внутри `upload_dir` |
| S-06 | AST-разбор `server.py` | ✅ дублей `except` нет, `index_task` вынесен из недостижимого блока (стр. 253-282), успешный путь возвращает JSON |
| S-01 | `ipdb` в `StandardRAG` | ✅ удалён |
| S-02 | `ipdb`/bare `except` в `embedding_model/OpenAI.py` | ✅ удалены |
| S-03 | `pre_openie` → `NotImplementedError` (`HippoRAG.py:233-236`) | ✅ с внятным сообщением |
| S-20 | `chunker.py:45-52` | ✅ `page_num=page_num`, магическая константа `*10` убрана, комментарий объясняет почему |
| **S-12** | `embedding_store.py:76-155` | ⚠️ **частично** — см. раздел 3 |

### Этап 1 — все 8

| ID | Проверка | Итог |
|----|----------|------|
| S-04 | `@cache_response` применён к `infer` (`openai_gpt.py:208`); оба пути возвращают 2-tuple, декоратор добавляет третий элемент (`openai_gpt.py:77,103`) | ✅ контракт унифицирован — **но не для всех потребителей**, см. D-5 |
| S-05 | `StandardRAG.py:265` → `self.retrieve(queries=queries)` | ✅ `retrieve_dpr` больше не вызывается |
| S-18 | `misc_utils.py:9` → `from .custom_types import QuerySolution, Triple  # noqa: F401 (реэкспорт)` | ✅ один класс |
| S-09 | `pyproject.toml` — 33 зависимости, extras `ru`/`offline`/`dev` | ✅ `spacy`, `pandas`, `pyarrow`, `requests`, `numpy`, `filelock`, `tenacity`, `httpx`, `rank-bm25`, `python-multipart` на месте |
| S-10 | `env_config.py` на pydantic-settings v2 (`SettingsConfigDict`, `field_validator`) | ✅ fallback на `pydantic.BaseSettings` удалён |
| S-13 | `grep '^from src\.'` по всему проекту → 0 | ✅ |
| S-22 | `build_working_dir()` в `config_utils.py:246`, внедрён в HippoRAG/StandardRAG/index_pdfs | ✅ |
| S-23 | `lm_studio_manager.py:219-225` — сверка загруженной модели с запрошенной | ✅ |

### Этап 2 — все 9 (это самый качественный блок)

| ID | Проверка | Итог |
|----|----------|------|
| S-15 | `text_processing("Чёрный ёж (рак знак)")` → `'чёрный ёж рак знак'` | ✅ «ё» сохранена |
| S-16 | `normalize_entity_name("  Марс!  ")` → `'марс'` — совпадает с `text_processing` | ✅ один hash-ключ |
| S-11 | `re.sub(r"[\W_]+", "", "марс", flags=re.UNICODE)` → `'марс'`, len 4 > 2 | ✅ synonymy-рёбра для кириллицы теперь строятся |
| S-21 | `RussianNER().extract("В продаже циркуль и позиция. Также шапка и левый берег.")` → `['продажа','циркуль','позиция','шапка','берег']` | ✅ «ци»/«ша»/«лев» как **подстроки** больше не матчатся |
| S-19 | `BaseConfig().dataset` → `'astrology'`, `Literal` дополнен (`config_utils.py:196-199`); WARNING при подстановке чужого шаблона (`HippoRAG.py:952-955`) | ✅ |
| S-17 | `np.any(node_weights)` → graceful fallback на dense retrieval (`HippoRAG.py:1929-1934`); `assert False` в `embedding_model/__init__.py` → `ValueError` | ✅ |
| S-29 | `run_ppr` → `RuntimeError` при пустом маппинге (`HippoRAG.py:2029-2032`) | ✅ |
| S-14 | `HippoRAG.py:1758-1759` — множитель применяется один раз | ✅ |
| S-36 | `Transformers.py:48-52` — префиксы `query: `/`passage: ` для E5 | ✅ |

### Этап 3 — все 8

| ID | Проверка | Итог |
|----|----------|------|
| S-30 | см. приёмку выше | ✅ |
| S-27 | `lifespan` вместо `on_event` (`server.py:35-72`); `utcnow` → 0 | ✅ |
| S-24 | WARNING о рассинхроне + безопасный доступ `citations[i] if i < len(citations) else None` (`HippoRAG.py:922-928`) | ✅ |
| S-25 | `sync_render` → `list` из 4 сообщений, TypeError ушёл | ✅ |
| S-26 | bare `except:` в `src/` → 0 | ✅ |
| S-31 | `threading.RLock` в `EmbeddingStore` (`embedding_store.py:37`); `ValueError` на битый parquet (`embedding_store.py:180-185`) | ✅ |
| S-32 | `temperature` → `config_dict.get("temperature", 0.0)` (было 0.7) (`openai_gpt.py:201`) | ✅ |
| S-37 | `_extract_list_field` — строгий `json.loads` + regex-fallback (`openie_openai.py:126`) | ✅ |

### Этап 4–6

| ID | Проверка | Итог |
|----|----------|------|
| S-28 | `EmbeddingCache` удалён? | ❌ **нет** — см. D-3 |
| S-34 | `print()` → logger | ✅ 0 в `src/` |
| S-35 | `RuntimeError from err`, `DirectoryNotFoundError` | ✅ |
| S-38 | ruff F-класс = 0? | ❌ **19** — см. D-4 |
| S-39 | `from __future__ import annotations` добавлен (4 файла `pdf_ingestion`) | ⚠️ есть, но ruff всё равно даёт 6× F821 — см. D-4 |
| S-43 | `.flake8` создан; версии сведены (`__init__.py` = `2.0.0-alpha.4`, build-файлы = `2.0.0-alpha.4`) | ✅ |
| S-33 | `pytest-cov` в dev, `--cov` убран из `addopts`, CI test-job есть | ⚠️ частично — см. D-2 |
| S-40 | `.gitignore` дополнен (`htmlcov/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `models/`, `archive/`, `*.egg-info/`, `temp_uploads/`) | ✅ |
| S-41 | docs схлопнуты 47 → `archive/docs/`, 11 активных | ✅ |
| S-42 | `collect_ignore_glob = ["standalone/*"]` в `tests/conftest.py:15` | ✅ |
| S-44 | main.py | ✅ отложен по плану |

---

## 3. Расхождения журнала с фактами (требуют правки)

### ⛔ D-1. S-12: данные сохранены, но цитата по-прежнему указывает на неверный документ

**Это моё блокирующее условие из вердикта по синтезу — и оно формально, но не по существу выполнено.**

Что сделано (правильно): `embedding_store.py:96-111` при повторной вставке не отбрасывает запись, а дописывает источник в список; `_coerce_metadata_entries` (`:124-142`) обеспечивает backward-совместимость со старым форматом parquet.

Что не доделано: `get_metadata()` (`:252-256`) возвращает **только первый** источник, а `HippoRAG.retrieve` (`HippoRAG.py:574`) обращается именно к нему:

```
metadata list          : [{'source_file': 'A.pdf', 'page_num': 1}, {'source_file': 'B.pdf', 'page_num': 7}]
get_metadata() ->      : {'source_file': 'A.pdf', 'page_num': 1}
CITATION that user sees -> source_file=A.pdf page_num=1
expected                 -> source_file=B.pdf page_num=7
```

`get_all_metadata_entries()` (`:258-263`) **не вызывается нигде в production-коде** — только в тестах. То есть исходный вред («цитаты указывают не на тот документ») для пользователя **сохраняется**: второй документ по-прежнему не будет назван в цитате.

**Итог:** система теперь хранит данные, но не показывает их. Починка хранилища без починки потребителя.

**Дополнительно — приёмка из приложения не выполнена по существу.** Пункт приложения гласит: «два PDF с одинаковым колонтитулом → в `vdb_chunk.parquet` **2 записи** с разными `source_file`». Фактически — **1 запись** со списком из 2 источников. Критерий в документе не соответствует реализации.

**Правка (30 мин):** в `HippoRAG.retrieve` (`HippoRAG.py:574-599`) строить цитату из всех записей — например, `source_file` = первый источник, а в `Citation` добавить поле `sources: List[str]` либо дописывать «источники: A.pdf, B.pdf»; в API-модели `SearchResult` отдавать список. Альтернатива — вернуть вариант «а» (включить `source_file` в ключ идемпотентности), но он меняет hash-ключи и требует переиндексации.

### ⛔ D-2. CI lint-job красный во всех четырёх шагах; «CI зелёный» — неверно

Журнал (Этап 5) заявляет «test-job в CI уже был» и в итоговом состоянии — «62/62 теста зелёные». Но workflow (`.github/workflows/ci-cd.yml`) — **upstream-версия**, и она запускает `black`/`isort`/`flake8`/`mypy`, а **не `ruff`**. Прогнал ровно те команды, что в CI:

| Шаг CI | Команда | Результат |
|---------|---------|-----------|
| Black | `black --check src/ tests/` | **exit 1** — 57 файлов требуют переформатирования (включая `src/hipporag/HippoRAG.py`) |
| isort | `isort --check-only src/ tests/` | **exit 1** — `Traceback`: в `[tool.isort]` невалидная опция `multi_line_mode` (правильно `multi_line_output`); isort падает на конфиге |
| Flake8 | `flake8 src/ tests/` | **exit 1** — **640 замечаний** (154 в `src/`, 486 в `tests/`) |
| MyPy | `mypy src/hipporag --ignore-missing-imports` | **exit 2** — `numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12 and greater`; причина: в `[tool.mypy]` стоит `python_version = "3.10"`, а numpy 2.x использует `type`-statements |

**Следствия для S-33** («ruff+mypy в CI/pre-commit»):
- `ruff` в CI **нет вообще** и в `dev`-extra его нет;
- `.pre-commit-config.yaml` переехал в `archive/`, а `archive/` в `.gitignore` → **pre-commit неактивен**;
- mypy в CI не заработает, пока `python_version` не поднять до 3.12 или не зафиксировать numpy < 2.

**Ещё:** job `docs` в CI выполняет `cd docs && make html` — в `docs/` нет Sphinx-Makefile, job упадёт при push в `main`.

**Правка (30 мин):** `multi_line_mode` → `multi_line_output`; `python_version` в `[tool.mypy]` → `"3.12"`; `black src/ tests/` и `flake8 --fix` (или расширить `.flake8`, чтобы не рубить легитимный код); в CI заменить black/isort на `ruff check` + `ruff format`; вернуть `.pre-commit-config.yaml` в корень.

### ⚠️ D-3. S-28 «удалён `EmbeddingCache`» — не удалён, при этом сломан

`src/hipporag/embedding_model/base.py:131-171` — класс на месте. `multiprocessing` и `threading` **не импортированы** (в файле импортируются только `json`, `dataclasses`, `typing`, `numpy`):

```python
class EmbeddingCache:
    @classmethod
    def _init_cache(cls):
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()   # ← NameError: name 'multiprocessing' is not defined
            cls._cache = cls._manager.dict()
            cls._lock = threading.Lock()                # ← NameError
```

Сейчас никем не вызывается (мёртвый код), поэтому не стреляет — но это **единственные настоящие F821** во всём `src/`. Заявление журнала «удалены `EmbeddingCache`, …» неверно.

Рядом: `embedding_model/base.py:78,85` — аннотация `"LLMConfig"` в классе `EmbeddingConfig` (copy-paste из `llm/base.py`) → F821, runtime-безопасно, но запутывает.

**Правка:** удалить `EmbeddingCache` (как и планировалось) — 10 минут.

### ⚠️ D-4. «ruff F-класс — 0» неверно: 19 замечаний

Фактический состав:

| Тип | Кол-во | Что это |
|-----|--------|---------|
| F821 | 8 | **2 настоящих** (`multiprocessing`, `threading` в `embedding_model/base.py`) + **6 ложных** (`ChunkWithCitation` ×4, `Image` ×2 — только в строковых аннотациях) |
| F401 | 4 | `__init__.py` без `__all__`: `embedding_model`, `information_extraction`, `llm`, `prompts` |
| F841 | 4 | неиспользуемые локальные переменные: `predicate_phrase` (`HippoRAG.py:1866`), `guided` (`transformers_offline.py:88`), `model` (`lm_studio_manager.py:256`), `e` (`qa_utils.py:57`) |

**Про S-39:** `from __future__ import annotations` действительно добавлен в 4 файла `pdf_ingestion`, но ruff всё равно даёт 6× F821 — потому что `ChunkWithCitation` импортируется **внутри функции** (`chunker.py:23`), а не на уровне модуля. Чтобы замолчать F821, нужен импорт на уровне модуля (в `TYPE_CHECKING` — для `Image`).

### ⚠️ D-5. Контракт `infer` применён не ко всем потребителям

`src/hipporag/utils/qa_utils.py:55`:

```python
response_message, metadata = llm_client.infer(messages=messages)
```

а `infer` теперь возвращает **3-tuple** `(message, metadata, cache_hit)`:

```
infer() returns: ('RESP', {'tokens': 1}, False) | len = 3
2-value unpack -> ValueError: too many values to unpack (expected 2, got 3)
```

Исключение проглатывается `except Exception` (`:57`) → функция **всегда возвращает `""`**. План формулировал S-04 как «единый контракт 3-tuple **для всех потребителей**» — этот потребитель пропущен. (Модуль сейчас никем не импортируется → Low по влиянию, но заявление о единстве контракта неверно.)

### ⚠️ D-6. `.env` не загружается библиотекой — `env_config.py` стал мёртвым кодом

`env_config.py` (единственный модуль с `load_dotenv(ENV_FILE)`) **не импортируется ни одним модулем пакета**: ни `hipporag/__init__.py`, ни `search_api/server.py`, ни `config_utils.py`. `load_dotenv()` вызывают только скрипты верхнего уровня (`index_pdfs.py:14`, `main.py:11`, `scripts/*.py`, `eval/*.py`).

Проверено:

```
import hipporag
env_config in sys.modules : False
LLM_NAME env              : None
BaseConfig()              : gpt-4o-mini | None      ← цель — облако OpenAI, а не LM Studio
```

Последствия:
1. **S-10 сделан в модуле, который никто не использует** — рефакторинг `env_config` на pydantic-settings v2 не влияет ни на что в проде; сам модуль импортируют только тесты.
2. Любое использование библиотеки вне скриптов (`python -c "from hipporag import HippoRAG; HippoRAG()"`, новый скрипт без `load_dotenv()`) молча уходит в облако OpenAI с моделью `gpt-4o-mini`.
3. `server.py:42-48` читает `os.getenv(...)` с дефолтами, **совпадающими** с текущим `.env` → фактическое игнорирование `.env` замаскировано совпадением значений. Стоит поменять `.env` — API его не увидит.

Работающие скрипты не затронуты, поэтому в ежедневной работе это не проявляется.

**Правка:** импортировать `env_config` в `hipporag/__init__.py` (или перенести `load_dotenv` в `config_utils.py`) — 5 минут.

### ⚠️ D-7. Факт-стор индекса содержит одну строку `"placeholder"` — и приёмка `/health` этого не замечает

Проверено на диске:

```
chunk:  rows=1422 (7 900 026 bytes)
entity: rows=6979 (22 379 346 bytes)
fact:   rows=1     (7 046 bytes)  content='placeholder'
graph.pickle: 4 286 205 bytes
```

Текущий код «placeholder» не пишет (`grep` по `src/` — только в тексте докстрингов), значит строка осталась от прежней версии кода.

**Что это значит:** `get_fact_scores` (`HippoRAG.py:1631-1638`) оперирует одним вектором, `rerank_facts` получает единственного кандидата «placeholder», и **графовая часть HippoRAG (PPR по фактам) работает вырожденно** — фактически поиск опирается на dense-retrieval, а не на граф. Это подтверждает, что переиндексация обязательна не из-за «hash-гигиены», а потому что текущий индекс **неполон**.

**Связанный пробел в S-30:** новая проверка непустоты индекса смотрит только на `chunks` (`server.py:51-58`), поэтому `/health` отдаёт `chunks: 1422` и выглядит здоровым, хотя фактов — 1. Приёмка «chunks > 0» недостаточна: стоит проверять и `facts`/`entities` (или минимальный порог по фактам), иначе такая проверка будет давать ложную уверенность и дальше.

### ⚠️ D-8. Тест S-12 закрепляет именно то поведение, ради устранения которого находка была заведена

`tests/unit/test_stage0_fixes.py:32`:

```python
assert store.get_metadata(hash_id)["source_file"] == "A.pdf"
```

Тест фиксирует, что цитата указывает на **первый** документ, — то есть буквально закрепляет исходный дефект как ожидаемое поведение. Именно этот сценарий я предупреждал в вердикте по синтезу: «тест на метаданные без фикса дефекта либо провалится, либо будет написан так, что закрепит текущее неверное поведение как ожидаемое». Здесь — второй вариант.

---

## 4. Сводка расхождений

| ID | Что заявлено в журнале | Что на самом деле | Влияние |
|----|------------------------|-------------------|---------|
| **D-1** | S-12 исправлен, приёмка «2 записи с разными source_file» | Данные сохранены, но цитата указывает на первый документ; в индексе **1 запись**; новый метод не используется в проде | **Высокое** — исходный вред не устранён |
| **D-2** | «CI зелёный», S-33 выполнен | black/isort/flake8/mypy — **все 4 красные**; ruff в CI нет; pre-commit в `archive/` (gitignored) | **Высокое** — CI не проходит |
| D-3 | S-28: `EmbeddingCache` удалён | Класс на месте, `multiprocessing`/`threading` не импортированы → NameError при вызове | Среднее (код мёртвый) |
| D-4 | ruff F-класс = 0 | 19 замечаний (2 настоящих F821, 6 ложных, 4 F401, 4 F841) | Низкое |
| D-5 | Единый контракт `infer` для всех потребителей | `qa_utils.py:55` распаковывает 2 из 3 → `ValueError` → тихий `""` | Низкое (модуль не используется) |
| D-6 | S-10 выполнен | `env_config.py` не импортируется нигде → `.env` не грузится библиотекой; `BaseConfig()` → `gpt-4o-mini` / `None` | Среднее |
| D-7 | Индекс в порядке, приёмка `chunks > 0` | `facts = 1` («placeholder») → графовая часть вырождена; проверка `/health` это не видит | **Среднее–высокое** (усиливает необходимость переиндексации) |
| D-8 | S-12 покрыт тестом | Тест закрепляет неверное поведение как ожидаемое | Среднее |

---

## 5. Рекомендация

**План принять.** Основная работа сделана качественно: 7 из 8 правок Этапа 0 полностью закрыты, весь Этап 2 выполнен корректно (это самый сложный блок — кириллица, нормализация, synonymy, NER, промпт), S-30 доведён до рабочего состояния и подтверждён живым прогоном API.

**Перед тем, как считать работу закрытой, внести 6 правок** (суммарно ~2–3 часа):

| # | Что | Файл | Время |
|---|-----|------|-------|
| 1 | **D-1**: цитата должна показывать все источники (или вернуть вариант «а») | `src/hipporag/HippoRAG.py:574-599`, `search_api/models.py` | 30 мин |
| 2 | **D-2**: `multi_line_mode` → `multi_line_output`; `python_version = "3.12"`; `black`/`ruff` в CI; `.pre-commit-config.yaml` в корень | `pyproject.toml`, `.github/workflows/ci-cd.yml` | 30 мин |
| 3 | **D-3**: удалить `EmbeddingCache` | `src/hipporag/embedding_model/base.py:131-171` | 10 мин |
| 4 | **D-4**: `__all__` в 4 `__init__.py`, убрать 4 неиспользуемые локальные переменные, модульные импорты для `ChunkWithCitation`/`Image` | `src/hipporag/**` | 30 мин |
| 5 | **D-6**: `load_dotenv` в `hipporag/__init__.py` (или в `config_utils.py`) | `src/hipporag/__init__.py` | 5 мин |
| 6 | **D-5/D-8**: поправить `qa_utils.py:55` на 3-tuple и переписать тест S-12 так, чтобы он **падал** на старом поведении | `src/hipporag/utils/qa_utils.py`, `tests/unit/test_stage0_fixes.py` | 20 мин |

**Отдельно (не входит в правки, но обязательно к использованию):**

- **Переиндексация** — не «для чистоты хешей», а потому что факт-стор содержит одну строку `placeholder` (D-7). После переиндексации проверить: `facts` > 0, `/health` показывает ненулевой `facts`, synonymy-рёбра > 0 в графе.
- **Расширить проверку непустоты индекса** (`server.py:51-58`) минимум на `facts` — иначе приёмка `chunks > 0` будет давать ложную уверенность.

---

## 6. Приложение: как воспроизвести ключевые проверки

```bash
# Baseline
python -m pytest tests/unit -q                     # 62 passed
python -m compileall -q src/hipporag                # exit 0
python -m ruff check src --select F                 # 19 errors (не 0)

# CI-как-есть
python -m black --check src/ tests/                 # exit 1, 57 файлов
python -m isort --check-only src/ tests/            # exit 1, Traceback (multi_line_mode)
python -m flake8 src/ tests/                        # exit 1, 640 замечаний
python -m mypy src/hipporag --ignore-missing-imports  # exit 2, numpy stub vs python_version=3.10

# S-08 — path traversal
python clean_database.py --clean-dataset ".."       # должен ОТКАЗАТЬ

# S-12 — цитата указывает на первый документ
# два PDF с одинаковым колонтитулом → get_metadata() вернёт source_file первого

# Приёмка S-30
# поднять API → GET /health → chunks: 1422 (проходит), но facts: 1 (дефект D-7)
```

---

## 7. Что проверено и признано чистым

- `pytest tests/unit` — 62/62, включая маркер `-m unit` (авто-маркировка в `conftest.py:158-166`).
- `compileall` — 0 синтаксических ошибок.
- Namespace: ни одного импорта `src.hipporag` во всём проекте; `hipporag` и `src/hipporag` — один модуль.
- `setup.py` удалён, `requirements.txt` → `-e .`, зависимости сведены в `pyproject.toml` (33 шт. + 3 extras).
- Отладочных вставок нет: `ipdb`/`set_trace`/`assert False`/bare `except:` → 0; `print()` → 0; `utcnow` → 0.
- `embedding_store.delete()` — корректная сортировка индексов по убыванию, off-by-one нет.
- `EmbeddingStore` — `RLock` на мутациях, `ValueError` на битом parquet, backward-совместимость формата метаданных.
- `docs/` — 47 файлов в `archive/docs/`, 11 активных, дубли убраны.
- `.gitignore` расширен; `git rm --cached` для 2 файлов `outputs/` выполнен (в staged-удалении).
- `tests/standalone` исключён из сбора pytest; `collect_ignore_glob` работает.
- Кириллица: «ё/Ё» сохранена, `normalize_entity_name` == `text_processing`, synonymy-регулярка даёт ненулевую длину для кириллицы, NER не матчит подстроки.
- Промпт: `dataset` по умолчанию `astrology`, рендер даёт русский астрологический шаблон с корректным one-shot, `sync_render` работает.
- Обработка ошибок API: внутренние трейлеки не утекают клиенту, `ConnectionError` от недоступного LM Studio корректно превращён в 500 с обобщённым текстом.
