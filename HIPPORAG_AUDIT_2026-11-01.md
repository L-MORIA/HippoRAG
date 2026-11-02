# Аудит проекта HippoRAG (локальная версия)

**Дата аудита:** 2026-11-01
**Объект:** `C:\Users\User\PycharmProjects\hipporag` — форк OSU-NLP-Group/HippoRAG с локальными доработками (RU-NER, PDF-ингест, FastAPI search API, env-конфиг, LM Studio)
**Метод:** статический анализ + воспроизводимые проверки (AST-разбор, `ast.parse`, реальные вызовы `EmbeddingStore`, `shutil.rmtree` в песочнице, проверка идентичности модулей/классов в `importlib`)
**Проверяющий:** Mavis (mavis)

---

## 1. Резюме

Проект работоспособен, но содержит **набор дефектов, которые молча портят качество ответов**, а также **две критические уязвимости** (потеря данных и запись файлов за пределы каталога).

Главная проблема проекта — не очевидные баги, а **тихое искажение результата**: цитаты указывают не на тот документ, промпт собирается не из того шаблона, LLM получает чужой one-shot пример. Всё это внешне выглядит как «работает, но отвечает странно».

| Severity | Кол-во | Характер |
|----------|--------|----------|
| Critical | 3 | Потеря данных, запись вне каталога, неработающий endpoint |
| High     | 6 | Тихая потеря метаданных цитат, несовместимость API, обход кеша |
| Medium   | 5 | Расхождение шаблонов, подстроковый NER, дрейф зависимостей |
| Low      | 4 | Deprecated API, дубли, шум линтера |

### Baseline на момент аудита

| Проверка | Результат |
|----------|-----------|
| `pytest tests/unit` | **42 passed** (0.25s) |
| `ruff check src --select E9,F,B` | **224 замечания** (из них ~30 — реальные дефекты, остальное F401-шум) |
| `ruff check` (весь проект) | 436 замечаний |
| Покрытие ядра (`src/hipporag`) | **≈ 0%** — 42 теста покрывают только `env_config` и `exceptions` |
| Python | 3.14.3 (системный), в `venv/` — тоже 3.14.3 |

> ⚠️ Проект заявлен как `requires-python = ">=3.10"`, но в `venv/` и в системе — 3.14.3. Код с `datetime.utcnow()` и `on_event` рассчитан на 3.10–3.12.

---

## 2. Critical

### C-01. `clean_database.py --clean-dataset` удаляет проект целиком (path traversal)

**Файл:** `clean_database.py:65`, `clean_database.py:81`, `clean_database.py:131`

```python
dataset_dir = self.base_dir / dataset_name      # base_dir = 'outputs', dataset_name — из argv
...
shutil.rmtree(dataset_dir)
```

**Trigger:** `python clean_database.py --clean-dataset ".."` (или `../..`).

**Impact:** `outputs/..` разрешается в корень проекта. `shutil.rmtree` проходит по нему и удаляет **исходники, документацию, скрипты, `.git`** — то есть весь репозиторий. Воспроизведено в песочнице:

```
before:  tmp\outputs, tmp\IMPORTANT.txt, tmp\outputs\ds1\keep.txt
после rmtree(outputs/..):
IMPORTANT.txt exists: False      ← файл вне outputs/ удалён
```

Тот же вектор в `clean_all()` (`clean_database.py:131`) — удаляет каждый каталог внутри `outputs`, но защиты от `..` там нет только потому, что `iterdir()` не выдаёт `..`; при этом `clean_dataset()` вызывается напрямую по имени из CLI.

**Дополнительно:** `clean_all()` (`clean_database.py:126-134`) **не делает бэкапа** и удаляет датасеты безвозвратно; `--dry-run` есть, но требует интерактивного `input()` — в CI/non-TTY скрипт зависнет.

**Fix direction:** перед `rmtree` валидировать, что `dataset_dir.resolve()` находится внутри `self.base_dir.resolve()`:

```python
base = self.base_dir.resolve()
target = (base / dataset_name).resolve()
if base not in target.parents or target == base:
    raise ValueError(f"Недопустимый путь удаления: {target}")
```

---

### C-02. Path traversal в `/upload-and-ingest` (запись за пределы `temp_uploads`)

**Файл:** `src/hipporag/search_api/server.py:198-202`

```python
from pathlib import PurePosixPath
safe_name = PurePosixPath(file.filename).name
if not safe_name.lower().endswith('.pdf'):
    continue
file_path = upload_dir / safe_name
with open(file_path, "wb") as buffer:
```

**Trigger:** multipart-запрос с `filename="..\\..\\evil.pdf"` (Windows) или `filename="C:\\Windows\\evil.pdf"`.

**Impact:** `PurePosixPath` не считает `\` разделителем на Windows, поэтому `.name` не вырезает путь — имя сохраняется целиком, и `upload_dir / safe_name` уходит за пределы каталога. Проверено:

```
'..\\..\\evil.pdf'      -> resolved= C:\Users\User\PycharmProjects\evil.pdf
'C:\\Windows\\evil.pdf'  -> resolved= C:\Windows\evil.pdf
```

Атакующий может записать произвольный `.pdf` в любой каталог, куда у процесса есть доступ. Эндпоинт **не требует аутентификации**.

**Fix direction:** использовать `Path(file.filename).name` + проверка, что итоговый путь остаётся внутри `upload_dir.resolve()`; для кросс-платформенности — `PureWindowsPath` + `PurePosixPath` одновременно.

---

### C-03. `/upload-and-ingest` на успешном пути возвращает `None` и не индексирует файлы

**Файл:** `src/hipporag/search_api/server.py:190-245`

**Структура (подтверждена AST-разбором):**

```
TRY at line 190, handlers: [(Exception, 212), (Exception, 243)]
  handler 212 body span 213 -> 241     ← весь «успешный» код внутри except
  handler 243 body span 244 -> 245     ← второй except недостижим
```

**Trigger:** любой успешный `POST /upload-and-ingest` с валидным PDF.

**Impact:** строки 215–241 (определение `index_task`, `background_tasks.add_task`, `return {...}`) физически находятся **внутри** блока `except Exception`, а сам `try` заканчивается на строке 211. При отсутствии исключения Python пропускает все `except`-обработчики и выходит из функции → **возвращается `None`**. Последствия:

1. FastAPI отвечает `200 OK` с телом `null` — клиент считает загрузку успешной.
2. `index_task` **никогда не регистрируется** — загруженные PDF остаются лежать в `temp_uploads` и не индексируются.
3. Временные файлы не удаляются (cleanup внутри `index_task`).
4. Второй `except Exception` на строке 243 недостижим — мёртвый код, маскирующий баг (ruff B025 это видит).

**Fix direction:** вынести строки 215–241 на уровень `try` (снять один отступ), удалить дублирующий `except Exception`, оставить единый `except` с логированием.

---

## 3. High

### H-01. Потеря метаданных цитат при одинаковом тексте чанков

**Файл:** `src/hipporag/embedding_store.py:63-103` (дедуп по хешу) + `src/hipporag/HippoRAG.py:391-401` (передача метаданных)

**Trigger:** два разных PDF содержат идентичный текст чанка (колонтитул, дисклеймер, «стр. 5», пустая страница). Это норма для PDF.

**Impact:** `insert_strings` считает хеш от **только текста** (`compute_mdhash_id(text)`), поэтому второй документ отбрасывается как «уже существующий», а его `source_file` / `doc_title` / `page_num` **теряются навсегда**. Воспроизведено:

```
insert A.pdf (page 1) -> stored
insert B.pdf (page 7) -> "Inserting 0 new records, 1 records already exist"
rows stored: 1
  metadata -> {'source_file': 'A.pdf', 'doc_title': 'A', 'page_num': 1}
```

Итог: цитата указывает на **A.pdf вместо B.pdf**. Для проекта с цитированием источников это прямое искажение ответа — причём без единой записи в лог.

**Fix direction:** включить `source_file` (и `page_num`) в ключ идемпотентности либо хранить метаданные списком на каждый `hash_id` и обновлять при повторной вставке.

---

### H-02. OpenAI-путь `infer()` возвращает 2-tuple, локальный — 3-tuple → `qa()` падает

**Файлы:** `src/hipporag/llm/openai_gpt.py:239` (`return response_message, metadata`) vs `openai_gpt.py:300` (`return messages[0]["content"], metadata, False`); потребитель — `src/hipporag/HippoRAG.py:973`:

```python
all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
```

**Trigger:** любой запрос через облачный OpenAI/Azure (не LM Studio) — то есть `llm_base_url` пуст.

**Impact:** `zip(*results)` даёт кортежи длины 2 → распаковка в 3 переменные падает:

```
ValueError: not enough values to unpack (expected 3, got 2)
```

Исключение возникает **после** того, как LLM уже ответил, — то есть на этапе разбора результатов, а не вызова модели. Локальный путь (текущая конфигурация проекта, LM Studio) работает, поэтому баг скрыт в ежедневной разработке и проявится при переходе на облако.

**Fix direction:** привести `infer()` к единому контракту 3-tuple `(content, metadata, cache_hit)` во всех ветках.

---

### H-03. Локальный LLM полностью обходит кеш

**Файл:** `src/hipporag/llm/openai_gpt.py:216-218` → `_infer_local_llm` (`241-306`)

**Trigger:** любое обращение к LM Studio (`llm_base_url` задан).

**Impact:** `_infer_local_llm` не вызывает ни `self.cache.read(params)`, ни `self.cache.write(...)` — в отличие от ветки OpenAI, где кеш работает (`openai_gpt.py:73-103`). Последствия:

1. Каждый повторный вопрос заново обращается к LLM — **в разы медленнее**, чем заявлено в `docs/INDEXING_GUIDE.md:73` («QA Cache автоматически сохраняет ответы»).
2. Файл `llm_cache/{model}_cache.sqlite` создаётся, но остаётся пустым — при этом `clean_database.py --clean-cache` обслуживает именно его, и `docs/CHANGELOG_RU.md` описывает кеш как рабочую подсистему.

**Fix direction:** вынести чтение/запись кеша в общий для обеих веток код до вызова бэкенда.

---

### H-04. `qa()` теряет привязку «фрагмент → источник» из-за `zip`-обрезки

**Файл:** `src/hipporag/HippoRAG.py:935-943`

```python
for i, (passage, citation) in enumerate(
    zip(retrieved_passages, query_solution.citations[: self.global_config.qa_top_k])
):
```

**Trigger:** `retrieval_top_k` (по умолчанию **200**) > `qa_top_k` (по умолчанию **5**).

**Impact:** `retrieved_passages` — это первые 5 доков, а `citations[:5]` — первые 5 цитат. Длины равны, поэтому `zip` здесь не обрезает, **но** пары строятся по **индексу**, а не по принадлежности: `docs[i]` и `citations[i]` независимы. При этом `citations` формируются в `retrieve()` из того же `sorted_doc_ids` (`HippoRAG.py:582-600`), поэтому соответствие 1:1 сохраняется — но только пока `citations` не пустее `docs`.

Реальная проблема: **`docs` и `citations` могут различаться по длине**, если `chunk_embedding_store.get_row()` вернул меньше элементов. Тогда `zip` молча обрезает, и часть фрагментов попадает в промпт **без** указания источника — модель получает «Фрагмент N» без источника и может атрибутировать его другому документу.

**Fix direction:** строить пары по `hash_id`, а не по позиции; при рассинхронизации логировать предупреждение.

---

### H-05. Двойной namespace `src.hipporag` vs `hipporag` — разные классы исключений

**Файлы:** 40+ корневых скриптов используют `from src.hipporag import ...` (`main.py:13`, `index_pdfs.py:19`, `add_documents.py:18`, `search_interactive.py:13`, `scripts/*.py`, `eval/*.py`, `tests/standalone/*`), при этом пакет установлен как `hipporag` (`src/hipporag.egg-info`), а тесты импортируют `hipporag.*` (`tests/unit/test_env_config.py:4`).

**Trigger:** любой скрипт из корня, импортирующий `src.hipporag`, в момент, когда пакет `hipporag` уже загружен (например, `pip install -e .` + импорт из теста).

**Impact:** подтверждено — модули загружаются дважды как **разные объекты**:

```
src.hipporag file: ...\src\hipporag\__init__.py
hipporag   file: ...\src\hipporag\__init__.py     ← тот же файл, но другой module object
SAME MODULE OBJECT: False
QuerySolution same class: False
exceptions same class: False
```

Конкретные последствия:

1. `src/hipporag/utils/env_config.py:20` импортирует `from hipporag.exceptions import ...` — **абсолютный** импорт, тогда как весь остальной пакет использует относительные. При загрузке как `src.hipporag` это тянет **вторую** копию пакета:
   ```
   exceptions used by env_config: hipporag.exceptions
   src.hipporag.exceptions:        src.hipporag.exceptions
   SAME CLASS: False
   ```
2. Исключение, выброшенное `src.hipporag.exceptions`, **не перехватывается** обработчиком `hipporag.exceptions`:
   ```
   NOT caught by hipporag.* -> MISMATCH (exception escapes)
   ```
   То есть `except HippoRAGException` в скрипте не сработает для исключения, поднятого из `env_config`.
3. `isinstance` между namespace'ами всегда `False` — ломает любую проверку типов на границе пакет/скрипт.

**Fix direction:** привести все корневые скрипты к единому `from hipporag import ...` (и наоборот — тесты), и убрать абсолютный импорт в `env_config.py:20` в пользу относительного `from ..exceptions import ...`.

---

### H-06. `index_pdfs.py` удаляет не тот каталог индекса при смене embedding-модели

**Файлы:** `index_pdfs.py:46` vs `src/hipporag/HippoRAG.py:127-131`

```python
# index_pdfs.py:46 — хардкод префикса
working_dir = os.path.join(output_dir, f"{llm_name}_sentence-transformers_{embedding_short}")

# HippoRAG.py:129-131 — реальный путь
self.working_dir = os.path.join(save_dir, f"{llm_label}_{embedding_label}")
```

**Trigger:** смена `embedding_model_name` на любую модель не из `sentence-transformers/*` (например `BAAI/bge-m3` или `nvidia/NV-Embed-v2`).

**Impact:** имя каталога расходится — `clean_old_index()` удалит **несуществующий** каталог (то есть ничего), а реальный индекс останется со старыми эмбеддингами. Проверено:

```
BAAI/bge-m3
   core       : gemma-4-e4b_BAAI_bge-m3
   index_pdfs : gemma-4-e4b_sentence-transformers_bge-m3   match= False   ← удалит не тот
   rebuild    : gemma-4-e4b_BAAI_bge-m3                    match= True
```

`scripts/rebuild_graph.py:46` при этом считает путь **правильно** — то есть два скрипта оперируют разными каталогами, а `index_pdfs.py` при этом делает `shutil.rmtree(working_dir)` (строка 53) **без бэкапа и без подтверждения**.

**Fix direction:** вынести вычисление имени каталога в одну общую функцию в `src/hipporag/utils/` и использовать её во всех скриптах.

---

## 4. Medium

### M-01. Астрологический шаблон промпта не используется по умолчанию

**Файлы:** `src/hipporag/HippoRAG.py:951-961`, `src/hipporag/utils/config_utils.py:196-201`, `src/hipporag/prompts/templates/rag_qa_astrology.py`

**Trigger:** `BaseConfig()` без явного `dataset="astrology"`. По умолчанию `dataset = None` → `is_template_name_valid("rag_qa_None")` → `False` → подставляется `rag_qa_musique`.

**Impact:** в LLM уходит **английский** промпт с one-shot примером из Википедии («The Last Horse», «Neville A. Stanton»), тогда как система работает с русскими астрологическими документами. Подтверждено рендерингом:

```
=== rag_qa_musique ===
  system : As an advanced reading comprehension assistant, your task is to analyze text passages...
  user   : Wikipedia Title: The Last Horse | The Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film...
  assistant : The employer of Neville A. Stanton is University of Southampton...
```

В `rag_qa_astrology.py` при этом есть корректный русский one-shot («Число гуа 8…», формат «Мысли:» / «Ответ:»), который просто не активируется. `dataset="astrology"` выставлен лишь в 6 из ~30 скриптов (`search_interactive.py:55`, `reindex_with_multilingual.py:25`, `tests/standalone/*`) — в `main.py`, `index_pdfs.py`, `quick_search.py`, `eval/evaluate.py` его нет.

Дополнительно: `config_utils.py:196` типизирует `dataset` как `Literal["hotpotqa", "hotpotqa_train", "musique", "2wikimultihopqa"] | None` — значение `"astrology"` **вне контракта**, а `dataset` влияет на `save_dir` (`config_utils.py:234-240`): `dataset="astrology"` → `save_dir = outputs/astrology`.

**Fix direction:** добавить `"astrology"` в `Literal`, задать `dataset="astrology"` по умолчанию (или в конфиг проекта) и вынести выбор шаблона в явную настройку, а не в fall-back на `musique`.

---

### M-02. `pdf_ingestion/chunker.py` выдумывает номера страниц

**Файл:** `src/hipporag/pdf_ingestion/chunker.py:44-46`

```python
page_ratio = start / max(len(tokens), 1)
estimated_page = int(page_num + page_ratio * 10) if page_ratio > 0 else page_num
```

**Trigger:** OCR-путь (`ocr_pdf_parser.py:125-132`, `146-153`), где текст **одной** страницы разбивается на несколько чанков.

**Impact:** `page_ratio * 10` — константа «10 страниц» без связи с реальным числом страниц документа. Для страницы 1 с 4 чанками получаются `page_num` = 1, 3, 5, 8 — цитаты указывают на несуществующие страницы. Формула не имеет смысла: смещение должно быть 0, т.к. все чанки пришли с одной страницы.

**Fix direction:** `page_num` не должен меняться внутри одной страницы — передавать `page_num` без модификации.

---

### M-03. NER: подстроковый матчинг без границ слов → ложные сущности

**Файл:** `src/hipporag/information_extraction/ner_russian.py:118-124`

```python
for term in DOMAIN_ENTITIES:
    idx = text_lower.find(term)
    if idx != -1:
        entities.append(term)
```

**Trigger:** любой текст, содержащий термин как подстроку. Словарь содержит короткие и опасные термины: `"ци"`, `"ша"`, `"янь"`, `"инь"`, `"зона"`, `"вода"`, `"рака"`, `"лев"`.

**Impact:** `text_lower.find("ци")` срабатывает на «циркуль», «позиция», «ацид», «цифра»; `"ша"` — на «шапка», «машина»; `"лев"` — на «левая», «левый». Эти ложные сущности попадают в граф как **узлы-сущности** (`scripts/rebuild_graph.py:79-84`), загрязняя его и размывая PPR-ранжирование. Ошибка тем заметнее, что сам словарь — «безопасные замены» (`semantic_chunker.py:79-87`), т.е. автор явно опасался ложных срабатываний, но не закрыл подстрочный матчинг.

Дополнительно: `ner_russian.py:97` дублирует `is_valid_entity` из `misc_utils.py:15` с **разной** логикой (в `misc_utils` есть проверка «только пунктуация», в `ner_russian` — проверка «формула даты»); в `scripts/rebuild_graph.py` импортируется версия из `misc_utils`, а внутри `RussianNER.extract` используется своя → фильтрация в двух местах различается.

**Fix direction:** использовать regex с границами слов: `re.search(rf'(?<!\w){re.escape(term)}(?!\w)', text_lower)`.

---

### M-04. Необъявленные зависимости: `spacy`, `pandas`, `requests`, `numpy`, `filelock`, `tenacity`, `httpx`

**Файлы:** `pyproject.toml:28-51`, `setup.py:9-34`, `requirements.txt`

| Пакет | Импортируется в | Объявлен в `pyproject.toml`? |
|-------|------------------|------------------------------|
| `spacy` | `src/hipporag/information_extraction/ner_russian.py:11` | **Нет** (нигде) |
| `pandas` | `src/hipporag/embedding_store.py:8` | **Нет** (нигде) |
| `requests` | `src/hipporag/utils/lm_studio_manager.py:12`, `src/hipporag/llm/openai_gpt.py:243` | **Нет** (нигде) |
| `numpy` | 13 файлов в `src/` | **Нет** (есть только в `requirements.txt`) |
| `filelock` | `embedding_model/base.py:98`, `llm/openai_gpt.py:11`, `llm/bedrock_llm.py:10`, `llm/transformers_llm.py:11` | **Нет** (только `setup.py`/`requirements.txt`) |
| `tenacity` | `utils/llm_utils.py:8`, `llm/openai_gpt.py:15` | **Нет** (только `setup.py`/`requirements.txt`) |
| `httpx` | `llm/openai_gpt.py:9` | **Нет** (только `requirements.txt`) |
| `python-multipart` | нужен FastAPI для `File(...)` (`search_api/server.py:184`) | **Нет** в `pyproject.toml` (есть в `setup.py`, `requirements.txt`) |

**Impact:** `pip install -e .` по `pyproject.toml` даёт **неработающую установку** — падает на `import pandas` ещё на `import hipporag`. `spacy` требуется всем скриптам перестроения графа (`scripts/rebuild_graph.py:27`, `scripts/rechunk_semantic.py`, `scripts/reindex_russian.py`), при этом модель `ru_core_news_sm` (`ner_russian.py:112`) не скачивается автоматически — нужен `python -m spacy download ru_core_news_sm`.

**Fix direction:** синхронизировать три файла зависимостей; вынести `spacy` (+ опциональный extra с моделью) и `pandas`/`requests`/`numpy`/`filelock`/`tenacity`/`httpx` в `pyproject.toml`.

---

### M-05. `ensure_lm_studio_ready` возвращает `True` при загруженной любой модели

**Файл:** `src/hipporag/utils/lm_studio_manager.py:193-221`

```python
loaded_model = self.get_loaded_model()
if loaded_model is None:
    return self.load_model_in_lm_studio(model_name)
logger.info(f"LM Studio готов к работе с моделью: {loaded_model}")
return True          # ← model_name не сверяется с loaded_model
```

**Trigger:** в LM Studio загружена модель, отличная от запрошенной (например, просили `gemma-4-e4b`, а загружена `llama-3`).

**Impact:** функция возвращает `True`, вызывающий код (`index_pdfs.py:234`) продолжает индексацию — запросы уйдут к **неверной модели**, а сообщение в логе утверждает, что всё готово. Молчаливая подмена модели → результаты индекса и ответов несопоставимы между прогонами.

**Fix direction:** сравнивать `loaded_model` с `model_name` (с учётом того, что LM Studio возвращает id вида `author/model`); при несовпадении — загружать нужную или падать с явной ошибкой.

---

## 5. Low

### L-01. `PromptTemplateManager.sync_render` всегда падает

**Файл:** `src/hipporag/prompts/prompt_template_manager.py:153-154`

```python
def sync_render(self, name: str, **kwargs):
    return asyncio.run(self.render(name, **kwargs))
```

`render()` — синхронная функция, возвращающая строку, а не корутину. `asyncio.run()` требует awaitable:

```
TypeError: An asyncio.Future, a coroutine or an awaitable is required
```

Метод не вызывается в текущем коде, но остаётся в API как «рабочая» точка входа.

### L-02. Deprecated API в актуальном окружении

- `src/hipporag/search_api/server.py:51` — `@app.on_event("startup")`: в FastAPI 0.136 выдаёт `DeprecationWarning: on_event is deprecated, use lifespan event handlers instead`.
- `src/hipporag/search_api/models.py:34, 53, 61, 71` — `datetime.utcnow()`: deprecated начиная с Python 3.12.
- Проект заявлен как `>=3.10`, фактически разрабатывается на 3.14.3 — расхождение стоит зафиксировать в `pyproject.toml`.

### L-03. `F821 Undefined name` в `pdf_ingestion` — **ложная тревога линтера**

ruff сообщает `F821 Undefined name 'ChunkWithCitation'` в `chunker.py:20`, `text_pdf_parser.py:70`, `semantic_chunker.py:143`, `ocr_pdf_parser.py:94` и `F821 Undefined name 'Image'` в `ocr_pdf_parser.py:84,90`.

**Проверено — это не runtime-баги:** импорт выполняется внутри функции (`from .base import ChunkWithCitation`), а в аннотациях используется строковая форма (`List["ChunkWithCitation"]`), которая не разрешается на импорте. Ruff не видит локальный импорт. Исправляется добавлением `from __future__ import annotations` либо импортом на уровне модуля (в `TYPE_CHECKING` — для `Image`).

Стоит зафиксировать в документе явно, чтобы эти 7 сообщений не отвлекали от реальных находок.

### L-04. Шум линтера и отсутствие CI-контроля качества

`ruff check` на всём проекте даёт 436 замечаний, из которых ~260 автофиксируемых; подавляющая часть — `F401` (неиспользуемые импорты) в `src/` и `tests/standalone/`. При этом:
- `ruff` **не подключён** как pre-commit/CI-шаг (в `pyproject.toml` есть `[tool.flake8]`, но не `[tool.ruff]`; `ruff` нет ни в `setup.py`, ни в `pyproject.toml` dev-зависимостях);
- `pytest` в `addopts` (`pyproject.toml:146`) требует `pytest-cov`, который есть только в extra `dev` → «голая» установка + `pytest` упадёт на неизвестном аргументе `--cov`;
- `mypy` настроен (`pyproject.toml:111-123`), но не запускается и не входит в dev-зависимости;
- `.github/` — новый (untracked) каталог; работоспособность workflow не проверена.

---

## 6. Проверено и признано чистым

| Область | Результат |
|---------|-----------|
| `pytest tests/unit` | 42/42 passed — `env_config` и `exceptions` корректны |
| `.env` | содержит только `LLM_NAME` и `LLM_BASE_URL`, **без секретов**; в `.gitignore` (строка «Ignore the env file») |
| Hardcoded API-ключи | не найдено ни одного в `*.py` (в `openai_gpt.py:150` — заглушка `sk-dummy` для локального LLM) |
| `clean_database.py` / `delete_document.py` | имеют `--dry-run` и текстовое подтверждение `yes/no` — правильный подход (в отличие от C-01) |
| `embedding_store.delete()` | корректно сортирует индексы по убыванию перед удалением (`embedding_store.py:181`) — off-by-one нет |
| `HippoRAG.delete()` | удаляет из графа и всех трёх embedding-store, выставляет `ready_to_retrieve = False` (`HippoRAG.py:505-508`) |
| `index_pdfs.py` | подтверждение и dry-run есть, `find_pdf_files` использует `rglob` без выхода за пределы каталога |
| `main.py` | 3140 строк, 62 функции, **дублирующихся определений нет**; имена функций уникальны |
| `semantic_chunker.clean_pdf_text` | дефис переноса, слипшиеся слова, кратные пробелы — обработка корректна; `known_pairs` — безопасные точные замены |
| `pdf_ingestion/base.py` | `ChunkWithCitation` — корректный dataclass с `hash_id`-свойством |
| `misc_utils.normalize_entity_name` | корректная нормализация (lowercase, trim пунктуации, схлопывание пробелов) |

---

## 7. План исправлений

### Этап 0 — Горячие фиксы (блокируют корректность, 1–2 часа)

Порядок важен: сначала пункты, которые **молча искажают** результат.

| # | Находка | Действие | Файл |
|---|---------|----------|------|
| 0.1 | C-01 | Валидация пути перед `rmtree` (`resolve()` + проверка родителя) | `clean_database.py:65` |
| 0.2 | C-03 | Снять отступ у строк 215–241, удалить мёртвый `except` на 243 | `search_api/server.py:190-245` |
| 0.3 | C-02 | Безопасное имя файла + проверка, что путь внутри `upload_dir` | `search_api/server.py:198-202` |
| 0.4 | H-02 | Единый контракт `infer()` → 3-tuple во всех ветках | `llm/openai_gpt.py:239,300` |
| 0.5 | H-01 | Включить `source_file` в ключ идемпотентности либо хранить метаданные списком | `embedding_store.py:63-103` |
| 0.6 | M-02 | Убрать выдумывание номеров страниц | `pdf_ingestion/chunker.py:44-46` |

**Проверка этапа:** после 0.5 — два PDF с одинаковым колонтитулом должны дать 2 записи с разными `source_file`; после 0.4 — `rag_qa` должен отработать с облачным OpenAI.

### Этап 1 — Целостность данных и окружения (1–2 дня)

| # | Находка | Действие |
|---|---------|----------|
| 1.1 | H-05 | Единый namespace: перевести корневые скрипты на `from hipporag import ...`; заменить абсолютный импорт в `env_config.py:20` на относительный |
| 1.2 | M-04 | Синхронизировать `pyproject.toml` / `setup.py` / `requirements.txt`; добавить `spacy`, `pandas`, `requests`, `numpy`, `filelock`, `tenacity`, `httpx`, `python-multipart` |
| 1.3 | H-06 | Вынести вычисление имени `working_dir` в общую функцию; использовать в `index_pdfs.py`, `scripts/rebuild_graph.py`, `scripts/rechunk_*.py` |
| 1.4 | H-03 | Вернуть кеширование в локальный LLM-путь |
| 1.5 | M-05 | Сверять `loaded_model` с запрошенным `model_name` |
| 1.6 | — | Бэкап перед `rmtree` в `index_pdfs.py:53` и `scripts/rechunk_*.py` (в `rebuild_graph.py` бэкап графа уже есть — распространить на parquet) |
| 1.7 | — | Проверить `.github/` workflow: должен устанавливать пакет с `[dev]`-extra и запускать `pytest` |

### Этап 2 — Качество выдачи (2–3 дня)

| # | Находка | Действие |
|---|---------|----------|
| 2.1 | M-01 | Добавить `"astrology"` в `Literal` `dataset`; задать его по умолчанию; убрать silent-fallback на `rag_qa_musique` (логировать WARNING при подстановке чужого шаблона) |
| 2.2 | M-03 | Regex с границами слов в `RussianNER`; устранить дублирование `is_valid_entity` (оставить одну реализацию в `misc_utils.py`) |
| 2.3 | H-04 | Строить пары «фрагмент→цитата» по `hash_id`; логировать рассинхронизацию |
| 2.4 | L-01 | `sync_render` → просто `return self.render(...)` |
| 2.5 | L-02 | Перевести `on_event` на `lifespan`; `datetime.utcnow()` → `datetime.now(timezone.utc)` |

### Этап 3 — Тесты и качество кода (3–5 дней)

| # | Находка | Действие |
|---|---------|----------|
| 3.1 | — | **Покрытие ядра с ~0%.** Минимальный набор: `embedding_store` (insert/delete/idempotency), `chunk_text_semantic`, `chunker` (номера страниц), `PromptTemplateManager.render`, `qa()` (сборка промпта) |
| 3.2 | L-04 | Добавить `[tool.ruff]` в `pyproject.toml`; включить `ruff check` + `mypy` в CI и pre-commit; вычистить `F401` автофиксом (`--fix`) |
| 3.3 | L-04 | Убрать `--cov` из `addopts` или добавить `pytest-cov` в обязательные dev-зависимости |
| 3.4 | L-03 | Закрыть 7× `F821` через `from __future__ import annotations` |
| 3.5 | — | Проверить работоспособность `.github/` workflow (установка + `pytest`) |

### Этап 4 — Документация и структура (1–2 дня)

| # | Находка | Действие |
|---|---------|----------|
| 4.1 | — | **В `docs/` ~50 файлов с дублирующимися именами** (`FINAL_REPORT.md`, `FINAL_SUMMARY.md`, `FINAL_SOLUTION_REPORT.md`, `COMPLETE_SOLUTION_REPORT.md`, `SOLUTION_SUMMARY.md`, `SUMMARY.md`, `CONCLUSIONS.md`, `ИТОГОВОЕ_РЕШЕНИЕ.md`, `ИТОГОВЫЙ_ОТЧЕТ.md`, `ГОТОВО.md`, `CHECKLIST.md`, `COMPLETION_CHECKLIST.md`, `COMPLETION_REPORT.md`, `IMPLEMENTATION_GUIDE.md`, `REFACTORING_*.md` ×4, `README_*.md` ×7 и др.). Оставить один источник истины, остальные — в `archive/` |
| 4.2 | — | Проверить факты в `docs/INDEXING_GUIDE.md` против кода: заявлены «1422 чанка», «6979 сущностей», «8401 узел», «23 метода поиска», «LLM выбирается из 4 моделей» — числа зафиксированы на момент конкретного прогона и устаревают; сделать их вычисляемыми или убрать |
| 4.3 | — | `docs/INDEXING_GUIDE.md:73` обещает работающий QA-кеш — после H-03 обещание станет правдой |
| 4.4 | — | Добавить `README` раздел: фактические зависимости, модель `ru_core_news_sm`, команды индексации/поиска |
| 4.5 | — | `tests/standalone/` (44 файла) — это ручные скрипты, а не тесты; перенести в `scripts/manual/` или помечать как явно не-pytest, чтобы не путать с `tests/unit` |

---

## 8. Рекомендуемый порядок (резюме)

```
Неделя 1:  Этап 0 (6 правок, ~2ч)  →  снимает искажение ответов и 2 уязвимости
           + покрытие тестами embedding_store и chunker

Неделя 2:  Этап 1 (namespace, зависимости, working_dir, кеш)
           + Этап 2 п. 2.1 (dataset=astrology) → качество промпта

Неделя 3:  Этап 2 остальное + Этап 3 (тесты, ruff/mypy в CI)
           + Этап 4 (схлопывание docs/)
```

**Ключевой принцип:** Critical и High из Этапа 0–1 бьют по **достоверности результата**, а не по стабильности. Система работает и отвечает, но часть ответов ссылается на неверные источники, а часть сценариев (облачный OpenAI, смена embedding-модели, `/upload-and-ingest`) падает или работает вхолостую.

---

## 9. Приложение: воспроизведение находок

| Находка | Как проверить |
|---------|---------------|
| C-01 | `python clean_database.py --clean-dataset ".."` в тестовой директории → удаляется корень |
| C-02 | `PurePosixPath('..\\..\\x.pdf').name` → `..\..\x.pdf` (не sanitized) |
| C-03 | `ast.parse` на `search_api/server.py` → handler 212 body = 213..241 |
| H-01 | `EmbeddingStore.insert_strings` дважды с одинаковым текстом и разными метаданными → 1 запись |
| H-02 | `zip(*[( 'a', {}) ])` в 3 переменные → `ValueError: not enough values to unpack` |
| H-05 | `src.hipporag is hipporag` → `False`; `isinstance` между namespace'ами → `False` |
| H-06 | Сравнение строк `working_dir` для `BAAI/bge-m3` → `match=False` |
| M-01 | `BaseConfig().dataset` → `None`; `is_template_name_valid('rag_qa_None')` → `False` |
| M-02 | `chunk_text_with_metadata` на 1 странице → `page_num` = 1, 3, 5, 8 |
| L-01 | `asyncio.run(render(...))` → `TypeError: An asyncio.Future, a coroutine or an awaitable is required` |
