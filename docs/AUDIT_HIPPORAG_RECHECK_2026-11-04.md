# Повторная проверка пакета `src/hipporag` по внешнему аудиту от 31.10.2026

**Дата проверки:** 04.11.2026 · **Проверяемый коммит:** `30f4304` (main, 2026-11-04) — «feat(H1-3): performance budget закрыт — default смёнён на режим 5 (p95 225мс, втрое быстрее), merge_baseline для снапшотов, PERFORMANCE.md, 268 тестов»
**Основание:** внешний аудит кода пакета `src/hipporag` от 31.10.2026, выполненный на форк-коммите `9e881f8` («Audit: fix security vulnerabilities and functional bugs»): 7 находок (1 критическая, 4 существенных, 2 организационных).
**Метод:** свежий клон `origin/main`; установка `pip install -e ".[dev]"` (Python 3.12, Windows; в окружении **намеренно отсутствуют** `vllm` и `outlines` — это часть проверок); прогон штатного тест-сьюта командой CI (`pytest tests/unit -m unit`); воспроизведение lint-трио CI (`black --check`, `isort --check-only`, `ruff check`); точечные проверки-эксплуатации каждой находки: import-пробы без vllm, стабы LLM/эмбеддера/токенизатора, разбор кода построчно.

---

## 1. Итог

Из **7 находок аудита от 31.10** в текущем main: **4 закрыты полностью, 1 частично, 1 откатилась, 1 неактуальна** (репозиторий отсоединён от upstream-сети форков и развивается самостоятельно).

Одновременно при пересборке `src/hipporag` из другой кодовой базы **потеряны 3 рабочих фикса коммита `9e881f8`**, и одна из потерь снова ломает `StandardRAG.rag_qa` целиком — это новый критический дефект **R-1** (§4.1). Штатный тест-сьют эти регрессии не ловит: у `StandardRAG` покрыта только конструкция, но не `rag_qa`.

Тесты и lint: `pytest tests/unit -m unit` → **268 passed**; `black --check` → 168 файлов без изменений; `isort`, `ruff` → чисто. 51 ошибка при первом прогоне — локальная проблема окружения Windows (права на `Temp\pytest-of-ALEXEY`), к репозиторию отношения не имеет, снимается `--basetemp` (§5.1).

| # | Находка аудита от 31.10 | Статус | Подтверждение |
|---|---|---|---|
| **S-1** | 🔴 NameError в `StandardRAG.__init__` (`embedding_base_url` не в сигнатуре) → падение любой конструкции | ✅ **закрыто** | блок удалён целиком; конструкция проверена живьём; есть тест `tests/unit/test_core_utils.py:155` |
| **S-2** | 🟠 `import hipporag.StandardRAG` падает без vllm (модульный импорт `openie_vllm_offline`) | ✅ **закрыто** | импорт убран из `StandardRAG`; в `HippoRAG.py:18–26` обёрнут в `try/except ImportError`; vllm — в extra `offline`. Живьём: пакет, `StandardRAG`, `HippoRAG` импортируются без vllm |
| **S-3** | 🟠 необъявленные зависимости (`outlines`, `sentence-transformers`) | ⚠️ **частично** | `sentence-transformers>=2.2.0` добавлен (`pyproject.toml:35`), `pip install -e ".[dev]"` работает; `outlines` **по-прежнему не объявлен** (§3.3) |
| **S-4** | 🟠 баги `transformers_offline` (decode-спан, free-form batch, `max_tokens`) | ❌ **откатилось** | файл вернулся в до-аудитное состояние; смягчается отключением offline-OpenIE и guarded-импортами (§3.4) |
| **S-5** | 🟡 rerank: misleading-лог, фолбэк пустыми списками | ✅ **закрыто и улучшено** | настоящий fallback к порядку скоростей (`rerank.py:155–159`), лог честный (§3.5) |
| **S-6** | 🟢 форк отставал от upstream на 10 коммитов | ➖ **неактуально** | репо отсоединено от upstream-сети (`docs: record repo switch to private…`), развивается самостоятельно |
| **S-7** | 🟢 отсутствие CI | ✅ **закрыто** | полный `ci-cd.yml`: lint + тесты (ubuntu+windows matrix) + eval-smoke + index-fixture; pre-commit; локально всё зелёное (§3.7) |

| # | Новый дефект (потерянный фикс `9e881f8`) | Приоритет | Статус |
|---|---|---|---|
| **R-1** | `StandardRAG.rag_qa` → `AttributeError: 'StandardRAG' object has no attribute 'prompt_template_manager'` | 🔴 критический | открыто (§4.1) |
| **R-2** | мёртвый кэш эмбеддингов запросов: каждый `retrieve()` перекодирует запросы | 🟡 средний | открыто (§4.2) |
| **R-3** | `transformers_llm.infer` декодирует промпт+ответ, `completion_tokens` завышены | 🟡 средний | открыто (§4.3) |
| **R-4** | misleading-лог «No facts found after reranking» в цикле `retrieve()` | ⬜ косметика | открыто (§4.4) |

---

## 2. Контекст: что изменилось между проверками

За период 31.10 → 04.11 репозиторий прошёл путь от форка HippoRAG к самостоятельному продукту:

- **+130 коммитов** (`9e881f8..30f4304`), **272 файла, +37 819 / −2 525 строк**;
- отсоединение от upstream-сети форков, собственный роадмап и цикл «аудит → исправления» (S-01..S-45, раунды ревью, protected main);
- вокруг `src/hipporag` построено приложение: `src/astro_app` (23 режима), `search_api` (FastAPI + аутентификация), `pdf_ingestion`, `vector_stores`, CLI-меню;
- `setup.py`/`requirements.txt` заменены на `pyproject.toml` как единый источник зависимостей (`requirements.txt` теперь указатель);
- появилась полноценная CI и 268 unit-тестов.

Важно для настоящего отчёта: **`src/hipporag` при этом пересобран из другой базы** (слияние `1fb7418` «Merge fork/main into main»), поэтому фиксы коммита `9e881f8` в пакете сохранились не все — отсюда регрессии R-1…R-3.

---

## 3. Проверка закрытия находок аудита от 31.10

### 3.1 S-1 (NameError в `StandardRAG.__init__`) — ✅ закрыто

Коммит `9e881f8` добавил в `__init__` блок `if embedding_base_url is not None:` без параметра в сигнатуре — любая конструкция падала. В текущем main блок удалён целиком, сигнатура чистая (`StandardRAG.py:24–33`). Проверка живьём:

```
rag = StandardRAG(global_config=cfg)        # cfg: save_dir=tmp, llm_base_url=localhost, openie_mode=offline
→ OK, rag.ready_to_retrieve is False
```

Есть регрессионный тест `tests/unit/test_core_utils.py::test_standard_rag_instantiates_offline`. Дефект не воспроизводится.

### 3.2 S-2 (импорт без vllm) — ✅ закрыто

Раньше цепочка `StandardRAG.py → openie_vllm_offline.py → vllm_offline.py → from vllm import …` роняла импорт на системах без vllm. Сейчас:

- из `StandardRAG.py` импорт `openie_vllm_offline` убран полностью (импорты файла: `StandardRAG.py:1–18`);
- в `HippoRAG.py:18–26` оба offline-OpenIE обёрнуты в `try/except ImportError` с деградацией в `None`;
- vllm вынесен в extra `offline` (`pyproject.toml:67–70`) с пометкой «Offline-режим OpenIE отключён в этой сборке».

Проверка живьём в окружении **без vllm**:

```
import hipporag                       → OK
import hipporag.StandardRAG           → OK
from hipporag import HippoRAG         → OK
```

### 3.3 S-3 (необъявленные зависимости) — ⚠️ частично

- ✅ `sentence-transformers>=2.2.0` объявлен (`pyproject.toml:35`) — нужен `embedding_model/Transformers.py`;
- ✅ `pip install -e ".[dev]"` в чистое окружение проходит (проверено) — сама установка по декларации починена;
- ❌ **`outlines` по-прежнему не объявлен** ни в `pyproject.toml`, ни в `requirements.txt`, при этом импортируется на уровне модуля: `llm/transformers_offline.py:3–4` (`import outlines.generate`, `import outlines.models`).

Последствия смягчены guarded-импортами: `import hipporag` без outlines работает (проверено), а `hipporag.llm.transformers_offline` даёт `ModuleNotFoundError: outlines` только при явном запросе бэкенда. Но раз бэкенд остался в коде, зависимость должна быть объявлена (например, в extra `offline` рядом с vllm) — иначе «pip install по декларации» не даёт работоспособный `Transformers/…`-LLM.

### 3.4 S-4 (баги `transformers_offline`) — ❌ откатилось

Файл вернулся в состояние **до** аудита:

- `infer()` (`transformers_offline.py:61–71`): `self.model.generate(input_ids, max_new_tokens=…)` → `transformers_output[0]["generated_text"]` — vLLM-стиль доступа к результату HF `generate()` (возвращает тензор) → `TypeError` на первом же вызове; метрики токенов считаются от полного текста;
- free-form путь `batch_infer` (`:94–100`): в `generate()` передаются **сырые строки** и параметр `max_tokens` (вместо `max_new_tokens`), результат затем безусловно прогоняется через `completion.model_dump_json()` (`:101`) — падение и здесь;
- guided-путь (outlines) корректен.

Смягчающие обстоятельства: offline-OpenIE в этой сборке отключён (комментарий в `pyproject.toml:68`), импорты защищены (S-2), в CI бэкенд не выполняется. Но при «реанимации» offline-режима (заявленная цель extra `offline`) дефекты всплывут немедленно. Фиксы из `9e881f8` (декод только сгенерированной части; токенизация входов + `max_new_tokens` в batch-пути) стоит вернуть.

### 3.5 S-5 (rerank: фолбэк и misleading-лог) — ✅ закрыто и улучшено

В `9e881f8` при сбое LLM лог говорил «falling back to unreranked facts», а фактически возвращались пустые списки. Сейчас реализован настоящий фолбэк — к порядку скоростей (кандидаты приходят предсортированными по score):

```python
# rerank.py:146
logger.warning(f"LLM rerank call failed, falling back to score order: {e}")
...
# rerank.py:155–159
if not result_indices and candidate_items:
    result_indices = list(range(min(len(candidate_items), len_after_rerank or len(candidate_items))))
```

Дополнительно добавлен guard `if not candidate_items: continue` в цикле сопоставления. Замечаний нет.

### 3.6 S-6 (отставание от upstream) — ➖ неактуально

Репозиторий отсоединён от сети форков и переведён в собственный цикл разработки; синхронизация с `OSU-NLP-Group/HippoRAG` более не является требованием. Пункт снят с учёта.

### 3.7 S-7 (отсутствие CI) — ✅ закрыто

`.github/workflows/ci-cd.yml`: lint-джоба (black, isort, ruff, mypy advisory) + тест-матрица **ubuntu + windows** на Python 3.12 + `eval-smoke` (10 вопросов, без LLM) + `index-fixture` (BM25 smoke по фикстуре, честная заглушка, если фикстура не в репо). Плюс `.pre-commit-config.yaml`. Воспроизведение локально:

```
black --check src/ tests/ main.py   → All done! 168 files would be left unchanged.
isort  --check-only src/ tests/ main.py → All checks passed!
ruff   check src/ tests/ main.py    → All checks passed!
```

---

## 4. Новые дефекты: потерянные фиксы коммита `9e881f8`

### 4.1 R-1 (🔴 критический): `StandardRAG.rag_qa` падает на каждом вызове

`qa()` использует менеджер шаблонов (`StandardRAG.py:330` и `:342`):

```python
if self.prompt_template_manager.is_template_name_valid(name=f"rag_qa_{self.global_config.dataset}"):
...
self.prompt_template_manager.render(name=f"rag_qa_{prompt_dataset_name}", prompt_user=prompt_user)
```

но в `__init__` (`StandardRAG.py:24–95`) атрибут **не создаётся**. Это ровно тот дефект, который чинил и `9e881f8`, и upstream. Воспроизведение живьём (валидные `QuerySolution`, LLM застаблен):

```
res = rag.rag_qa(queries=[QuerySolution(question="What is X?", docs=["doc one", "doc two"])])
→ AttributeError: 'StandardRAG' object has no attribute 'prompt_template_manager'
```

Итог: retrieval (`StandardRAG.retrieve`) работает, а QA — нет; `main.py`-путь со `--rag_type standard` на QA-этапе сломан. Тест `test_standard_rag_instantiates_offline` покрывает только конструкцию, поэтому сьют зелёный.

**Фикс (одна строка в `__init__`, по образцу `HippoRAG`):**

```python
from .prompts.prompt_template_manager import PromptTemplateManager
...
self.prompt_template_manager = PromptTemplateManager(
    role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
)
```

и юнит-тест на `rag_qa` со стабом `llm_model.infer` — чтобы дефект больше не возвращался.

### 4.2 R-2 (🟡 средний): мёртвый кэш эмбеддингов запросов

`get_query_embeddings` (`StandardRAG.py:407–418`) снова проверяет **оба** под-словаря, но заполняется только `'passage'` (`:420–426`) — `'triple'` не заполняется никем. Условие `query not in ...['triple'] or query not in ...['passage']` истинно всегда → кэш никогда не попадает, каждый `retrieve()` перекодирует запросы. В `9e881f8` это было исправлено (проверять только `'passage'`).

Воспроизведение живьём (счётчик вызовов `batch_encode`, один и тот же запрос дважды):

```
retrieve(["q1"]); retrieve(["q1"])  → batch_encode вызван 2 раза (ожидался 1)
```

На больших корпусах с повторными запросами это заметный лишний расход (для платных API — денег). Фикс: проверять и заполнять только `'passage'` (в `StandardRAG` факт-эмбеддинги не используются).

### 4.3 R-3 (🟡 средний): `transformers_llm.infer` декодирует промпт+ответ

`llm/transformers_llm.py:149–153` вернулся к `decode(response[0])` и `completion_tokens = response.shape[1]` — то есть в «ответ» попадает и промпт, и метрики завышены. В `9e881f8` исправлено вычитанием длины промпта. Воспроизведение живьём со стабами токенизатора/модели (промпт 4 токена, сгенерировано 2):

```
message = 'WRONG:[1, 2, 3, 4, 5, 6]'   # ожидалось 'ANSWER' (только [5, 6])
completion_tokens = 6                   # ожидалось 2
```

Фикс из `9e881f8`: `prompt_len = params["prompt_text"].shape[1]`; `decode(response[0][prompt_len:])`; `completion_tokens = response.shape[1] - prompt_len`.

### 4.4 R-4 (⬜ косметика): misleading-лог в цикле ретрива

`StandardRAG.py:175`: `logger.info("No facts found after reranking, return DPR results")` внутри цикла по запросам — у StandardRAG нет ни фактов, ни реранка; лог вводит в заблуждение при чтении журналов. В `9e881f8` строка уже удалялась. Рекомендация: удалить.

---

## 5. Прогон тестов и lint

### 5.1 Тест-сьют

Команда CI — `pytest tests/unit -m unit`:

```
первый прогон:  217 passed, 51 errors   ← PermissionError: [WinError 5] Temp\pytest-of-ALEXEY
повтор:         268 passed in 2.29s     (--basetemp=<свежая папка>)
```

Все 51 ошибок первого прогона — один и тот же `PermissionError` на базовой tmp-директории pytest (устаревшая папка с чужими правами на конкретной машине). К репозиторию отношения не имеет; на CI-раннерах не воспроизводится. Для локальных прогонов на этой машине — задавать `--basetemp`.

### 5.2 Пробелы покрытия, объясняющие зелёный сьют при красном R-1…R-3

В `tests/unit` отсутствуют проверки на:

- `rag_qa`-путь `StandardRAG` (есть только тест конструкции);
- кэш `get_query_embeddings` (повторный `retrieve` не должен перекодировать);
- decode-спан и метрики `transformers_llm.infer`;
- guarded-импорты без vllm/outlines (сценарий «чистая установка, offline не нужен»);
- фолбэк реранка к score-order.

Рекомендуется закрыть хотя бы первые три — это дешевле, чем повторная потеря фиксов при следующем слиянии.

---

## 6. Положительные изменения, отмеченные при проверке

- `utils/misc_utils.py:18 parse_fact_content()` — грамотная эволюция security-фикса из `9e881f8`: `ast.literal_eval` основной (факты хранятся как `str(tuple)`), `json.loads` — fallback, с внятным комментарием о том, почему `json.loads` ломал графовый путь поиска;
- OpenIE-парсинг (`information_extraction/openie_openai.py:38–49`) — robust-схема «строгий `json.loads` + regex-fallback», безопаснее и толерантнее одновременно;
- `requirements.txt` → указатель на `pyproject.toml` как единый источник зависимостей;
- документированная, защищённая main-ветка с обязательными checks и pre-commit.

---

## 7. Рекомендации (приоритизированные)

| Приоритет | Действие | Оценка трудозатрат |
|---|---|---|
| 🔴 P0 | R-1: инициализировать `prompt_template_manager` в `StandardRAG.__init__` + юнит-тест на `rag_qa` со стабом LLM | минуты |
| 🟠 P1 | R-2: проверять/заполнять только `'passage'` в `get_query_embeddings` + тест на отсутствие перекодирования | минуты |
| 🟠 P1 | R-3: вернуть decode-only-completion и честные `completion_tokens` в `transformers_llm.infer` + тест со стабом | минуты |
| 🟠 P1 | S-3: объявить `outlines` (в extra `offline` рядом с vllm) | строка |
| 🟡 P2 | S-4: при реанимации offline-режима вернуть фиксы `transformers_offline` из `9e881f8` (tensor-входы, `max_new_tokens`, декод только завершения) | ~час |
| ⬜ P3 | R-4: удалить misleading-лог из `retrieve()` | строка |
| ⬜ P3 | Добавить в сьют тесты на guarded-импорты и фолбэк реранка | ~час |

---

## Приложение: методика и окружение

- **Окружение:** Windows 10 x64, Python 3.12.3 (venv), `torch==2.5.1+cpu`, `transformers==4.45.2`, `sentence-transformers 5.7.0`; проект установлен как `pip install -e ".[dev]"`; `vllm` и `outlines` в окружении **отсутствуют** (используется как условие проверок S-2/S-3).
- **Проверяемое состояние:** `origin/main` = `30f4304`, рабочая копия чистая.
- **Воспроизведение статусов §3–§4:** import-пробы `python -c "import hipporag.StandardRAG"`; для R-1/R-2 — вызовы `StandardRAG.rag_qa`/`retrieve` со стабами `llm_model.infer` и `embedding_model.batch_encode`; для R-3 — `TransformersLLM.__new__` + стабы токенизатора/модели/кэша; выводы по коду — построчный разбор `StandardRAG.py`, `rerank.py`, `llm/transformers_llm.py`, `llm/transformers_offline.py`, `pyproject.toml` на коммите `30f4304`.
- **Сопутствующие артефакты проверки** (вне репозитория): скрипты точечных проверок `verify_1104.py`, `verify_cache_1104.py`.
