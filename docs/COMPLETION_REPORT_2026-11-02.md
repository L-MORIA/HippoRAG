# Отчёт о завершении: переиндексация, слияние, пуш

**Дата:** 2026-11-02
**Исполнитель:** Mavis (mavis)
**Репозиторий:** `L-MORIA/HippoRAG` (личный fork) · референс `osu-nlp-group/hipporag` не затрагивался

---

## 1. Результат

| Этап | Статус | Комментарий |
|------|--------|-------------|
| Диагностика переиндексации | ✅ | Исполнитель **успел** — `reindex_run.log`: `INDEX EXIT: 0` |
| Все 5 критериев приёмки | ✅ 6/6 | `scripts/verify_all_mavis.py` (5 критериев + контроль графового пути) |
| `git add -A` + коммит | ✅ | `54f5486` |
| Слияние `fork/main` | ✅ | `1fb7418` (merge-коммит) |
| Тесты на слитом коде | ✅ | pytest 62 passed; ruff/black/isort/compileall — чисто |
| `git push fork main` | ✅ | fast-forward `9e881f8..1fb7418` → `L-MORIA/HippoRAG` |
| Откат на бэкап | — | не потребовался (все критерии PASS) |

Итоговое состояние после пуша:

```
$ git rev-list --left-right --count main...fork/main
0    0                       ← синхронизировано
$ git rev-list --left-right --count main...origin/main
3    0                       ← локально 3 коммита впереди референса (не пушились)
```

`origin/main` = `c617143` — **не изменён**. PR в upstream не создавался.

---

## 2. Что оказалось недоделанным у исполнителя

**Недоделан был только git-коммит.** Переиндексация — операционный шаг, занявший ~6 часов, — **завершилась успешно**, хотя исполнитель исчерпал лимит на этапе упаковки в git:

```
INFO:hipporag.HippoRAG:Graph construction completed!
INFO:hipporag.HippoRAG:Graph info: {'num_phrase_nodes': 6528, 'num_passage_nodes': 1369,
  'num_total_nodes': 7897, 'num_extracted_triples': 6847, 'num_synonymy_triples': 96048,
  'num_total_triples': 103368}
INFO:hipporag.HippoRAG:Writing graph with 7897 nodes, 736089 edges
INFO:__main__:Индексация успешно завершена!
INDEX EXIT: 0
```

Итоговый индекс (было → стало):

| | было | стало |
|---|------|-------|
| `chunk_embeddings` | 1422 | **1369** |
| `entity_embeddings` | 6979 | **6528** |
| `fact_embeddings` | **1** («placeholder») | **6847** |
| граф | — | 7897 узлов / 736 089 рёбер |
| synonymy-рёбра | 0 | **726 250** (entity↔entity) |

---

## 3. Дефекты, найденные и исправленные при приёмке

### D-A. Баг в самом скрипте верификации (блокировал гейт)

`scripts/verify_reindex.py:67-70` — `names = set(g.vs["name"])`, затем `names[g.es[e.index].source]` → `TypeError: 'set' object is not subscriptable`. Скрипт падал на критерии 2.

Исправлено на `g.vs[e.source]["name"]` (индексы вершин, а не поиск по set).

### D-B. Регрессия: графовый путь поиска был отключён (Critical, тихая)

В рабочем дереве факты разбирались так:

```python
# src/hipporag/HippoRAG.py:1980 (было)
candidate_facts = [json.loads(fact_row_dict[id]["content"]) for id in real_candidate_fact_ids]
```

Но факты **хранятся** как `str(tuple)` (`HippoRAG.py:319`: `insert_strings([str(fact) for fact in facts])`), то есть это Python-repr, а не JSON. Проверено на данных: **все 6847 фактов** не проходят `json.loads`:

```
content: "('экзальтация', 'helps with', 'экстремальные ситуации')"
→ JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Цепочка отказа:** `json.loads` → `JSONDecodeError` → `except Exception` в `rerank_facts` → возврат `[], []` → `retrieve()` уходит в `dense_passage_retrieval` → **PPR по фактам (графовая часть HippoRAG) не работает ни разу**, без единой ошибки наверх.

Почему это не поймали прежние проверки:
- исключение глушится `logger.error` + graceful fallback → «работает, но деградировано»;
- критерий 5 проверяет лишь наличие цитат в ответе, а не задействован ли граф;
- в `fork/main` (коммит `9e881f8`) стоял корректный `ast.literal_eval` — регрессия возникла в незакоммиченном дереве исполнителя.

Исправлено: добавлена `parse_fact_content()` в `src/hipporag/utils/misc_utils.py` — `ast.literal_eval` (безопасная альтернатива `eval()`) с fallback на `json.loads` и предупреждением при неудаче; вызывается из `rerank_facts`.

**Контроль в приёмке:** добавлена 6-я проверка «графовый путь работает» — перехват логгера `hipporag.HippoRAG` и проверка отсутствия `Error in rerank_facts`. До фикса — падала, после — PASS.

### D-C. `.env` не игнорировался и попал бы в коммит

`gitignore:23` — `.env# Артефакты инструментов и локальные данные…` (нет перевода строки, шаблон `.env` «съел» комментарий). `git check-ignore .env` → exit 1.

Сейчас `.env` содержит только `LLM_NAME`/`LLM_BASE_URL` (секретов нет), но при `git add -A` файл попал бы в репозиторий. Исправлено: `.env` на отдельной строке, `git check-ignore .env` → exit 0.

### D-D. Шумовые diffы датасетов

8 файлов `reproduce/dataset/*.json` отличались от HEAD только переводами строк (проверено: `text_equal=True`, `json_equal=True` для всех). Откачены — коммит не содержит 8 гигантских однострочных диффов в данных.

---

## 4. Слияние с `fork/main`: что потребовало решения

Сухая проверка (`git merge-tree --write-tree HEAD fork/main`) дала exit 0, но реальное слияние дало **22 конфликтных файла**: upstream за 15 коммитов добавил векторные бэкенды (`qdrant`/`chroma`/`milvus`) и переработал `embedding_store` в `BaseEmbeddingStore` + фабрику.

**Принцип разрешения:** база — аудированный код (все 45 находок и их фиксы), сверху точечно переинтегрированы upstream-добавления:

| Из upstream | Куда | Статус |
|---|---|---|
| `BaseEmbeddingStore` (ABC) + `get_embedding_store()` | `embedding_store.py` — аудированная `EmbeddingStore` теперь наследует ABC | ✅ |
| `Chunk`, `RetrievalResult` | `utils/misc_utils.py` | ✅ |
| `vector_store_type`, `qdrant_*`, `chroma_*`, `milvus_*`, `bedrock_*` | `utils/config_utils.py` | ✅ |
| экспорт `Chunk`, `RetrievalResult` | `__init__.py` | ✅ |
| `reason_step()` | уже был в аудированной версии (с фиксом D-5) | ✅ |
| новые файлы: `vector_stores/`, `preprocessing.py`, `bedrock_mantle.py`, `state_utils.py`, `sample_data.py`, `ircot_2wikimultihopqa.py`, `examples/`, `tests/integration/` | — | ✅ |
| `main_azure.py`, `main_dpr.py` | удалены (upstream удалил в коммите безопасности) | принято удаление |
| `setup.py` | удалён (зависимости в `pyproject.toml`) | принято удаление |

**Критический момент:** upstream-овый `EmbeddingStore.insert_strings(texts)` **не принимает метаданные** — взять его файл было нельзя, сломалась бы вся механика цитат (S-12). Поэтому взята аудированная реализация, а ABC добавлен сверху.

Upstream-файлы не проходили линтеры (upstream не форматировал): применены `ruff --fix`, `black`, `isort` → 8 F401/F541 устранены, 16 файлов отформатированы.

---

## 5. Приёмка на слитом коде

```
pytest tests/unit          → 62 passed
ruff check src tests       → All checks passed
black --check src/ tests/  → 123 files unchanged
isort --check-only         → exit 0
compileall src/hipporag    → exit 0

scripts/verify_all_mavis.py → ИТОГ: 6/6
  [PASS] 1: facts > 100 — facts = 6847
  [PASS] 2: synonymy-рёбра > 0 — entity↔entity = 726250 из 736089
  [PASS] 3: /health facts и без предупреждения — chunks=1369, facts=6847
  [PASS] 4: масштабы сопоставимы — entities=6528, facts=6847, chunks=1369 (перекос x5.0)
  [PASS] 5: /search на русском + citations[].sources — HTTP 200, sources[0]=['Эзотерический фэншуй.pdf']
  [PASS] Г: графовый путь поиска работает (rerank_facts без ошибок)
```

---

## 6. Коммиты

| Коммит | Содержание |
|--------|------------|
| `54f5486` | 45 находок аудита (S-01…S-45), инфраструктура (pyproject/CI/pre-commit/.gitignore), 62 юнит-теста, `scripts/verify_all_mavis.py`, фикс `parse_fact_content` |
| `1fb7418` | Merge `fork/main` + реинтеграция векторных бэкендов и конфиг-полей upstream |

Пуш: `9e881f8..1fb7418  main -> main` → `https://github.com/L-MORIA/HippoRAG.git` (fast-forward).

---

## 7. Остаточные замечания (не блокеры)

| # | Что | Статус |
|---|-----|--------|
| 1 | Векторные бэкенды (qdrant/chroma/milvus) интегрированы на уровне API, но `HippoRAG` по-прежнему создаёт `EmbeddingStore` напрямую — переключение `vector_store_type` требует отдельной правки | низкий |
| 2 | 16 голых `except:` и ~600 `print()` в корневых скриптах — гигиена S-26/S-34 была по `src/` | низкий |
| 3 | mypy — 208 замечаний (advisory в CI) | низкий |
| 4 | `main.py` — 3140 строк (развилка №5, отложена осознанно) | низкий |
| 5 | pre-commit-хук не установлен (`.git/hooks/pre-commit` отсутствует) — нужен `pre-commit install` | средний |
| 6 | `src/hipporag/utils/typing.py` удалён в аудите, но upstream его ещё импортирует в части кода — проверено, что активный путь его не использует | низкий |

---

## 8. Рекомендация

Проект готов к боевому использованию: индекс полон и функционально корректен, все 45 находок аудита закрыты, CI-ворота зелёные, изменения в личном репозитории.

Перед следующим циклом: `pre-commit install` (хук не установлен) и отдельная задача на подключение `vector_store_type` к `HippoRAG`, если векторные бэкенды действительно нужны.
