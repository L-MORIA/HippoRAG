# План разделения main.py на модули

**Дата:** 2026-11-02
**Статус:** v2 — правки по рецензии [`MAIN_SPLIT_PLAN_REVIEW.md`](./MAIN_SPLIT_PLAN_REVIEW.md) внесены, готов к выполнению
**Объект:** `main.py` — 4070 строк, 72 функции, 23 режима поиска + QA-кэш + меню

> **Журнал правок по рецензии** (все три утверждения рецензии проверены исполнителем независимо — AST-разбор, `git check-ignore`, содержимое `__editable__.hipporag-2.0.0a4.pth` — и подтверждены):
>
> | № | Возражение | Правка в v2 |
> |---|-----------|-------------|
> | 1 (блок.) | пакет в корне не попадает в editable-установку и wheel (`.pth` добавляет только `src/`; auto-discovery src-layout) | пакет перенесён в **`src/astro_app/`** (§2) |
> | 2 (блок.) | `data/synonyms.json` под `.gitignore` (`data/`) → тихое деградирование keyword-expansion во всех 23 режимах на свежем клоне | словарь внутри пакета: **`src/astro_app/data/synonyms.json`** + `[tool.setuptools.package-data]` (§2, §4.1) |
> | 3 (сущ.) | единая сигнатура `fn(hipporag, query, top_k=10)`: 5 функций `top_k` не принимают (partial → TypeError), у `search_self_rag` дефолт 30, у `search_semantic_filter` — 50 (тихая урезка выдачи) | контракт вызова **`fn(hipporag, query)`** — дефолты функций сохраняются; `top_k` — поле реестра, передаётся только если задано; для 5 функций — адаптеры (§4) |
> | 4 | оценка 1,5–2 дня занижена (полный smoke 23 режимов ≈ 12–20 мин × 7 шагов) | smoke **только по затронутым режимам** на каждом шаге, полный снапшот — на базлайне и финале; оценка **3–5 дней** (§5) |
> | 5 | имя `astro_search` вводит в заблуждение (внутри ещё QA-кэш и CLI) | пакет переименован в **`astro_app`**; если владельцу милее прежнее имя — переименование на шаге 7 без функциональных последствий |

---

## 1. Цели и принципы

1. **main.py становится тонкой точкой входа** (~50 строк): парсинг аргументов + вызов меню.
2. **Единый реестр режимов** вместо `if mode == "1" ... elif mode == "23"`: декларативная таблица
   `MODES: dict[str, ModeSpec]`. Меню рисуется из реестра — добавление режима = одна запись,
   а не правка в 4 местах (меню, `_qa_run_test`, «тестирование всех методов», документация).
3. **Контракт вызова режима — `fn(hipporag, query)`**: каждый режим сохраняет СВОИ дефолты
   (включая `top_k=30` у Self-RAG и `top_k=50` у semantic filter). Реестр передаёт `top_k`
   только если он явно указан в `ModeSpec` (§4). Возражение 3 рецензии закрыто конструкцией,
   а не осторожностью.
4. **Ленивая загрузка тяжёлых моделей сохраняется** (`lru_cache`-синглтоны `_get_crossencoder`,
   `_get_e5`, `_get_colbert`, `_get_qwen`…) — переезжают в свои модули без изменений логики.
5. **Каждый шаг миграции = отдельный коммит + зелёные проверки** (`pytest tests/unit` — 110,
   py_compile, ruff, black/isort, smoke по затронутым режимам). Сломанный промежуточный
   коммит недопустим.
6. Импорты в стиле `from hipporag import ...` (единый namespace, S-13).
7. **Measure-first:** любое проектное решение в этом плане снабжено измерением
   (пакет/установка — §2 и приложение рецензии; gitignore — `git check-ignore`;
   сигнатуры — AST-разбор). Новые решения перед внесением в план проверяются так же.

## 2. Целевая структура

```
main.py                      # ~50 строк: точка входа → astro_app.cli.main()

src/astro_app/               # пакет внутри src/ — попадает в editable-установку
  __init__.py                #    И В wheel (auto-discovery src-layout подхватит
  config.py                  #    hipporag И astro_app); Возражение 1 закрыто
                             # константы (SAVE_DIR, LLM_*, пути), STOP_WORDS,
                             # SENTENCE_START, load_hipporag(), инициализация env
  data/
    synonyms.json            # SYNONYM_MAP как данные ВНУТРИ пакета (Возражение 2):
                             # попадает в git, wheel и sdist; не конфликтует с
                             # data/ корпуса (та остаётся в .gitignore)
  synonyms.py                # load_synonyms(): читает data/synonyms.json
                             # через importlib.resources — работает и из wheel
  query.py                   # _expand_query, _word_re, _is_meta_reformulation,
                             # reformulate_query, split_query_terms                (строки 417–695)
  textutils.py               # show_text_safe, fix_chunk_start, _tokenize          (665–695, 819–833)
  primitives.py              # _build_bm25, _embedding_scores, _get_chunk_meta     (823–852, 1143–1166)
  rerankers.py               # _get_crossencoder(_ru), _rerank_with_crossencoder*,
                             # _get_e5, _get_colbert, _colbert_*                   (853–924, 1012–1022, 1529–1596)
  display.py                 # show_results                                        (3287–3344)
  qa_cache.py                # _qa_load/_qa_save/_qa_lookup/_qa_store/_qa_norm/
                             # _qa_derive_short/_qa_run_test/_qa_export_training   (3846–4070)
  modes/                     # 23 режима — по группам, контракт fn(hipporag, query)
    __init__.py              # сборка реестра MODES из подмодулей
    dense.py                 # search_multi_vector, search_mxbai, search_e5_rerank      (697–762, 925–1011, 1023–1142)
    bm25_hybrid.py           # search_bm25_hybrid(_ru), search_rrf_context              (1167–1324)
    graph.py                 # search_hipporag_graph, search_graphrag_agent             (763–818, 2044–2236)
    small_to_big.py          # _build_sign_index, _sign_in_query, _count_sign,
                             # _keyword_boost, _chunk_sign_score,
                             # search_small_to_big(_ru)                                 (1325–1528, 1697–1804)
    llm_filter.py            # _filter_by_section, _llm_filter_results,
                             # search_with_llm_filter, _get_qwen, _get_frida,
                             # search_qwen_frida                                        (1805–2043)
    sentence_dat.py          # _is_bibliographic_query, search_sentence_level,
                             # _dat_evaluate/_compute_alpha/_combine_results,
                             # search_dat_sentence_llm                                  (2237–2562)
    self_rag.py              # search_self_rag                                          (2563–2708)
    hyde.py                  # search_hyde                                              (3169–3286)
    sign_zones.py            # extract_sign_full/polished/summary, _sentence_index,
                             # _zones_in_text, _sign_zone_class, _find_sign_zones,
                             # _find_zone_signs, _sign_zone_answer,
                             # search_semantic_filter                                   (2709–3046, 3598–3845)
    filtered.py              # search_filtered                                          (3047–3168)
  cli.py                     # main(): меню из реестра MODES, обработка t/e/q,
                             # show_results, цикл ввода                                 (3345–3597)
```

Ссылки на строки — по текущему файлу (4070 строк); при миграции «плывут» на ±10 строк.

**Сопровождающая правка `pyproject.toml`** (иначе данные не попадут в дистрибутив —
смежная с Возражением 2 дыра, закрытая проактивно):

```toml
[tool.setuptools.package-data]
astro_app = ["data/*.json"]
```

## 3. Порядок миграции (7 шагов, каждый = коммит + зелёные проверки)

| Шаг | Что переезжает | Почему первым/в таком порядке |
|-----|----------------|-------------------------------|
| 1 | Каркас `src/astro_app/` + `config.py` + `synonyms.{py,json}` (+ package-data в pyproject) | Фундамент и данные; сразу фиксирует правила упаковки, чтобы дальнейшие шаги не приходилось переделывать |
| 2 | `textutils.py` + `display.py` | Утилиты без зависимостей от режимов |
| 3 | `qa_cache.py` | Самодостаточная подсистема (кэш/экспорт), используется меню t/e |
| 4 | `query.py` + `primitives.py` | Общий фундамент для всех режимов |
| 5 | `rerankers.py` | Тяжёлые ленивые синглтоны в одном месте; режимы начнут импортировать отсюда |
| 6 | `modes/*` — **группами** (dense → bm25 → graph → small_to_big → llm_filter → sentence_dat → self_rag/hyde → sign_zones → filtered), после каждой группы: smoke **только затронутых режимов** | Режимы зависят только от primitives/rerankers — мигрируют независимо |
| 7 | `cli.py` + новый `main.py` + реестр `MODES` + **полный снапшот-сравнение 23 режимов** | Последним: меню переключается на реестр, старый if/elif удаляется |

## 4. Реестр режимов (шаг 7)

```python
# src/astro_app/modes/__init__.py
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ModeSpec:
    label: str
    fn: Callable            # вызывается как fn(hipporag, query) — СВОИ дефолты каждого режима
    star: bool = False      # ⭐ рекомендованный
    special: str = ""       # предупреждение («нужен знак или зона в запросе»)
    top_k: int | None = None  # None → НЕ передавать, работает дефолт функции
                              # (search_self_rag=30, search_semantic_filter=50 — сохраняются)

MODES: dict[str, ModeSpec] = {
    "1": ModeSpec("Dense (multi-vector)", dense.search_multi_vector),
    ...
    "14": ModeSpec("Self-RAG", self_rag.search_self_rag, top_k=30),  # фиксируем текущий дефолт явно
    "22": ModeSpec("Pattern Search: знак ↔ зона", sign_zones.search_semantic_filter,
                   special="нужен знак или зона в запросе, иначе честный отказ", top_k=50),
}
```

Правила вызова в `cli.py` (единственные допустимые формы):

```python
spec = MODES[mode]
results = spec.fn(hipporag, query) if spec.top_k is None else spec.fn(hipporag, query, top_k=spec.top_k)
```

**Адаптеры для 5 функций без `top_k`** (AST-проверено: `search_with_llm_filter`,
`search_graphrag_agent`, `extract_sign_full`, `extract_sign_polished`,
`extract_sign_summary` — принимают только `(hipporag, query)`; `partial` с `top_k`
дал бы TypeError):

```python
def _mode_graphrag(h, q):        # тонкие адаптеры рядом с функциями в их модулях
    return search_graphrag_agent(h, q)
```

- Меню печатает таблицу из `MODES` (номер, label, ⭐/⚠) — правка «пометить режим 22/23
  специализированным» (коммит 086daf1) превращается в поле `special`.
- `_qa_run_test` «Тестирование всех методов» проходит по `MODES` — дублирующий список исчезает.
- Дополнительные параметры, которые функция РЕАЛЬНО принимает (например, `candidate_pool`
  у режимов dense/bm25), — `functools.partial` допустим: он передаёт то, что есть в сигнатуре.

## 5. Приёмка

1. **Базлайн ДО первого шага:** снапшот выдачи всех 23 режимов на контрольном запросе
   («как обустроить прихожую для Овна») → `outputs/refactor_baseline/` (JSON: режим → docs/citations).
2. **После каждого шага:** `pytest tests/unit` (110), `compileall`, ruff/black/isort (как в CI),
   smoke **затронутых** режимов шага со сравнением с базлайном (измеренная стоимость:
   extract_sign_polished ≈ 2 мин, search_qwen_frida ≈ 2 мин, LLM-режимы 10–42 с).
3. **Финал (шаг 7):** полный снапшот 23 режимов и сравнение с базлайном.
4. **Упаковка:** `pip wheel . --no-deps` → в wheel присутствует `astro_app` + `data/synonyms.json`
   (проверка Возражений 1–2 на артефактах, не на веру).
5. Точки входа не меняются: `hipporag.bat`, `menu.py`, `test_menu.bat` продолжают работать;
   импорт `astro_app` работает из любого cwd (editable-установка уже прописывает `src/`).

## 6. Риски

| Риск | Митигация |
|------|-----------|
| Лямбды/замыкания в меню теряют контекст при переносе | Режимы — функции верхнего уровня, а не лямбды (заодно закрывает B023-класс) |
| Дублирование констант при миграции группами | Константы только в `config.py`; каждый шаг — grep на остатки в main.py |
| Случайное изменение поведения поиска | Снапшот-базлайн (§5.1) и сравнение после каждого шага (§5.2) |
| Данные вне пакета / под gitignore | Словарь внутри `src/astro_app/data/` + package-data; проверка wheel на артефакте (§5.4) |
| Сигнатурные несовместимости режимов | Контракт `fn(hipporag, query)` + `top_k` в реестре + адаптеры; AST-сверка сигнатур на шаге 6 |
| Батч-правка «всё сразу» | Строго по шагам 1–7, один шаг = один коммит |

**Оценка:** 3–5 рабочих дней (smoke по затронутым режимам на каждом шаге, полный снапшот
на базлайне и финале; измерения длительности режимов — из рецензии, §4.1).

---

## 6а. Журнал выполнения (факт)

| Шаг | Статус | Верификация |
|-----|--------|-------------|
| 0. Базлайн-снапшот 23 режимов | ✅ | `outputs/refactor_baseline/pre/` (23/23 ok, manifest с git-HEAD) |
| 1. `src/astro_app/` каркас + synonyms.json в пакете + package-data | ✅ d258b1a | wheel содержит `astro_app` + `data/synonyms.json`; load_synonyms() == main.SYNONYM_MAP (66 ключей) |
| 2. textutils + display | ✅ d051179 | эквивалентность stdout на реальных данных всех 23 режимов |
| 3. qa_cache | ✅ 9e25073 | реальный кэш (6 записей) + запись/экспорт во временных копиях |
| 4. query + primitives | ✅ 674b275 | BM25 1369 скоров идентичны; _get_chunk_meta 50/50; split/expand идентичны |
| 5. rerankers | ✅ 2b5284a | CE-en/ru идентичны; ColBERT bit-exact (веса идентичны, max_diff=0) |
| 6. modes/* (11 модулей) + signs.py | ✅ 63d9ee6→dd9dd3b | пост-снапшот на слитом коде: **21/23 EXACT**; 18/23 — HIT кэша (задуманное поведение режима 23); 18/… — баг отсутствующих `_QWEN_MODEL`/`_FRIDA_MODEL` в llm_filter.py (глобали не вошли в извлечение; `global`-декларация подавляла F821) — **исправлено и верифицировано живым прогоном, см. §7** |
| 7. cli.py + MODES + фасад main.py | ✅ 1de7d5a | main.py = 126 строк (было 4070); `__all__` = 89 имён (все тесты/evaluate_search покрыты); smoke меню — рендер из реестра, выход чистый; 110 passed |

**Итог по переезду:** 4070 строк → main.py 126 + пакет astro_app (~3900 строк в 20 модулях).
**Выученные уроки (для протокола):** (1) перегенерация по жёстким номерам строк сломала границы — только AST-извлечение по именам; (2) сканер глобалей дважды давал ложноотрицательный результат из-за присваиваний внутри функций после `global` — исправлен и снабжён самопроверкой «ловит дыру»; (3) monkeypatch-цели — namespace ВЫЗЫВАЮЩЕГО, не поставщика (обновлено в 4 тестовых файлах); (4) main.py как фасад обязан реэкспортировать полный API — тесты и evaluate_search зависят от `import main`.

## 7. Отложенные исправления (после завершения миграции — по решению владельца)

| # | Находка | Детали | Источник |
|---|---------|--------|----------|
| 1 | ~~Дисплей режима 12~~ → **✅ ИСПРАВЛЕН** | Причина: `not is_selfrag` в условии выбора ветки распаковки — Self-RAG и Small→Big возвращают одну форму 6-кортежей. Фикс: исключение убрано. Верификация: снапшот дисплея до/после (`display_pre/post.json`) — **22/23 режима без изменений, изменился только режим 12 (падение → рендер)**; тесты `test_show_results_modes.py` (5 шт., один был красным до фикса) | снапшот дисплея + тест |
