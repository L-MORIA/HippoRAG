# План разделения main.py на модули

**Дата:** 2026-11-02
**Статус:** план (развилка №5 сводного аудита — активирована владельцем) **— на рецензии, 3 правки до начала работ**
**Объект:** `main.py` — 4070 строк, 72 функции, 23 режима поиска + QA-кэш + меню

> ⚠️ **Рецензия:** [`MAIN_SPLIT_PLAN_REVIEW.md`](./MAIN_SPLIT_PLAN_REVIEW.md).
> Два блокирующих замечания и одно существенное:
> 1. пакет должен лежать в `src/astro_search/` — иначе не попадёт ни в editable-установку, ни в wheel (проверено: wheel содержит только `hipporag`; `.pth` добавляет в `sys.path` только `src`);
> 2. `data/synonyms.json` попадает под `.gitignore` (`data/`) → словарь не уедет в репозиторий → тихое деградирование keyword-expansion во всех режимах;
> 3. единая сигнатура `fn(hipporag, query, top_k=10)` не соответствует коду: 5 функций `top_k` не принимают, у `search_self_rag` дефолт 30, у `search_semantic_filter` — 50.
>
> После этих трёх правок план можно выполнять.

---

## 1. Цели и принципы

1. **main.py становится тонкой точкой входа** (~50 строк): парсинг аргументов + вызов меню.
2. **Единый реестр режимов** вместо `if mode == "1" ... elif mode == "23"`: декларативная таблица
   `MODES: dict[str, ModeSpec]`, где `ModeSpec = (название, функция, флаги «специализированный»/«⭐»)`.
   Меню рисуется из реестра — добавление режима = одна запись, а не правка в 4 местах.
3. **Единая сигнатура режима**: `fn(hipporag, query, top_k=10) -> list[Result]`. Режимы, которым
   нужны нестандартные параметры, оборачиваются `functools.partial`.
4. **Ленивая загрузка тяжёлых моделей сохраняется** (сейчас `lru_cache`-синглтоны `_get_crossencoder`,
   `_get_e5`, `_get_colbert`, `_get_qwen`…) — переезжают в свои модули без изменений логики.
5. **Каждый шаг миграции = отдельный коммит + зелёные проверки** (`pytest tests/unit` — 110,
   py_compile, ruff, black/isort, smoke-прогон меню). Сломанный промежуточный коммит недопустим.
6. Импорты в стиле `from hipporag import ...` (единый namespace, S-13).

## 2. Целевая структура

```
main.py                      # ~50 строк: точка входа → astro_search.cli.main()

astro_search/
  __init__.py                # реэкспорт MODES и run()
  config.py                  # константы (SAVE_DIR, LLM_*, пути), STOP_WORDS,
                             # SENTENCE_START, load_hipporag(),初始化 env
  synonyms.py                # SYNONYM_MAP → перенос в данные (data/synonyms.json)
                             # + функция load_synonyms() (словарь — данные, не код)
  query.py                   # _expand_query, _word_re, _is_meta_reformulation,
                             # reformulate_query, split_query_terms                (строки 417–695)
  textutils.py               # show_text_safe, fix_chunk_start, _tokenize          (665–695, 819–833)
  primitives.py              # _build_bm25, _embedding_scores, _get_chunk_meta     (823–852, 1143–1166)
  rerankers.py               # _get_crossencoder(_ru), _rerank_with_crossencoder*,
                             # _get_e5, _get_colbert, _colbert_*                   (853–924, 1012–1022, 1529–1596)
  display.py                 # show_results                                        (3287–3344)
  qa_cache.py                # _qa_load/_qa_save/_qa_lookup/_qa_store/_qa_norm/
                             # _qa_derive_short/_qa_run_test/_qa_export_training   (3846–4070)
  modes/                     # 23 режима — по группам, сигнатура fn(hipporag, query, top_k)
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
                             # show_results,цикл ввода                                  (3345–3597)
```

Ссылки на строки — по текущему файлу (4070 строк); при миграции «плывут» на ±10 строк.

## 3. Порядок миграции (7 шагов, каждый = коммит + зелёные проверки)

| Шаг | Что переезжает | Почему первым/в таком порядке |
|-----|----------------|-------------------------------|
| 1 | `synonyms.py` (+SYNONYM_MAP в `data/synonyms.json`) | Нет зависимостей; самая «мясистая» константа (~100 строк) уходит из кода в данные |
| 2 | `textutils.py` + `display.py` | Утилиты без зависимостей от режимов |
| 3 | `qa_cache.py` | Самодостаточная подсистема (кэш/экспорт), используется меню t/e |
| 4 | `config.py` + `query.py` + `primitives.py` | Общий фундамент для всех режимов |
| 5 | `rerankers.py` | Тяжёлые ленивые синглтоны в одном месте; режимы начнут импортировать отсюда |
| 6 | `modes/*` — **группами** (dense → bm25 → graph → small_to_big → llm_filter → sentence_dat → self_rag/hyde → sign_zones → filtered), после каждой группы: smoke-запуск затронутых режимов из меню | Режимы зависят только от primitives/rerankers — мигрируют независимо |
| 7 | `cli.py` + новый `main.py` + реестр `MODES` | Последним: меню переключается на реестр, старый if/elif удаляется |

## 4. Реестр режимов (шаг 7)

```python
# astro_search/modes/__init__.py
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ModeSpec:
    label: str          # «Dense (multi-vector)»
    fn: Callable        # fn(hipporag, query, top_k=10) -> list
    star: bool = False  # ⭐ рекомендованный
    special: str = ""   # предупреждение («нужен знак или зона в запросе»)

MODES: dict[str, ModeSpec] = {
    "1": ModeSpec("…", dense.search_multi_vector),
    ...
    "22": ModeSpec("Pattern Search: знак ↔ зона", sign_zones.search_semantic_filter,
                   special="нужен знак или зона в запросе, иначе честный отказ"),
}
```

- Меню печатает таблицу из `MODES` (номер, label, ⭐/⚠) — правка «пометить режим 22/23 специализированным»
  (коммит 086daf1) превращается в поле `special`.
- `_qa_run_test` «Тестирование всех методов» проходит по `MODES` —oday дублирующий список исчезает.
- Нестандартные сигнатуры (например, режимы со `candidate_pool`) — `partial(fn, candidate_pool=300)`.

## 5. Приёмка (после каждого шага и финально)

1. `python -m pytest tests/unit` — 110 passed (рост покрытия по мере выноса модулей — добавить импорт-тесты реестра).
2. `python -m compileall .` + ruff/black/isort — как в CI.
3. Smoke-меню: интерактивный запуск, поочерёдный прогон всех 23 режимов на контрольном запросе
   («как обустроить прихожую для Овна») — сравнение выдачи с до-рефакторным снапшотом
   (зафиксировать снапшот выдачи ДО начала работ: `python main.py` → режимы 1–23 → JSON в `outputs/refactor_baseline/`).
4. Точки входа не меняются: `hipporag.bat`, `menu.py`, `test_menu.bat` продолжают работать
   (main.py сохраняет CLI-контракт).

## 6. Риски

| Риск | Митигация |
|------|-----------|
| Лямбды/замыкания в меню теряют контекст при переносе | Режимы — функции верхнего уровня, а не лямбды (заодно закрывает B023-класс) |
| Дублирование констант при миграции группами | Константы только в `config.py`; каждый шаг — grep на остатки в main.py |
| Случайное изменение поведения поиска | Снапшот выдачи (§5.3) до старта и после каждого шага |
| Батч-правка «всё сразу» | Строго по шагам 1–7, один шаг = один коммит |

**Оценка:** 1,5–2 рабочих дня (включая снапшот-базлайн и smoke-прогоны).
