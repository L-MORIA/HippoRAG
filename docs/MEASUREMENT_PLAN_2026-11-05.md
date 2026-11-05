# План измерения неотслеженных метрик (Mavis)

**Дата:** 2026-11-05
**Основание:** `ROADMAP_REVISION_DEEPSEEK.md` §10.1 (метрики: «не измерен», «неизвестно»), `DEEPSEEK_PLAN_2026-11-04.md` (H2-6: citation precision / hallucination), `H1_3_LATENCY_REPORT_2026-11-04.md` §7 (открытое решение о default), `scripts/measure_page_ceiling.py` (не запущен)

---

## Сводка: что не измерено

| # | Метрика | Цель | Где в плане | Как измеряем |
|---|---------|------|-------------|--------------|
| M1 | **Recall@10** (документы + страницы) — все режимы | ≥ 85% | §10.1 | `eval_search_quality.py --modes 1,2,5` — скрипт уже считает recall@k |
| M2 | **Доля ответов без цитаты** (citation coverage) | 0% | §10.1, H2-6 | H2-6 golden set + grounding eval — парсер уже извлекает |
| M3 | **`citations[].sources`** — все источники чанка | все | §10.1, Mavis П-2 | unit-тест на `EmbeddingStore` с двумя PDF |
| M4 | **Покрытие тестами `src/hipporag`** | ≥ 40% | §10.1 | `pytest --cov=src/hipporag --cov-report=term` |
| M5 | **Время установки на чистую машину** | ≤ 15 мин | §10.1, H2-3 | таймер: clone → venv → pip install -e → import main |
| M6 | **H1-4 Ablation графа** — synonymy on/off | доказать пользу | §9.2 H1-4 | прогон режима 2 с/без synonymy-рёбер (два индекса) |
| M7 | **Grounding метрики** (citation precision, hallucination rate, abstention) | precision ≥ 0.8 | H2-6 §8a | `scripts/grounding_astro_retriever.py` на golden set (36 вопросов) |
| M8 | **Page recall потолок** (page as unit of search) | recall@5 ≥ 80% | H1-6, `measure_page_ceiling.py` | `python scripts/measure_page_ceiling.py` (уже готов) |

---

## Детали по каждому измерению

### M1. Recall@10 — все retrieval-режимы

**Что:** recall@10 по документам и страницам для режимов 1, 2, 5. Сейчас измерен только recall@5.
**Как:** тот же скрипт, `--modes 1,2,5` — метрики recall@10 уже считаются внутри, нужно только вывести.
**Время:** ~5 мин (без LLM, режимы 1/2/5 — retrieval only).
**Доступ:** LM Studio не нужен.

### M2. Citation coverage — доля ответов без цитаты

**Что:** из H2-6 golden set (36 вопросов) прогнать генеративные режимы 12/19/21 через grounding eval и посчитать: сколько ответов содержат ≥ 1 валидную цитату `[n]`.
**Как:** `scripts/grounding_astro_retriever.py` — метрики citation precision / abstention уже реализованы.
**Время:** ~30 мин (LLM нужен для каждого вопроса).
**Доступ:** LM Studio с gemma-4-e4b.

### M3. `citations[].sources` — все источники чанка

**Что:** unit-тест: `EmbeddingStore.insert_strings` с одним текстом из двух PDF → `get_all_metadata_entries()` возвращает 2 записи → `build_citation_from_metadata` возвращает `sources = ['A.pdf', 'B.pdf']`.
**Как:** pytest, без LLM.
**Время:** 10 мин.
**Доступ:** не нужен.

### M4. Покрытие тестами `src/hipporag`

**Что:** `pytest --cov=src/hipporag --cov-report=term-missing` — текущий % покрытия.
**Как:** одна команда.
**Время:** ~2 мин.
**Доступ:** не нужен.

### M5. Время установки на чистую машину

**Что:** таймер от `git clone` до `python -c "import main"` на чистом каталоге.
**Как:** скрипт-обёртка с таймером.
**Время:** ~10–15 мин на самой установке + 2 мин на замер.
**Доступ:** сеть (pip install).

### M6. H1-4 Ablation графа — synonymy on/off

**Что:** построить два индекса (с synonymy-рёбрами и без), прогнать режим 2 на 40 вопросах, сравнить recall/MRR/latency.
**Как:** `scripts/build_index_fixture.py` с флагом `--no-synonymy` (нужно добавить) или два прогона `HippoRAG.index_pdfs` с `synonymy_edge_topk=0` и текущим значением.
**Время:** ~2 ч индексации (два индекса) + ~10 мин eval.
**Доступ:** LM Studio не нужен (режим 2 — Graph).

**Упрощённый вариант:** прогнать только режим 2 на текущем индексе с флагом `synonymy_edge_topk=0` в runtime (без построения второго индекса) — это не совсем чистый ablation, но даёт первый сигнал.

### M7. Grounding метрики на golden set

**Что:** citation precision (доля цитат, подтверждающих утверждение), abstention rate (доля корректных отказов), hallucination rate (доля фактов вне корпуса).
**Как:** `scripts/grounding_astro_retriever.py` — скрипт уже реализован, прогон на 36 вопросах golden set.
**Время:** ~30 мин (LLM для каждого вопроса).
**Доступ:** LM Studio с gemma-4-e4b.

### M8. Page recall потолок

**Что:** запустить `scripts/measure_page_ceiling.py` — 6 ретриверов (bm25, bm25_ce, bm25_ce_chunk, dense_mean, dense_max, rrf_ce) на страницах вместо чанков.
**Как:** скрипт готов, запускается одной командой. Если recall@5 ≥ 80% — страница как единица поиска решает проблему; если нет — проблема в чанкинге/эмбеддингах.
**Время:** ~10–15 мин (без LLM).
**Доступ:** не нужен (только эмбеддинги из кэша).

---

## Порядок выполнения

| Этап | Измерения | Время | Зависимости |
|------|-----------|-------|-------------|
| **1 (без LLM, без сети)** | M3, M4, M8 | ~30 мин | нет |
| **2 (с LLM)** | M2, M7 | ~60 мин | LM Studio + gemma-4-e4b |
| **3 (retrieval)** | M1 | ~5 мин | не нужен |
| **4 (установка)** | M5 | ~15 мин | сеть (pip) |
| **5 (индексация)** | M6 | ~2 ч | без LLM, но нужен cross-encoder |

**Итого:** этапы 1+3+4 можно сделать за один прогон (~50 мин). Этапы 2 и 5 требуют больше времени.

---

## Выход

Результаты всех измерений фиксируются в `eval/dataset/RESULTS_REPORTED.md` (обновление) + `docs/MEASUREMENT_REPORT_2026-11-05.md` (новый отчёт со сводной таблицей). Каждый замер снабжается командой воспроизведения.
