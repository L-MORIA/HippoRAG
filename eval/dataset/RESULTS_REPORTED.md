# Результаты замера H1-1 — агрегированные значения

**Датасет:** `eval/dataset/questions.jsonl` — 40 вопросов, все `status=verified`
(11 библиография / 15 домен / 14 перефразировка; 3 исходных PDF).
**Модель:** `google/gemma-4-e4b` (LM Studio, JIT) · **Скрипт:** `scripts/eval_search_quality.py`
**Дата замера:** 2026-11-03 · **Индекс:** `astrology_test` (1369 чанков), локальный

> **Происхождение цифр.** Значения ниже **зафиксированы автором по результатам
> живого прогона** и продублированы здесь, чтобы они были видны и проверяемы
> в репозитории. Сырые отчёты (`outputs/reports/eval_after_llm.{json,md}`)
> лежат в каталоге `outputs/`, который находится в `.gitignore`, поэтому
> **независимого воспроизведения из репозитория нет**: для него нужны локальный
> индекс, LM Studio и эмбеддинг-модели. Агрегаты здесь — это фиксация
> показателей, а не подтверждение их корректности.

## Retrieval — режим 1 «Multi-vector»

| Метрика | Значение |
|---------|----------|
| recall@5 (документы) | **97.5 %** |
| recall@5 (страницы) | 27.5 % |
| MRR@10 (документы) | 0.921 |
| nDCG@10 | 0.935 |
| p50 / p95 | 6.2 с / 12.8 с |

Режим 2 «Граф» на том же датасете: recall@5 = 95.0 % (док) / 60.0 % (стр),
MRR@10 = 0.913 (док) / 0.477 (стр), nDCG@10 = 0.928, p50 545 мс / p95 661 мс.

Режим 5 «BM25 + RU cross-encoder» (`DiTy/cross-encoder-russian-msmarco`):
recall@5 = 97.5 % (док) / **65.0 %** (стр), MRR@10 = 0.902 (док) / **0.520** (стр),
nDCG@10 = 0.921, p50 **197 мс** / p95 **225 мс**, LLM не используется.
Отчёт: `outputs/reports/eval_mode5.{json,md}` (вне репозитория).

### Как читать постраничный recall

27.5 % (режим 1) и 60.0 % (baseline-таблица ниже, режим 2) — **разные алгоритмы
подсчёта на разных режимах, а не регрессия**: «Граф» привязывает чанки к
страницам через графовые связи, «Multi-vector» — через dense-эмбеддинги и
LLM-переформулировку. Сравнение «до/после» допустимо **внутри одного режима
на одном индексе**. Подробнее — врезка «Уточнение» в
`docs/ROADMAP_REVISION_DEEPSEEK.md` §8a.

## Baseline «до» и «после» (режим 2, 40 вопросов)

| Индекс | Чанков | recall@5 (док) | recall@5 (стр) | MRR@10 (стр) | p50, мс |
|---|---|---|---|---|---|
| `_backup_20260514_pre_audit` | 1422 | 95.0 % | 60.0 % | 0.416 | 1056 |
| `astrology_test` | 1369 | 95.0 % | 60.0 % | 0.477 | 545 |

Состав попаданий разный: `page_rank` изменился у 17 из 40 вопросов.

## LLM-режимы (adequacy + latency)

| Режим | adequacy | отказы | пусто | LLM вызовов/вопрос | p50 | p95 |
|-------|----------|--------|-------|-------------------|-----|-----|
| 12 Self-RAG | **0.753** | 2.5 % | 0 % | 1.00 | 23.8 с | 37.9 с |
| 19 GraphRAG | 0.667 | **10.0 %** | 0 % | 1.00 | 16.6 с | 33.0 с |
| 21 DAT+LLM | **0.812** | 7.5 % | 0 % | 3.00 | 21.1 с | 29.4 с |

Остаточные отказы: 19 GraphRAG — 10 % (4/40), 21 DAT — 7.5 % (3/40),
12 Self-RAG — 2.5 % (1/40). Отказы корректные (LLM не нашёл данных в источниках).

## Вердикты бюджета латентности (H1-3)

Снимок `eval/dataset/latency_baseline.json` собран слиянием трёх отчётов
(`eval_after_llm`, `eval_after_mode2`, `eval_mode5`).

| Режим | p50 | p95 | Бюджет | Вердикт |
|---|---|---|---|---|
| 2 «Граф» (default) | 545 мс | 661 мс | 10 с | ok |
| 5 «BM25+CE-ru» | 197 мс | 225 мс | 30 с (роль expert) | ok |
| 1 «Multi-vector» | 6.2 с | 12.8 с | 30 с | ok |
| 21 «DAT+LLM» | 21.1 с | 29.4 с | 30 с | ok (впритык) |
| 19 «GraphRAG» | 16.6 с | 33.0 с | 30 с | experimental |
| 12 «Self-RAG» | 23.8 с | 37.9 с | 30 с | experimental |

## M1–M8: замеры 2026-11-05

Источник: коммиты `e0c27b8` (M1/M3/M4/M8), `44f449e` (M7) и `7dd77fd`
(M5/M6). Сырые отчёты — `outputs/reports/eval_recall10.{json,md}`,
`outputs/reports/page_ceiling.json`, `outputs/reports/grounding_eval.{json,md}`,
`outputs/reports/m5_install_time.json` и `outputs/reports/ablation_synonymy.json`.
Подробный разбор, условия и команды — в
[docs/MEASUREMENT_REPORT_2026-11-05.md](../../docs/MEASUREMENT_REPORT_2026-11-05.md).

| # | Метрика | Результат | Цель | Вердикт |
|---|---|---|---|---|
| M1 | Recall@10 (документы) | **97.5 %** (режимы 2 и 5, 40 вопросов) | ≥ 85 % | ✅ PASS |
| M1 | Recall@10 (страницы) | **72.5 %** (режим 2) / **75.0 %** (режим 5) | ≥ 85 % | ❌ ниже цели |
| M2 | Доля ответов без цитаты | **0 %** (продуктовый путь: 24 из 24 по M7; legacy 12/19/21: по 0 без цитаты — §M2 ниже, прогон 2026-11-05) | 0 % | ✅ PASS |
| M3 | `citations[].sources` | оба PDF в цитате | все источники | ✅ PASS |
| M4 | Покрытие `src/hipporag` | **40 %** (1928 / 4813) | ≥ 40 % | ✅ PASS, на границе |
| M5 | Установка на чистую машину | **11.1 мин** (667.7 с) | ≤ 15 мин | ✅ PASS |
| M6 | Ablation графа (synonymy on/off) | `recall@5_doc` **3.3913** против **3.0870**, MRR@10 **0.516** против **0.4931** | доказать пользу | ✅ PASS |
| M7 | Grounding, 8 метрик | **7/8 PASS**; citation precision **0.621** | precision ≥ 0.8 | ⚠️ один провал |
| M7 | Page recall@5 (H1-6) | **87.0 %** | ≥ 80 % | ✅ PASS |
| M8 | Page recall@5, 7 упрощённых ретриверов | **73.9 %** (лучший `rrf_ce`) | ≥ 80 % | ❌ для набора; полный путь — 87.0 % |

### M1 — recall@10 по режимам

| Режим | n | recall@10 (док) | recall@10 (стр) | MRR@10 (док) | nDCG@10 (док) | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 «Граф» | 40 | **97.5 %** | 72.5 % | 0.912 | 0.928 | 470 мс | 588 мс |
| 5 «BM25+CE-ru» | 40 | **97.5 %** | **75.0 %** | 0.902 | 0.921 | 204 мс | 249 мс |

Режим 1 в прогон не включён: в отчёте `llm.available=false`, а режим 1
использует LLM-переформулировку.

### M2 — citation coverage

По per-question данным M7 (`outputs/reports/grounding_eval.json`), продуктовый
путь «режим 5 + page-level H1-6 + grounding»: 36 вопросов, 24 ответа, у всех
**≥ 1 валидная цитата** (минимум 1, максимум 5), ответов без цитаты — **0
(0 %)**. 12 отказов цитат не имеют по определению. Marker validity среди
ответов — 1.000, uncited sentences — 0 %.

**Отдельный прогон legacy-режимов 12/19/21 (выполнен 2026-11-05)** по заданию
[TASK_M2_LEGACY_MODES_2026-11-05.md](../../docs/TASK_M2_LEGACY_MODES_2026-11-05.md):
адаптер `scripts/m2_legacy_modes.py`, 36 вопросов golden set, `qa_top_k=5`,
LM Studio `google/gemma-4-e4b`, индекс `outputs/astrology_test` (mpnet,
1369 чанков, 3 PDF), один прогон на режим, Python 3.14.3.

| Режим | Ответов | С цитатой | Без цитаты | M2 coverage | Marker validity | Answer rate | Abstention | False answer | Cit. precision | Cit. recall | p50, мс | p95, мс |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 Self-RAG | 19 | 19 | **0** | **100 %** | 1.000 | 73.9 % | 84.6 % | 15.4 % | 0.177 | 0.177 | 12 583 | 20 490 |
| 19 GraphRAG | 20 | 20 | **0** | **100 %** | 1.000 | 82.6 % | 92.3 % | 7.7 % | 0.122 | 0.211 | 19 945 | 32 749 |
| 21 DAT+LLM | 22 | 22 | **0** | **100 %** | 1.000 | 91.3 % | 92.3 % | 7.7 % | 0.468 | 0.548 | 24 704 | 35 861 |

Итог: **0 ответов без цитаты у всех трёх** — цель M2 «0 %» достигнута и на
legacy-режимах. Нюанс режима 21: два отказа (`b04-owen-planet-kitchen`,
`u07-pickle-recipes`) содержат непустой текст с цитатой, но помечены
`abstained=true`; по определению M2 они не входят в знаменатель. Если считать
их ответами, ответов без цитаты всё равно 0. Вторичные метрики ниже
продуктового пути: precision
0.122–0.468 против 0.621 в M7; у режима 12 abstention 84.6 % (< 90 %) и
false answer 15.4 % (> 10 %); p95 режимов 19/21 выше 30 с. Полный отчёт —
[MEASUREMENT_REPORT_2026-11-05.md](../../docs/MEASUREMENT_REPORT_2026-11-05.md)
§10.1; предсказания — `eval/grounding/predictions_mode{12,19,21}.jsonl`,
отчёты — `outputs/reports/m2_mode{12,19,21}.json|.md` и
`outputs/reports/m2_legacy_summary.md`.

### M5 — установка на чистую машину

| Этап | Секунды |
|---|---:|
| Копирование репозитория | 45.0 |
| Создание venv | 12.1 |
| `pip install` | 567.9 |
| `import main` | 42.6 |
| **Итого** | **667.7** |

**11.1 мин** при пороге ≤ 15 мин — PASS. Основное время — `pip install`
(567.9 с, 85 %). Сырой отчёт: `outputs/reports/m5_install_time.json`.

### M6 — ablation synonymy

| Вариант | `recall@5_doc` | MRR@10 | synonymy-рёбер |
|---|---:|---:|---:|
| С synonymy | **3.3913** | **0.5160** | 726 250 |
| Без synonymy | 3.0870 | 0.4931 | 0 |
| Дельта | **+0.3043** | **+0.0229** | — |

Режим 2, 40 вопросов. Относительный прирост `recall@5_doc` ≈ **+9.7 %**,
MRR@10 — **+4.6 %**. Оба показателя лучше с synonymy, поэтому решение —
**оставить synonymy** (H1-4 закрыт). В JSON поле называется `recall@5_doc`,
но значение больше 1, поэтому это не доля; в проценты не пересчитывается.
Сырой отчёт: `outputs/reports/ablation_synonymy.json`.

### M7 — grounding на golden set

| Метрика | Значение | Порог | Статус |
|---|---:|---:|---|
| Abstention rate (unanswerable) | 92.3 % | ≥ 90 % | ✅ |
| False answer rate | 7.7 % | ≤ 10 % | ✅ |
| Answer rate (answerable) | **100.0 %** | ≥ 90 % | ✅ |
| Citation precision | **0.621** | ≥ 0.80 | ❌ |
| Citation recall | 0.848 | ≥ 0.80 | ✅ |
| Marker validity | 1.000 | ≥ 1.000 | ✅ |
| Uncited sentence rate | 0.0 % | ≤ 10 % | ✅ |
| Answer accuracy | **73.9 %** | ≥ 70 % | ✅ |
| **Page recall@5 (H1-6)** | **87.0 %** | ≥ 80 % | ✅ |

36 вопросов (23 answerable, 5 borderline, 13 unanswerable), полный page-level
путь H1-6, `qa_top_k=5`. Латентность: p50 14 234 мс, p95 25 950 мс.
Единственный провал — citation precision: модель добавляет лишние цитаты,
хотя нужные источники находит (citation recall 0.848).

**Попытка поднять precision (задание H2-6, 2026-11-05 вечер) — отрицательный
результат.** Baseline M7 переснят (36 предсказаний: P 0.578, R 0.848,
page recall 87.0 %). Офлайн-политики отбора цитат: лучший `hybrid` P 0.652
(uncited 16.7 % > 10 %, потеряно 5 gold-цитат) — ни одна не проходит.
Prompt-only «одна лучшая ссылка на утверждение»: P **0.714**, но R **0.783**
(< 0.80) — модель схлопнула `a13` до единственной не-gold цитаты. Промпт
откачен, код/gold/пороги не менялись. Артефакты:
`eval/grounding/predictions_m7_baseline.jsonl`,
`eval/grounding/m7_citation_policies.md` (таблицы политик + prompt-only +
per-question аудит), `eval/grounding/predictions_m7_prompt_one.jsonl`.

### M8 — ablation page-ретриверов

| Ретривер | R@5 | R@10 | R@20 | R@50 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| **rrf_ce** | **73.9 %** | **82.6 %** | 91.3 % | 91.3 % | 0.506 |
| bm25_ce_chunk | 69.6 % | 73.9 % | 82.6 % | 87.0 % | 0.534 |
| bm25_ce | 65.2 % | 73.9 % | 78.3 % | 87.0 % | 0.529 |
| bm25 | 60.9 % | 65.2 % | 69.6 % | 82.6 % | 0.473 |
| rrf | 47.8 % | 47.8 % | 65.2 % | 82.6 % | 0.371 |
| dense_max | 26.1 % | 34.8 % | 47.8 % | 65.2 % | 0.192 |
| dense_mean | 26.1 % | 30.4 % | 39.1 % | 56.5 % | 0.171 |

23 answerable-вопроса golden set, 653 страницы. Лучший из этих семи
ретриверов — 73.9 % при пороге 80 %. Это **не потолок полного H1-6-пути**:
M7 на том же golden set даёт 87.0 %. Разница — keyword-boost в
`page_retrieval.py`, которого нет в упрощённых ретриверах M8. Вывод
«проблема выше retrieval» верен только для набора M8.

### Команды воспроизведения

```powershell
# M1
python scripts/eval_search_quality.py \
    --index outputs/astrology_test \
    --modes 2,5 --k 10 --warmup \
    --out outputs/reports/eval_recall10.json \
    --markdown-out outputs/reports/eval_recall10.md

# M3
pytest tests/unit/test_stage0_fixes.py::test_same_text_different_documents_keep_metadata -q

# M4
pytest --cov=src/hipporag --cov-report=term-missing

# M7
$env:ASTRO_RETRIEVER_TOP_K = "5"
python scripts/eval_grounding.py --run --retriever grounding_astro_retriever:retrieve `
    --out-json outputs/reports/grounding_eval.json `
    --out-md outputs/reports/grounding_eval.md

# M5 и M6
# Скрипты замера в репозиторий не попали; в outputs/reports лежат сырые JSON
# с числами. Для M6 нужны два прогона режима 2: с synonymy и с
# synonymy_edge_topk=0.

# M8
python scripts/measure_page_ceiling.py
```

## Как воспроизвести

```bash
# 1. Поднять LM Studio с моделью gemma-4-e4b на http://localhost:1234/v1
#    (режиму 5 LLM не нужен — он работает и без LM Studio)
# 2. Прогнать замер на нужном индексе:
python scripts/eval_search_quality.py \
    --index outputs/astrology_test \
    --modes 1,2,5,12,19,21 \
    --warmup \
    --out outputs/reports/eval_after_llm

# 3. Пересчитать вердикты и слить снимок, не теряя чужие режимы:
python scripts/benchmark_budget.py --from-report outputs/reports/eval_after_llm.json --merge-baseline

# 4. Сверить метрики с таблицами выше.
#    Без индекса и LLM доступна только проверка контракта датасета и метрик:
pytest tests/unit/test_eval_smoke_contract.py tests/unit/test_performance_budget.py -v
```

## Что проверяется в CI

`tests/unit/test_eval_smoke_contract.py` (джоб `Eval Smoke` в
`.github/workflows/ci-cd.yml`) фиксирует без LLM и без индекса:

* контракт датасета: разбор, схема, уникальность `qid`, статус `verified`,
  валидация через `validate_dataset`;
* корректность метрик `recall@k` / `MRR@k` / `nDCG@k` на синтетической выдаче
  (идеальное попадание, полный промах, попадание ровно на 5-й позиции);
* согласованность разметки: `expected_terms` каждого вопроса встречается в его
  же `reference_answer`.

Индекс-зависимая часть (реальный retrieval на фиксированном индексе) в CI не
воспроизводится: индекс не лежит в репозитории. Это остаток задачи H1-5.
