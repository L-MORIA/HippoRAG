# H1-3 — бюджет латентности: замеры и закрытие блока

**Дата:** 2026-11-04 · **Задача:** H1-3 «Performance-бюджет» (план
`docs/ROADMAP_REVISION_DEEPSEEK.md`, §9.2) · **Статус:** закрыто по замерам;
открыто только продуктовое решение о смене default.

---

## 1. Что требовалось

1. Закрыть числом p95 режима по умолчанию («2» Граф) — в снимке стояло `null`,
   вердикт был честно «unknown».
2. Замерить режим 5 «BM25 + RU cross-encoder» — кандидат в default, который до
   этого на датасете H1-1 не измерялся вообще.
3. Пересчитать вердикты и обновить снимок `eval/dataset/latency_baseline.json`,
   не потеряв уже измеренные режимы.

## 2. Условия замеров

| Параметр | Значение |
|---|---|
| Датасет | `eval/dataset/questions.jsonl`, 40 вопросов, все `verified` (sha256 `3d8a70a9…e16d8bf`) |
| Индекс | `outputs/astrology_test`, 1369 чанков, 3 PDF (191 + 316 + 146 страниц) |
| Эмбеддинги | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| LLM | `google/gemma-4-e4b` (LM Studio, JIT) — режимам 2 и 5 не требуется |
| Cross-encoder (режим 5) | `DiTy/cross-encoder-russian-msmarco` (HF-кэш, CUDA) |
| Машина | одна локальная машина; при смене GPU/CPU числа надо перемерить |
| Метрики | `recall@5` (k=5), `MRR@10`, `nDCG@10`, `p50/p95` по 40 вопросам |

Источники чисел (вне репозитория, `outputs/` в `.gitignore`):

* `outputs/reports/eval_after_mode2.json` — режим 2, p50/p95 + качество;
* `outputs/reports/eval_mode5.json` — режим 5, новый прогон `--warmup`;
* `outputs/reports/eval_after_llm.json` — режимы 1, 12, 19, 21 (замер H1-1).

## 3. Результаты

### 3.1. Латентность и вердикты бюджета

| Режим | Роль | p50 | p95 | Бюджет | Запас | Вердикт |
|---|---|---|---|---|---|---|
| 2 «Граф» | default | 545 мс | **661 мс** | 10 000 мс | 9 339 мс | `ok` |
| 5 «BM25+CE-ru» | expert | **197 мс** | **225 мс** | 30 000 мс | 29 774 мс | `ok` (как default тоже прошёл бы 10 с) |
| 1 «Multi-vector» | llm | 6 153 мс | 12 807 мс | 30 000 мс | 17 193 мс | `ok` |
| 21 «DAT+LLM» | llm | 21 097 мс | 29 404 мс | 30 000 мс | 596 мс | `ok`, впритык |
| 19 «GraphRAG» | llm | 16 605 мс | 32 954 мс | 30 000 мс | −2 954 мс | `experimental` |
| 12 «Self-RAG» | llm | 23 804 мс | 37 937 мс | 30 000 мс | −7 937 мс | `experimental` |

### 3.2. Качество тех же прогонов

| Режим | recall@5 док | recall@5 стр | MRR@10 док | MRR@10 стр | nDCG@10 | p95 |
|---|---|---|---|---|---|---|
| 5 «BM25+CE-ru» | 97,5 % | **65,0 %** | 0,902 | **0,520** | 0,921 | **225 мс** |
| 2 «Граф» | 95,0 % | 60,0 % | 0,913 | 0,477 | **0,928** | 661 мс |
| 1 «Multi-vector» | 97,5 % | 27,5 % | **0,921** | 0,161 | **0,935** | 12 807 мс |

Читать так:

* режим 2 **уложился в бюджет default с 14-кратным запасом** (661 мс против
  10 с), LLM ему не нужен — по латентности default законен;
* режим 5 **втрое быстрее режима 2** и лучше по страничному recall
  (65,0 % против 60,0 %) и page-MRR (0,520 против 0,477). По документному
  recall оба близки (97,5 % / 95,0 %), а документный уровень в корпусе из трёх
  книг почти не различает режимы — рабочая метрика страничная;
* nDCG и MRR на документном уровне чуть выше у режимов 2 и 1 — это ожидаемо:
  страничная привязка и топ-1 документ у них точнее, но именно страничные
  попадания важны для ответа с цитатой;
* режимы 12 и 19 остаются «экспериментальными» (p95 > 30 с) и в меню помечены
  «⏱ экспериментальный».

## 4. Что добавлено в код (механика слияния снимков)

**Проблема.** `benchmark_budget.py --update-baseline` собирает снимок из
**одного** отчёта и перезаписывает файл целиком. Прогон только режима 5 стёр бы
из `latency_baseline.json` режимы 1, 12, 19 и 21 — то есть «закрытие» одного
режима уничтожало данные остальных.

**Решение.**

* `src/astro_app/performance.py::merge_baseline(existing, fresh)` — заменяет
  только одноимённые режимы, остальные сохраняет; проставляет `merged_at`,
  `merged_modes` и происхождение; исходные словари не мутирует.
* `scripts/benchmark_budget.py --merge-baseline` — тот же CLI, но сливает
  снимок вместо замены. Флаги `--update-baseline` и `--merge-baseline`
  взаимоисключающие (`argparse` группа), случайно не совместить.
* **Правило про `llm`.** Прогон без LLM (режимы 2 и 5) не должен затирать
  модель, которой мерили LLM-режимы: иначе в снимке вместо
  `google/gemma-4-e4b` оказывалось конфигурационное `gemma-4-e4b`. Правило
  реализовано в `merge_baseline` и покрыто тестом.

Снимок собран последовательно из трёх отчётов:

```bash
python scripts/benchmark_budget.py --from-report outputs/reports/eval_after_llm.json --update-baseline
python scripts/benchmark_budget.py --from-report outputs/reports/eval_after_mode2.json --merge-baseline
python scripts/benchmark_budget.py --from-report outputs/reports/eval_mode5.json --merge-baseline
```

Итог в `eval/dataset/latency_baseline.json`: 6 режимов (`1`, `2`, `5`, `12`,
`19`, `21`), `llm: google/gemma-4-e4b`, `merged_modes: ["5"]`, бюджеты
10 000 / 30 000 / порог 30 000 мс.

## 5. Проверка тестами

| Проверка | Команда | Результат |
|---|---|---|
| Контракт бюджета | `python -m pytest tests/unit/test_performance_budget.py -q` | **16 passed** (было 12; +3 на слияние, +1 на `llm`) |
| Весь unit-набор (как в CI) | `python -m pytest tests/unit -m unit -q` | **268 passed** |
| Форматирование | `python -m black --check scripts/benchmark_budget.py src/astro_app/performance.py tests/unit/test_performance_budget.py` | 3 файла без изменений |
| Линт | `python -m ruff check <те же файлы>` | All checks passed |
| Снимок vs политика | `test_repo_baseline_snapshot_is_consistent` | режимы 12/19 — `experimental`, 2/5/1/21 — `ok` |

Новые тесты:

* `test_merge_baseline_keeps_unmeasured_modes` — слияние не теряет режимы,
  обновляет одноимённый, не мутирует исходный снимок;
* `test_merge_baseline_into_empty_and_verdicts` — слияние в пустой снимок даёт
  верный вердикт;
* `test_merge_baseline_marks_new_experimental_mode` — новый медленный режим
  попадает в экспериментальные;
* `test_merge_baseline_keeps_llm_name_when_fresh_run_has_no_llm` — прогон без
  LLM не затирает модель LLM-замеров.

## 6. Ограничения

* Замер на **одной машине** и одном индексе; числа не переносятся на другое
  железо без перезамера (`--merge-baseline` обновит только измеренное).
* Сырые отчёты лежат вне репозитория, поэтому **независимого воспроизведения
  из репозитория нет** — нужен свой индекс и (для LLM-режимов) LM Studio.
* Режим 5 требует RU cross-encoder `DiTy/cross-encoder-russian-msmarco`
  (~700 МБ, тянется при первом запуске, грузится на CUDA).
* `rerank` по-прежнему не выделен из `retrieval_ms` — это отдельная доработка,
  не входящая в H1-3.
* 40 вопросов дают шаг метрики 2,5 п.п.; разница 65,0 % против 60,0 % — это
  два вопроса из сорока, а не статистически значимый разрыв.

## 7. Открытое решение владельца: менять ли default на режим 5

| Аргумент «за» | Аргумент «против» |
|---|---|
| p95 225 мс против 661 мс (втрое быстрее) | первый запуск качает cross-encoder ~700 МБ |
| recall@5 (стр) 65,0 % против 60,0 % | нужен CUDA/CPU-режим для cross-encoder |
| page-MRR 0,520 против 0,477 | nDCG@10 чуть ниже: 0,921 против 0,928 |
| LLM не нужен, работает без LM Studio | +1 зависимость в рантайме |

Если решение «да», правка минимальна:

```python
# src/astro_app/modes/__init__.py
DEFAULT_MODE = "5"
RECOMMENDED_MODES = ("2", "5")
```

плюс абзац в README о первом запуске и скачивании cross-encoder. Тест
`tests/unit/test_h1_2_default_mode.py` и снимок бюджета при этом остаются
согласованными: p95 режима 5 (225 мс) укладывается и в 10-секундный бюджет
default.

## 8. Как воспроизвести

```bash
# 1. Режим 5 — без LM Studio (нужен только cross-encoder в HF-кэше):
python scripts/eval_search_quality.py \
    --index outputs/astrology_test --modes 5 --warmup \
    --out outputs/reports/eval_mode5.json --markdown-out outputs/reports/eval_mode5.md

# 2. Режим 2 — без LLM:
python scripts/eval_search_quality.py \
    --index outputs/astrology_test --modes 2 --warmup \
    --out outputs/reports/eval_after_mode2.json

# 3. Вердикты и слияние снимка:
python scripts/benchmark_budget.py --from-report outputs/reports/eval_mode5.json --merge-baseline

# 4. Тесты:
python -m pytest tests/unit/test_performance_budget.py tests/unit/test_h1_2_default_mode.py -q
python -m pytest tests/unit -m unit -q
```

## 9. Связанные документы

* план: `docs/ROADMAP_REVISION_DEEPSEEK.md`, §9.2 (H1-3);
* политика и таблица вердиктов: `docs/PERFORMANCE.md`;
* агрегаты замеров: `eval/dataset/RESULTS_REPORTED.md`;
* машинно-читаемый снимок: `eval/dataset/latency_baseline.json`;
* контракт бюджета: `src/astro_app/performance.py`,
  `tests/unit/test_performance_budget.py`.
