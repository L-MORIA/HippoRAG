# H1-6: page-level retrieval и page recall@5 (2026-11-05)

> **Статус:** реализация, тесты и приёмка готовы. M7 на полном page-level
> пути: **page recall@5 87.0 %** (20 из 23 answerable) — цель 80 % достигнута.
> M8 — это ablation 7 упрощённых ретриверов, лучший из них даёт 73.9 %; он
> не отменяет результат полного пути, потому что в `page_retrieval.py` есть
> keyword-boost. Числа «baseline» в §1 взяты из
> [H2_6_GROUNDING_REPORT_2026-11-05.md](./H2_6_GROUNDING_REPORT_2026-11-05.md),
> а не получены этим кодом; результаты M7/M8 — из
> [MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md).
>
> **Связанные документы:** [ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md)
> §9.2 (H1-6), [H2_6_GROUNDING_REPORT_2026-11-05.md](./H2_6_GROUNDING_REPORT_2026-11-05.md),
> [H1_3_LATENCY_REPORT_2026-11-04.md](./H1_3_LATENCY_REPORT_2026-11-04.md),
> [HOW_TO_RUN.md](./HOW_TO_RUN.md) §7.2.

## 1. Зачем

Замер H2-6 показал, что заземление упирается не в промпт, а в доставку
доказательства: page recall@5 у продуктового режима 5 — **56,5 %** (13 из 23
answerable), у графового `rag_qa` — 8,7 %, у dense — 21,7 %. Даже при 50
фрагментах режим 5 доходит только до 65,2 %. При таком потолке answer rate
≥ 90 % и accuracy ≥ 70 % недостижимы.

Причина — единица выдачи. Режим 5 возвращает **чанки**, а несколько чанков
одной страницы занимают несколько слотов top-5. Страница, которая в корпусе
есть, может не попасть в пять «слотов», хотя её текст уже найден.

**Цель H1-6:** page recall@5 ≥ **80 %** на 23 answerable-вопросах golden set,
затем перезамер answer rate и accuracy H2-6.

## 2. Что сделано

| Файл | Изменение |
|---|---|
| `src/astro_app/page_retrieval.py` | Новый page-level гибридный ретрив: группировка чанков по странице, BM25 по полному тексту страницы, dense-максимум по чанкам, RRF, keyword-boost, RU cross-encoder по top-50. |
| `src/astro_app/modes/bm25_hybrid.py` | Обёртка `search_bm25_hybrid_ru_pages()` — page-level вариант режима 5. |
| `scripts/grounding_astro_retriever.py` | Адаптер для `eval_grounding.py --run` по умолчанию отдаёт страницы; чанковый baseline доступен через `ASTRO_RETRIEVER_PAGE_LEVEL=0`. |
| `scripts/eval_grounding.py` | `page recall@k`, `page coverage@k`, порог `page_recall_at_5 ≥ 0.80`, флаг `--require-page-recall`, page@5 в построчной таблице и сравнении прогонов. |
| `tests/unit/test_page_retrieval.py` | 12 тестов page-индекса и гибридного поиска. |
| `tests/unit/test_astro_retriever.py` | Page-level по умолчанию + чанковый режим при `ASTRO_RETRIEVER_PAGE_LEVEL=0`. |
| `tests/unit/test_eval_grounding.py` | Page recall/coverage, порог, `retrieved_provided`, отказ `--require-page-recall` без метрики. |

## 3. Как устроен page-level ретрив

1. **Группировка.** Чанки объединяются по `(source_file, page_num)`; текст
   страницы — склейка чанков в порядке индекса. Чанки без страницы
   (`page_num < 1`) в страничный индекс не попадают.
2. **BM25 по странице.** Корпус BM25 — полные тексты страниц, а не отдельные
   чанки. Запрос расширяется синонимами (`_expand_query`), синонимы-фразы
   разбиваются на токены.
3. **Dense-канал.** Эмбеддинг запроса сравнивается с эмбеддингами чанков;
   скор страницы — максимум по её чанкам. Если эмбеддингов нет, остаётся BM25.
4. **RRF.** Ранги BM25 и dense сливаются по `1 / (60 + rank)`; поверх —
   слабый keyword-boost (`1 + 0.1 * boost`), чтобы не перебивать каналы.
5. **RU cross-encoder.** Top-50 страниц переранжируются
   `DiTy/cross-encoder-russian-msmarco`. Реранкер режет вход по 512 символам,
   поэтому ему передаётся **окно вокруг лучшего чанка страницы**, а не начало
   страницы.
6. **Фрагмент.** В LLM уходит до 4000 символов страницы (окно вокруг лучшего
   чанка); цитата — `doc_title` + `page_num`.

Кэш страничного индекса привязан к `(id(store), id(texts), len(texts))`;
`reset_page_index_cache()` сбрасывает его в тестах и при смене индекса.

## 4. Метрика

`page recall@5` в `eval_grounding.py` — **доля answerable-вопросов, у которых
хотя бы одна размеченная страница попала в первые 5 записей `retrieved`**.
Это то же определение, что в H2-6 («доля вопросов, у которых размеченная
страница попала в контекст»), поэтому baseline 56,5 % сравним напрямую.

Дополнительно считается `page coverage@5` — доля **всех** gold-страниц в top-5.
Она нужна для пяти вопросов с двумя источниками: recall отвечает «нашлась ли
хоть одна страница», coverage — «насколько полно».

Если в предсказании нет блока `retrieved`, метрика равна `None`
(«не измерена»), а не нулю. `--require-page-recall` в этом случае завершает
прогон кодом 2.

## 5. Как воспроизвести

Нужны индекс `outputs/astrology_test` (1369 чанков, 653 страницы, 3 PDF),
эмбеддинг-модель и LM Studio с `google/gemma-4-e4b`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:ASTRO_RETRIEVER_TOP_K = "5"
$env:LLM_API_NAME = "google/gemma-4-e4b"

python scripts/eval_grounding.py --run `
  --retriever grounding_astro_retriever:retrieve `
  --require-page-recall `
  --out-json eval/grounding/report_h1_6.json `
  --out-md eval/grounding/report_h1_6.md `
  --out-predictions eval/grounding/predictions_h1_6.jsonl
```

Быстрая проверка доставки доказательства **без LLM** (ею сделана §9):

```powershell
python scripts/check_page_recall.py --min-recall 0.8
```

Чанковый baseline H2-6 для сравнения:

```powershell
$env:ASTRO_RETRIEVER_PAGE_LEVEL = "0"
python scripts/eval_grounding.py --run `
  --retriever grounding_astro_retriever:retrieve `
  --out-json eval/grounding/report_h1_6_chunks.json `
  --out-md eval/grounding/report_h1_6_chunks.md `
  --out-predictions eval/grounding/predictions_h1_6_chunks.jsonl
```

Настройки page-level пути:

| Переменная | По умолчанию | Смысл |
|---|---:|---|
| `ASTRO_RETRIEVER_PAGE_LEVEL` | `1` | `0/false/no/off` — вернуться к чанкам. |
| `ASTRO_RETRIEVER_RERANK_POOL` | `50` | Сколько страниц отдавать в RU-CE. |
| `ASTRO_RETRIEVER_PAGE_MAX_CHARS` | `4000` | Максимум символов страницы во фрагменте. |
| `ASTRO_RETRIEVER_TOP_K` | `5` | Сколько страниц идёт в LLM. |

## 6. Тесты

```text
pytest tests/unit -m unit -q
420 passed, 1 skipped
```

Покрыто: группировка чанков по странице, fallback метаданных через `get_row`,
пропуск чанков без страницы, детерминированные ранги при ничьих, dense-максимум,
работа без эмбеддингов, дедупликация страниц, добор top-k за пределами
`rerank_pool`, page recall/coverage и порог 80 %.

## 7. Ограничения и риски

- **Потолок нового индекса неизвестен.** Дедупликация снимает конкуренцию
  чанков одной страницы, но BM25 по полному тексту страницы может размывать
  редкие термины по сравнению с BM25 по чанку. Гарантии, что 80 % достижимы,
  нет — это покажет только замер. **Проверено 2026-11-06:** 87,0 % @5 на
  golden set (23 answerable) и 80,0 % @5 на 40 вопросах H1-1 — §9.
- **Page-level меняет гранулярность цитаты.** Цитата становится страницей, а не
  чанком; citation precision/recall H2-6 нужно пересчитать заново.
- **Фрагменты длиннее.** До 4000 символов на страницу вместо чанка ~512
  токенов; это влияет на латентность и стоимость prompt.
- **RU-CE обязателен для измеренного качества.** При недоступной модели
  реранкер деградирует в исходный порядок и печатает предупреждение; строгий
  режим — `HIPPORAG_REQUIRE_CROSSENCODER=1`.
- **M8 — ablation, а не потолок полного пути.** 7 упрощённых page-ретриверов
  дают максимум 73.9 % @5; полный H1-6-путь в M7 — 87.0 %. Разница —
  keyword-boost в `page_retrieval.py`. Просто «страница вместо чанка» без
  boost и полного реранка недостаточно.

## 8. Приёмка

| Критерий | Статус |
|---|---|
| Page-level/hybrid ретрив реализован | ✅ |
| Метрика page recall@k и порог 80 % в eval | ✅ |
| Unit-тесты и линтеры | ✅ 420 passed, 1 skipped (на 2026-11-06 — 465 passed, 1 skipped) |
| page recall@5 ≥ 80 % на 23 answerable | ✅ **87.0 %** (M7, полный H1-6-путь; независимо воспроизведено — §9) |
| Answer rate и accuracy H2-6 перемерены | ✅ M7: **100 %** и **73.9 %**; citation precision 0.621 — ❌ |

Следующий шаг: citation precision 0.621 < 0.80 — единственный провал H2-6.
Page-level путь уже подключён к Web UI и CLI (коммит `87ace00`), поэтому при
исправлении citation precision H2-6 перемеряется сразу на текущем пути.

## 9. Независимая проверка (2026-11-06)

Отдельный прогон без LLM на том же индексе: адаптер
`scripts/grounding_astro_retriever.py` → `search_bm25_hybrid_ru_pages`,
запуск — `scripts/check_page_recall.py` (есть `--min-recall` как gate).

| Метрика | Значение |
|---|---|
| page recall@5 | **87,0 %** (20/23) — совпадает с M7 |
| page recall@10 | 87,0 % |
| page recall@20 | 100 % (23/23) |
| page coverage@5 | 87,0 % |
| MRR@10 (страницы) | 0,670 |
| промахи | `a10-scorpio-zones` (стр. 94), `a14-bagua-squares` (стр. 3), `b03-scorpio-taurus` (стр. 94 и 109) |

Отдельно перемерен режим 5 на **другом наборе** — 40 вопросах H1-1
(`eval_search_quality.py --modes 5 --warmup`, отчёт
`outputs/reports/eval_mode5_pages.json`): страничный recall@5 **80,0 %** (было
65,0 % на чанках), документный 100 %, p50 317 мс / p95 416 мс (было 197 / 225
мс). Снимок `eval/dataset/latency_baseline.json` обновлён через
`--merge-baseline` — затронут только режим 5.

Латентность ретрива:

| Где | p50 | p95 |
|---|---|---|
| в процессе (23 вопроса, прогрев) | 262 мс | 340 мс |
| `/api/ask`, живой сервер (5 вопросов) | 279 мс | 4,3 с — первый запрос |
| `/api/ask` полностью (wall, LLM) | 21,9 с | 28,8 с |

Первый запрос после старта сервера — 4,3–5,4 с: строится page-индекс (BM25 по
653 страницам) и прогревается cross-encoder. Дальше 0,23–0,38 с. Холодный
старт можно скрыть прогревом при загрузке индекса — отдельная задача.

> **Про старый живой чек (2026-11-06 00:27).** В
> `outputs/reports/h1_6_web_check.json` записаны retrieval p50 6,5 с /
> p95 9,0 с. Причина не в коде: тот сервер поднялся с эмбеддингами и
> cross-encoder **на CPU** — в его логе `No device provided, using cpu`; на CPU
> 50 пар по 512 токенов считаются секундами. На CUDA тот же путь даёт
> 0,26–0,43 с (лог нового чека — `Use pytorch device: cuda`). Качество из того
> прогона не пострадало (ретривер детерминирован), но латентность оттуда брать
> нельзя. Перед замером латентности проверяйте в логе сервера строку про
> устройство. Старый JSON сохранён рядом:
> `outputs/reports/h1_6_web_check.json.bak_20261106_181412`.
