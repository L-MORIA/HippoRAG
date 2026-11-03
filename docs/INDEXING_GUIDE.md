# Руководство по индексации PDF документов

> **Текущее состояние:** все методы работают на CPU; актуальные числа индекса
> печатает `python scripts/show_index_stats.py --save_dir outputs/astrology`.
>
> Снимок на 2026-11-02: 1369 чанков, 6528 сущностей, 6847 фактов,
> 7897 узлов / 736 089 рёбер.
>
> Embedding: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
> LLM: выбирается при запуске из 4 моделей

## Быстрый старт

### 1. Подготовка PDF

Поместите PDF в `data/pdfs/` (рекурсивный поиск по поддиректориям).

### 2. Запуск индексации

```bash
python index_pdfs.py
```

Или через меню: `python menu.py` → пункт 1.

Параметры:
```bash
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology
python index_pdfs.py --enable_ocr --ocr_language rus
```

> `--output_dir` должен совпадать с `SAVE_DIR`, который читает приложение
> (`outputs/astrology` по умолчанию у индексатора). Если в `.env` задан
> `SAVE_DIR=outputs/astrology_test`, используйте его и здесь.

### 3. Проверка

После индексации запустите `python main.py` для поиска любым из 23 методов.
Числа индекса проверяются командой:

```bash
python scripts/show_index_stats.py --save_dir outputs/astrology
```

## Качество поиска (eval)

Метрики считаются на размеченном датасете `eval/dataset/questions.jsonl`
(правила разметки — `eval/README.md`): `recall@5` по документам **и** страницам,
`MRR@10`, `nDCG@10`, отказы/пустые ответы, adequacy, p50/p95. Эталон пишется
вручную; LLM-судья источником истины не является.

```bash
# проверить разметку против индекса (модели не грузятся)
python scripts/eval_search_quality.py --validate-only

# «до»: режим 2 на бэкапе pre-audit
python scripts/eval_search_quality.py --index outputs/_backup_20260514_pre_audit --modes 2

# «после»: кандидаты на текущем индексе
python scripts/eval_search_quality.py --modes 2,5,20 --warmup
```

### До/после

Заполняется после ручной разметки 30–40 вопросов со `status: "verified"`.
Черновики (`draft`) в приёмку не идут — их включает только `--include-draft`.

| Режим | Индекс | n | recall@5 (док) | recall@5 (стр) | MRR@10 | nDCG@10 | p50, мс | p95, мс |
|---|---|---|---|---|---|---|---|---|
| 2 Граф | бэкап pre-audit | — | — | — | — | — | — | — |
| 2 Граф | текущий | — | — | — | — | — | — | — |

> Smoke на 3 черновых вопросах (не приёмка): режим 2 на текущем индексе —
> recall@5 документы 100%, страницы 66.7%, MRR@10 1.000, p50 539 мс, p95 676 мс.
> В корпусе три книги, поэтому документные метрики почти всегда 1.0 —
> рабочая метрика страничная.

На каждый режим в отчёте есть `n_candidates` и число прогонов с `n < 3`:
на 1–2 кандидатах `min_max_normalize` возвращает всем 1.0, поэтому такие
режимы нельзя сравнивать с остальными.

## Структура индексов

```
outputs/astrology/
└── gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2/
    ├── graph.pickle              # Граф знаний
    ├── chunk_embeddings/
    │   └── vdb_chunk.parquet     # семантические чанки
    ├── entity_embeddings/
    │   └── vdb_entity.parquet    # сущности
    ├── fact_embeddings/
    │   └── vdb_fact.parquet      # факты (triples)
    └── llm_cache/                # Кэш LLM запросов
```

## Процесс индексации

1. Загрузка PDF → извлечение текста (с OCR если нужно)
2. Семантическое разбиение на чанки (threshold 0.7)
3. Sentence-level разбиение (число зависит от корпуса)
4. Извлечение сущностей (NER): spaCy + domain dictionary (не LLM)
5. Извлечение фактов (OpenIE) через LLM
6. Построение графа знаний
7. Генерация эмбеддингов
8. Сохранение индексов

## Возобновление индексации

Поддерживается: уже обработанные чанки пропускаются при повторном запуске.
Перед новой индексацией скрипт очищает старый индекс и кэш NER.

## Sentence-level chunking

Система поддерживает два уровня:
- **Семантические чанки** — для BM25, dense, графового поиска
- **Предложения** — для sentence-level BM25 (метод 20) и DAT (метод 21)

## Примечания

- Все методы работают на **CPU** (CUDA не требуется)
- Russian cross-encoder: `DiTy/cross-encoder-russian-msmarco`
- Embedding: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- LLM: любая из 4 моделей (загружается в LM Studio)
- QA Cache автоматически сохраняет ответы в `outputs/qa_cache/qa_cache.json`
