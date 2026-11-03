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
