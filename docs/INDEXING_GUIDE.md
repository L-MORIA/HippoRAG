# Руководство по индексации PDF документов

> **Текущее состояние:** 1422 семантических чанка, все методы работают на CPU.  
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

### 3. Проверка

После индексации запустите `python main.py` для поиска любым из 23 методов.

## Структура индексов

```
outputs/astrology/
└── gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2/
    ├── graph.pickle              # Граф знаний
    ├── chunk_embeddings/
    │   └── vdb_chunk.parquet     # 1422 семантических чанка
    ├── entity_embeddings/
    │   └── vdb_entity.parquet    # 6979 сущностей
    ├── fact_embeddings/
    │   └── vdb_fact.parquet      # Факты (triples)
    └── llm_cache/                # Кэш LLM запросов
```

## Процесс индексации

1. Загрузка PDF → извлечение текста (с OCR если нужно)
2. Семантическое разбиение на чанки (threshold 0.7) → **1422 чанка**
3. Sentence-level разбиение → **10559 предложений**
4. Извлечение сущностей (NER): spaCy + domain dictionary (не LLM)
5. Извлечение фактов (OpenIE) через LLM
6. Построение графа знаний: **8401 узел, 62978 рёбер**
7. Генерация эмбеддингов
8. Сохранение индексов

## Возобновление индексации

Поддерживается: уже обработанные чанки пропускаются при повторном запуске.

## Sentence-level chunking

Система поддерживает два уровня:
- **Семантические чанки** (1422) — для BM25, dense, графового поиска
- **Предложения** (10559) — для sentence-level BM20 (метод 20) и DAT (метод 21)

## Примечания

- Все методы работают на **CPU** (CUDA не требуется)
- Russian cross-encoder: `DiTy/cross-encoder-russian-msmarco`
- Embedding: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- LLM: любая из 4 моделей (загружается в LM Studio)
- QA Cache автоматически сохраняет ответы в `outputs/qa_cache/qa_cache.json`
