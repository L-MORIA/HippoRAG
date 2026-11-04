# История изменений HippoRAG

## 2026-11-05 — H2-6: заземление Q&A

### Добавлено
- Единый контракт ответа `ОТВЕТ / ИСТОЧНИКИ / ОТКАЗ` и модуль
  `src/hipporag/qa/grounding.py`: парсер ссылок `[n]`, общий отказ
  `ABSTAIN_TEXT`, один repair-retry при ответе без валидных ссылок.
- `HippoRAG.qa` и `StandardRAG.qa` не вызывают LLM при пустом контексте и
  сохраняют `used_citations`, `abstained`, `grounding_errors`.
- `/search`: `citations` теперь содержит только использованные цитаты,
  добавлены `abstained` и `grounding_errors`, а `confidence` означает
  качество заземления (`0.0 / 0.5 / 1.0`), а не вероятность правильности.
- Режимы CLI 12 «Self-RAG», 19 «GraphRAG Agent» и 21 «DAT + LLM» используют
  общий `answer_grounded`; ответы содержат inline-ссылки `[n]`.
- Golden set `eval/grounding/golden_qa.jsonl` (36 вопросов) и
  `scripts/eval_grounding.py`: abstention, false answer rate, citation
  precision/recall, marker validity, uncited sentence rate, answer accuracy,
  p50/p95 и сравнение baseline → after.
- `scripts/grounding_retriever.py` — real-eval адаптер: прогоняет golden set
  через настоящий `HippoRAG.rag_qa`, сохраняет predictions и отчёт, позволяет
  сравнить `qa_top_k` без правки `.env`; ошибка вопроса фиксируется как
  `runtime_error:<Type>` и не роняет весь прогон.
- Логи `/search`: `abstained`, `grounding_errors`, `used_citations`,
  `retrieved_docs`, latency и p95.

### Изменено
- `qa_strict_grounding=True` по умолчанию: ответ без валидных inline-ссылок
  заменяется отказом. Аварийный откат — `qa_strict_grounding=False`.
- Ответы режимов 12/19/21 вместо свободного текста возвращают
  `ОТВЕТ / ИСТОЧНИКИ / ОТКАЗ`.

### Ограничения
- Режимы 9–11 (`sign_zones`) осознанно не переведены на Q&A-контракт: это
  извлечение материала, резюме и стилистическая обработка, а не ответ на
  вопрос.
- Отчёт `eval/grounding/report_2026-11-04.*` проверяет механику на stub-LLM
  (`quality_measurement=false`); real-eval адаптер готов, но реальный замер
  качества требует индекса и LM Studio.

## 2026-05-14 — 23 метода, QA Cache, 4 модели

### Добавлено
- **Метод 20**: Sentence-level chunking (BM25 по 10559 предложениям)
- **Метод 21**: DAT Sentence-level + LLM (Dynamic Alpha Tuning)
- **Метод 22**: Pattern Search (точный поиск «ЗНАК — ЗОНА» с приоритетом классификаций, 0.3с)
- **Метод 23**: QA Cache (кэш ответов, автосохранение, тестирование, экспорт)
- **Команда `t`**: тестирование всех методов, сравнение с кэшем
- **Команда `e`**: экспорт кэша в JSONL для fine-tuning
- **Выбор модели**: 4 модели при запуске (gemma-4-e4b, gemma-4-26b-a4b, qwen3.6 claude, nemotron-4b)
- **LLM-free fallback**: для методов 21, 22 — программный вывод при пустом ответе LLM
- **Accusative case**: правильные падежи зон (плиту, кладовку, спальню)

### Исправлено
- Все LLM таймауты увеличены до 300с
- Извлечение ответа из reasoning_content для reasoning-моделей
- Приоритет классификации «светская обитель» при множественных совпадениях
- Path import для QA Cache

## 2026-05-12 — Исправление безопасности

### Исправлено
- `eval()` → `json.loads()` / `ast.literal_eval()` в openie_openai.py, HippoRAG.py, rerank.py
- `subprocess shell injection` в menu.py
- CORS и upload path validation в server.py

## 2026-05-11 — Русский NER

### Добавлено
- Русский NER pipeline (spaCy + domain dictionary) вместо LLM-based NER
- `text_processing()` в misc_utils.py — сохраняет кириллицу
- Семантическое переразбиение чанков (1422 чанка, threshold 0.7)
- Перестройка графа (8401 узлов, 62978 рёбер)
