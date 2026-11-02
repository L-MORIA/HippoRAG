# История изменений HippoRAG

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
