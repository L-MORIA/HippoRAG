# История изменений HippoRAG

## 2026-11-05 — H2-1: веб-интерфейс

### Добавлено
- `src/astro_app/web/` — FastAPI-слой и статика поверх существующего
  приложения: `service.py` (три режима и заземлённый ответ), `server.py`
  (`/ui/config`, `/ui/modes`, `/api/ask`, `/api/feedback`, `/`), `models.py`,
  `feedback.py` и `static/` (HTML/CSS/JS).
- Три продуктовых режима: **5 «Быстрый»** (по умолчанию), **2 «Граф»** и
  **1 «Максимальный охват»**. Ответ содержит только использованные цитаты,
  найденные фрагменты и явный отказ, если ответа нет в источниках.
- Обратная связь «полезно / бесполезно» с комментарием: append-only JSONL
  `outputs/feedback/feedback.jsonl`, путь переопределяется `FEEDBACK_PATH`.
  Персональные данные не сохраняются.
- История запросов в `localStorage` (до 50 записей), экспорт текущего ответа
  и истории в Markdown/JSON, копирование ответа.
- `tests/unit/test_web_ui.py` — 14 тестов сервиса, маршрутов, обратной связи
  и статики.

### Изменено
- `src/hipporag/search_api/server.py`: guard пропускает публичную статику
  `/` и `/ui/*`, а `/api/ask`, `/api/feedback`, `/search` и `/health`
  остаются под `X-API-Key`.
- `pyproject.toml`: `astro_app.web/static/*` включены в package-data.

### Тесты
- `pytest tests/unit -m unit -q`: **435 passed, 1 skipped**.

### Ожидает приёмки
Сценарий с 5 внешними пользователями не проведён: нужны индекс
`outputs/astrology_test` и запущенный LM Studio. Детали —
в `docs/H2_1_WEB_UI_2026-11-05.md`.

## 2026-11-05 — H1-6: page-level retrieval

### Добавлено
- `src/astro_app/page_retrieval.py` — page-level гибридный ретрив: чанки
  группируются по `(source_file, page_num)`, BM25 считается по полному тексту
  страницы, dense-канал берёт максимум по чанкам, каналы сливаются RRF,
  top-50 переранжируется русским cross-encoder. Текст страницы обрезается
  вокруг лучшего чанка; реранкеру передаётся окно 512 символов вокруг него.
- `search_bm25_hybrid_ru_pages()` в `src/astro_app/modes/bm25_hybrid.py` —
  page-level вариант режима 5.
- `page recall@k`, `page coverage@k` и порог `page_recall_at_5 ≥ 0.80` в
  `scripts/eval_grounding.py`; флаг `--require-page-recall`, page@5 в
  построчной таблице и в сравнении прогонов.
- `tests/unit/test_page_retrieval.py` — 12 тестов page-индекса и гибридного
  поиска.

### Изменено
- `scripts/grounding_astro_retriever.py` по умолчанию отдаёт страницы, а не
  чанки. Чанковый baseline H2-6 остаётся доступен через
  `ASTRO_RETRIEVER_PAGE_LEVEL=0`.
- Page recall считается по определению H2-6 — «хотя бы одна gold-страница в
  top-k», поэтому baseline 56,5 % сравним напрямую. Дополнительно считается
  coverage по всем gold-страницам.

### Тесты
- `pytest tests/unit -m unit -q`: **420 passed, 1 skipped**.

### Ожидает замера
Реальный page recall@5 и перезамер H2-6 не сняты: в рабочей копии нет
`outputs/astrology_test`, PDF-корпуса и LM Studio. Команды и статус —
в `docs/H1_6_PAGE_RETRIEVAL_2026-11-05.md`.

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
- `scripts/grounding_astro_retriever.py` — второй real-eval адаптер: прогоняет
  golden set через продуктовый режим 5 (BM25 + RU-CE), а не через
  `HippoRAG.rag_qa`; настройки — `ASTRO_RETRIEVER_TOP_K`, `ASTRO_SAVE_DIR`,
  `ASTRO_LLM_NAME`, `ASTRO_LLM_API_NAME`, `ASTRO_LLM_BASE_URL`.
- `llm_api_name` в `BaseConfig` / `OpenAILLM` / `astro_app.config` и флаг
  `--llm-api-name`: id модели для API теперь можно задать отдельно от имени
  каталога индекса (LM Studio принимает только `google/gemma-4-e4b`, а индекс
  лежит в каталоге `gemma-4-e4b`). Env — `LLM_API_NAME`, `ASTRO_LLM_API_NAME`.
- Отчёт `docs/H2_6_GROUNDING_REPORT_2026-11-05.md` — замер на golden set,
  диагностика ретрива и разбор потерь по вопросам.

### Изменено
- `qa_strict_grounding=True` по умолчанию: ответ без валидных inline-ссылок
  заменяется отказом. Аварийный откат — `qa_strict_grounding=False`.
- Ответы режимов 12/19/21 вместо свободного текста возвращают
  `ОТВЕТ / ИСТОЧНИКИ / ОТКАЗ`.
- `qa_top_k` оставлен **5**: на 8 ложные ответы HippoRAG растут 7,7 % → 15,4 %,
  accuracy режима 5 падает 56,5 % → 52,2 %. `grounded_qa.lmstudio_chat`
  использует `LLM_API_NAME` вместо жёсткого `LLM_NAME`.

### Измерено (2026-11-05)
36 вопросов (answerable 23, borderline 5, unanswerable 13), индекс
`outputs/astrology_test/gemma-4-e4b_…`, LM Studio `google/gemma-4-e4b`:

| Метрика | HippoRAG @5 | Режим 5 @5 | Порог |
|---|---:|---:|---:|
| Answer rate | 34,8 % | **73,9 %** | ≥ 90 % |
| Abstention | 92,3 % | 92,3 % | ≥ 90 % |
| False answer rate | 7,7 % | 7,7 % | ≤ 10 % |
| Citation precision / recall | 0,125 / 0,125 | **0,667 / 0,765** | ≥ 0,80 |
| Marker validity | 1,000 | 1,000 | ≥ 1,00 |
| Uncited sentences | 0 % | 0 % | ≤ 10 % |
| Answer accuracy | 8,7 % | **56,5 %** | ≥ 70 % |

Page recall@5: графовый путь 8,7 %, режим 5 — 56,5 % (при 50 фрагментах
65,2 %). Ablation `qa_top_k`: 5 → 8 не улучшает ни один путь. Всё остальное
упирается в ретрив — задача H1-6 в roadmap.

### Ограничения
- Режимы 9–11 (`sign_zones`) осознанно не переведены на Q&A-контракт: это
  извлечение материала, резюме и стилистическая обработка, а не ответ на
  вопрос.
- Отчёт `eval/grounding/report_2026-11-04.*` проверяет механику на stub-LLM
  (`quality_measurement=false`); реальные числа — в отчётах
  `report_top5/8.*` и `report_mode5*.*`.
- Три порога (answer rate, accuracy, citation precision) не выполнены по
  причине ретрива: у 6 из 10 потерь размеченной страницы не было в контексте.

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
