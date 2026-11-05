# История изменений HippoRAG

## 2026-11-06 — H2-2: живая приёмка жизненного цикла индекса — PASS

### Приёмка
- Цикл «бэкап → загрузка PDF → индекс обновился → restore → откат» выполнен
  на реальном индексе `outputs/astrology_test` (1369 чанков, 3 PDF) через
  API с `X-API-Key` (fail-closed подтвердился).
- Загрузка `h2_2_acceptance.pdf` (1 страница, токен XYZZY): 4-й документ в
  `/index/documents`, контрольный поиск находит его первым и отвечает с
  цитатой. `POST /index/restore` из приёмочного бэкапа (65 МБ): снова
  3 документа, тестовый исчез из выдачи, старый корпус отвечает корректно
  («Шен-ци гуа 1 → юго-восток [1, 5]»).
- Приёмка H2-2 из ROADMAP закрыта.

### Находка
- `reindex`/`upload-and-ingest` при 0 извлечённых чанков возвращают успех
  без предупреждения (тихий no-op, обнаружено на первой попытке с короткой
  страницей). Рекомендация — явная диагностика в ответе API; записана в
  `H2_2_DATA_LIFECYCLE_2026-11-05.md` §5.

### Артефакты
- `scripts/make_h2_2_test_pdf.py` — генератор 1-страничного тестового PDF
  (ASCII, уникальный токен XYZZY, извлечение проверено PyMuPDF).
- Результат приёмки — `H2_2_DATA_LIFECYCLE_2026-11-05.md` §4.1.

## 2026-11-05 — H2-6: попытка поднять citation precision — отрицательный результат

### Измерено
- Baseline M7 переснят до любых изменений: 36 предсказаний
  (`eval/grounding/predictions_m7_baseline.jsonl`), P 0.578 / R 0.848 /
  page recall@5 87.0 %; расхождения с опубликованным M7 (0.621/0.848/87.0 %)
  — шум одной выборки при температуре 0.15.
- Офлайн-проверка пяти политик отбора цитат (`scripts/m7_citation_policies.py`,
  референс из задания): `identity` совпал с baseline; лучший `hybrid` —
  P 0.652, но uncited 16.7 % (порог ≤ 10 %) и потеря 5 gold-цитат
  (recall-запас 1.1 исчерпан). Ни одна политика не проходит порог 0.80.
- Prompt-only «одна лучшая ссылка на утверждение» (live, 36 вопросов):
  P **0.714** (+0.136), но R **0.783** (< 0.80) — модель схлопнула `a13`
  до единственного не-gold фрагмента (R 1.0 → 0.0) и потеряла gold в `b05`.
  Комбинация «промпт + селектор» исключена арифметически (recall ≤ 0.783).

### Решение
- Отрицательный результат зафиксирован (Шаг 7 задания): продакшн-код,
  golden set, метрики и пороги **не менялись**; промпт откачен после замера.
- Следующие гипотезы: ограничение контекста, ретрив/rerank, пересмотр
  постановки citation precision.

### Артефакты
- `eval/grounding/m7_citation_policies.{json,md}` — таблицы политик,
  prompt-only и per-question аудит (`lost_gold_ids` по каждому вопросу).
- `eval/grounding/report_m7_baseline.{json,md}`, `report_m7_prompt_one.{json,md}`,
  `predictions_m7_prompt_one.jsonl`, `m7_policy_dump.jsonl`.
- Обновлены: `H2_6_GROUNDING_REPORT` §9, `ROADMAP_REVISION_DEEPSEEK` (H2-6),
  `RESULTS_REPORTED.md`.

## 2026-11-05 — M2: замер legacy-режимов 12/19/21 выполнен

### Измерено
- Citation coverage (M2) на golden set (36 вопросов), один прогон на режим:
  **0 ответов без цитаты у всех трёх legacy-режимов** — 12 Self-RAG
  (19 ответов), 19 GraphRAG (20), 21 DAT+LLM (22); marker validity 1.000
  у всех. Оговорка «legacy отдельно не гонялись» снята.
- Вторичные метрики: citation precision 0.177 / 0.122 / 0.468; accuracy
  34.8 / 47.8 / 65.2 %; p50 12.6 / 19.9 / 24.7 с. Сильнейший из трёх —
  режим 21; все уступают продуктовому пути (M7: precision 0.621).

### Артефакты
- `eval/grounding/predictions_mode{12,19,21}.jsonl` — по 36 предсказаний
  (id совпадают с golden set).
- `outputs/reports/m2_mode{12,19,21}.json|.md`, `*_raw.jsonl`,
  `m2_legacy_summary.md` — отчёты `eval_grounding.py` и сводная таблица.
- Адаптер `scripts/m2_legacy_modes.py` и сводка `scripts/m2_legacy_summary.py`.

### Условия
- Индекс `outputs/astrology_test` (gemma-4-e4b + mpnet-base-v2, 1369 чанков,
  3 PDF, манифест `ok`), `qa_top_k=5`, температура 0.15, LM Studio
  `google/gemma-4-e4b`, Python 3.14.3 (в задании ожидался 3.11.9).
  Полный отчёт — §10.1 `MEASUREMENT_REPORT_2026-11-05.md`.

## 2026-11-05 — M2: задание на замер legacy-режимов 12/19/21

### Добавлено
- `docs/TASK_M2_LEGACY_MODES_2026-11-05.md` — воспроизводимое задание:
  golden set из 36 вопросов × режимы 12/19/21, `qa_top_k=5`, LM Studio
  `google/gemma-4-e4b`, `scripts/eval_grounding.py --no-thresholds`.
- Референсный адаптер `scripts/m2_legacy_modes.py` (в приложении A):
  перехватывает фрагменты в `answer_grounded`, различает ответ и retrieval
  fallback, пишет JSONL в формате `eval_grounding.py` и сырые ответы.
- Критерии приёмки, подводные камни и шаблон отчёта; ссылки добавлены в
  план измерений, отчёт M1–M8, H2-6 и `RESULTS_REPORTED.md`.

### Статус
- Замер **не выполнен**: нужна машина с LM Studio и индексом
  `outputs/astrology_test` на `paraphrase-multilingual-mpnet-base-v2`.
- Текущая M2 по-прежнему опирается на данные M7 продуктового пути
  (24 из 24 ответов с цитатой, 0 % без цитаты); это явно указано в отчёте.

## 2026-11-05 — H2-2: жизненный цикл индекса

### Добавлено
- `src/astro_app/index_lifecycle.py` — статус, документы, бэкап, restore,
  удаление и переиндексация без привязки к FastAPI.
- API: `GET /index/status`, `GET /index/documents`, `POST /index/backup`,
  `GET /index/backups`, `POST /index/restore`, `POST /index/delete`,
  `POST /index/reindex`.
- `/health` теперь отдаёт версию формата индекса, его состояние, текущую
  операцию и число документов.
- `/upload-and-ingest` идёт через `IndexLifecycle.reindex`: сначала бэкап,
  затем удаление старых чанков того же файла и повторная индексация.

### Безопасность
- Restore распаковывает копию во временный каталог, проверяет `graph.pickle`
  и манифест, подменяет каталог и пересоздаёт `HippoRAG`; при ошибке старый
  индекс возвращается на место.
- У общего чанка двух PDF удаляется только metadata-запись удаляемого
  источника — сам чанк остаётся.
- Имя бэкапа проверяется как basename `.zip`; выход за каталог бэкапов
  запрещён.

### Тесты
- `tests/unit/test_index_lifecycle.py` — 12 тестов.
- `tests/unit/test_search_api_index.py` — 9 тестов.
- Полный unit-набор: **457 passed, 1 skipped**.

### Ограничение
- Панели администратора в Web UI пока нет: операции доступны через API и
  Swagger. Живая приёмка на реальном индексе не проводилась.

## 2026-11-05 — M5/M6: установка и ablation synonymy

### Результаты
- **M5:** установка на чистую машину — **11.1 мин** (667.7 с) при пороге
  ≤ 15 мин — PASS. Основное время — `pip install` (567.9 с).
- **M6:** ablation synonymy на режиме 2 (40 вопросов): `recall@5_doc`
  **3.3913** против **3.0870**, MRR@10 **0.5160** против **0.4931**.
  Относительный прирост ≈ **+9.7 %** и **+4.6 %** — synonymy оставлена,
  H1-4 закрыт.
- Сырые отчёты: `outputs/reports/m5_install_time.json` и
  `outputs/reports/ablation_synonymy.json`.

## 2026-11-05 — M7: перезамер H2-6 после H1-6

### Результаты
- **Page recall@5 — 87.0 %** (20 из 23 answerable) — PASS при цели 80 %.
- **Answer rate — 100 %**, **accuracy — 73.9 %** — оба порога пройдены
  (было 73.9 % и 56.5 % в baseline H2-6).
- Abstention 92.3 %, false answer 7.7 %, citation recall 0.848,
  marker validity 1.000, uncited sentences 0 % — PASS.
- **M2 (citation coverage) — 0 % ответов без цитаты:** 24 из 24 ответов
  содержат ≥ 1 валидную цитату (по per-question данным M7, продуктовый
  путь; legacy-режимы 12/19/21 отдельно не гонялись).
- **Citation precision — 0.621** (< 0.80) — единственный непройденный порог.
- Латентность: p50 14 234 мс, p95 25 950 мс (36 вопросов).

M8 (73.9 %) — это ablation 7 упрощённых page-ретриверов, а не потолок
полного пути: разница объясняется keyword-boost в `page_retrieval.py`.
Отчёт и команда воспроизведения — в
[MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md).

## 2026-11-05 — M1/M3/M4/M8: результаты замеров

### Добавлено
- `docs/MEASUREMENT_REPORT_2026-11-05.md` — сводный отчёт по выполненным
  измерениям M1, M3, M4 и M8: условия, команды, числа и ограничения.
- `eval/dataset/RESULTS_REPORTED.md` — раздел M1–M8 с recall@10, page recall
  и покрытием тестами.

### Результаты
- **M1:** recall@10 по документам **97.5 %** (режимы 2 и 5, 40 вопросов) —
  PASS; по страницам 72.5 % (режим 2) и 75.0 % (режим 5) — ниже цели 85 %.
- **M3:** `citations[].sources` возвращает оба PDF для одного текста — PASS.
- **M4:** покрытие `src/hipporag` **40 %** (1928 / 4813) — PASS на границе.
- **M8:** лучший из 7 упрощённых page-ретриверов даёт **73.9 %** page
  recall@5 (`rrf_ce`). Это ablation, а не потолок полного H1-6-пути: M7 на
  полном пути даёт 87.0 %.
- M2, M5 и M6 на момент этого замера ещё не были готовы; они закрыты
  позже — см. записи «M7» и «M5/M6» выше.

### Исправлено
- `format_markdown_table` больше не подписывает `recall@10` как `recall@5`:
  заголовок берёт фактический `k` из summary. Добавлен тест
  `test_format_markdown_table_uses_actual_k`.
- `outputs/reports/eval_recall10.md` перегенерирован из JSON с исправленной
  подписью; значения не менялись.

## 2026-11-05 — H2-5: документация

### Добавлено
- `docs/USER_GUIDE.md` — полное руководство пользователя (RU): установка,
  индекс, Web UI, режимы, цитаты, отказ, обратная связь, история, экспорт,
  CLI, приватность и ограничения.
- `docs/ADMIN_GUIDE.md` — эксплуатация: переменные окружения, жизненный цикл
  индекса, бэкап/восстановление, API-ключи и маршруты, обратная связь, логи,
  чек-лист безопасности.
- `docs/ARCHITECTURE.md` — компоненты, путь запроса, режимы и единицы поиска,
  контракт заземления, хранилища, версионирование, тесты и замеры.
- `docs/TROUBLESHOOTING.md` — таблицы ошибок установки, LM Studio, индекса,
  Web UI, API/CORS и качества.
- `docs/FAQ.md` — частые вопросы о режимах, отказе, `confidence`, истории,
  приватности, эксплуатации и метриках.

### Изменено
- Заглушки `docs/QUICKSTART.md`, `docs/QUICKSTART_RU.md`,
  `docs/USER_GUIDE.md` заменены реальными руководствами.
- `docs/CONTRIBUTING.md` — ссылки на полный набор документации.
- `README_RU.md` — быстрый старт дополнен запуском Web UI и ссылками на
  руководства.
- Roadmap: статус H2-5 и отметки об исправленных расхождениях в §5.6/§11.

### Проверки
- `python scripts/check_docs_links.py`: **52 файла, 0 битых ссылок**.
- `pytest tests/unit -m unit -q`: **435 passed, 1 skipped**.

### Ожидает приёмки
Сценарий «новый пользователь проходит по документации» не проведён: нужны
индекс `outputs/astrology_test` и LM Studio.

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
