# Как запустить HippoRAG

Короткая инструкция: от чистого клона до первого ответа. Подробности по
индексации — в [INDEXING_GUIDE.md](./INDEXING_GUIDE.md). План развития проекта —
в [ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md).

## 1. Что нужно

- **Python 3.10–3.12** (проект проверяется на 3.10, 3.11, 3.12; 3.13+ не
  гарантируется).
- **LM Studio** с одной из моделей: `google/gemma-4-e4b`,
  `google/gemma-4-26b-a4b`, Qwen3.6-дистиллят или `nvidia/nemotron-3-nano-4b`.
- **8+ ГБ RAM**; все методы работают на CPU, CUDA не требуется.
- PDF-книги для индексации (в репозитории их нет — `data/` в `.gitignore`).

## 2. Установка

```bash
git clone https://github.com/L-MORIA/HippoRAG.git
cd HippoRAG
python -m venv .venv
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Проверка, что пакет и точка входа видны:

```bash
python -c "import main; print('ok')"
```

Дополнительно (экспериментально): offline-режим OpenIE с локальной моделью —
`pip install -e ".[offline]"`. Он тянет `vllm` и `outlines` (guided-декодирование
по JSON-схеме); `outlines` официально не выпускается для Python 3.14, поэтому
extra ставится только на Python 3.10–3.13. Подробности — в
`docs/AUDIT_OUTLINES_DEPENDENCY_2026-11-04.md`.

## 3. LM Studio

1. Установите LM Studio и загрузите одну из моделей выше.
2. Запустите локальный сервер: **Developer → Start Server**, порт `1234`.
3. Проверьте, что API отвечает:

```bash
curl http://localhost:1234/v1/models
```

## 4. Конфигурация

Создайте в корне проекта файл `.env` (он в `.gitignore`, в git не попадёт):

```dotenv
LLM_NAME=google/gemma-4-e4b
LLM_BASE_URL=http://localhost:1234/v1
SAVE_DIR=outputs/astrology
```

`SAVE_DIR` важен: индексатор по умолчанию пишет в `outputs/astrology`, а
приложение без этой переменной читает `outputs/astrology_test`. Одна и та же
переменная должна быть задана и при индексации, и при поиске.

## 5. PDF и индексация

Положите PDF в `data/pdfs/` (подкаталоги поддерживаются):

```bash
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology
```

Для сканированных PDF добавьте OCR (нужен установленный Tesseract):

```bash
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology \
  --enable_ocr --ocr_language rus
```

Индексация трёх книг занимает часы и требует работающего LM Studio: факты
извлекаются через LLM. Прогресс пишется в `indexing.log`.

## 6. Проверка индекса

```bash
python scripts/show_index_stats.py --save_dir outputs/astrology
```

Скрипт печатает число чанков, сущностей, фактов, узлов и рёбер графа. Числа
зависят от корпуса — не сверяйте их с README, сверяйте с этим выводом.

### 6.1. Формат индекса и версии

Индекс описывает себя манифестом `index_manifest.json` рядом с `graph.pickle`:
версия формата, дата сборки, LLM, embedding-модель и счётчики. Версии трёх
форматов объявлены в одном месте — `src/astro_app/versions.py`
(`INDEX_FORMAT_VERSION`, `QA_CACHE_VERSION`, `PACKAGE_DATA_VERSION`).

При загрузке индекса версия сверяется с кодом. Если манифеста нет (индекс
собран старым кодом) или версия другая, в лог уходит предупреждение — работа
продолжается, но результат может быть несопоставим с текущим кодом. Проверить
статус и отметить заведомо совместимый индекс:

```bash
python scripts/index_manifest.py            # статус формата
python scripts/index_manifest.py --adopt    # отметить существующий индекс как формат 1
```

`--adopt` ничего не перестраивает: это ручное подтверждение совместимости.
Если индекс собран другим кодом или другой embedding-моделью — переиндексируйте.

Package-data (`src/astro_app/data/`) защищены версией и отпечатками файлов:
при расхождении загрузка словаря падает с `PackageDataError`, а не читает
чужие данные. После намеренного изменения данных пересоберите манифест:

```bash
python scripts/update_package_data_manifest.py           # пересобрать
python scripts/update_package_data_manifest.py --check   # только проверить
```

## 7. Первый запрос

```bash
python main.py
```

1. Выберите модель (Enter — текущая из `.env`).
2. Дождитесь `Загрузка индекса... Готово!`.
3. Выберите режим. **По умолчанию — режим 5 «BM25+CE-ru»** (Enter в строке выбора
   подставляет его); рекомендованы также режим 2 «Граф» (графовый поиск) и
   22 «Pattern Search» (точный поиск «ЗНАК — ЗОНА», без LLM). Остальные режимы —
   экспертные, у каждого в меню указано, когда он уместен. Бюджеты латентности —
   в [PERFORMANCE.md](./PERFORMANCE.md).

> **Первый запуск режима 5** качает cross-encoder `DiTy/cross-encoder-russian-msmarco`
> (~700 МБ). Если модель недоступна, режим не падает, а печатает в stderr
> предупреждение «ПЕРЕРАНК НЕ ВЫПОЛНЕН» — качество выдачи тогда ниже измеренного.
> Строгий режим (ошибка вместо деградации) включается переменной
> `HIPPORAG_REQUIRE_CROSSENCODER=1`.

Если индекс не найден, приложение сообщит об этом или загрузит пустой индекс:
в первую очередь проверьте `SAVE_DIR` и имя embedding-модели.

### 7.1. Заземлённый ответ: формат и отказ

Генеративные режимы **12 «Self-RAG»**, **19 «GraphRAG Agent»** и **21 «DAT + LLM»**
отвечают по одному контракту заземления:

```text
ОТВЕТ: Овну соответствуют прихожая и плита [1]. Управитель — Марс [1].
ИСТОЧНИКИ: 1
ОТКАЗ: нет
```

- `[n]` — номер переданного фрагмента; парсер проверяет, что ссылка входит
  в диапазон и что у ответа есть хотя бы одна валидная ссылка.
- Если ответа во фрагментах нет, модель возвращает
  `ОТВЕТ: В предоставленных фрагментах нет ответа на этот вопрос.` и
  `ОТКАЗ: да`.
- При `qa_strict_grounding=True` (по умолчанию) ответ без валидных `[n]`
  заменяется этим же отказом. Если модель ответила по существу, но забыла
  ссылки, допускается ровно один repair-retry; повторный ответ снова
  проходит проверку.
- Режимы 9–11 (`Весь знак`, `Резюме`, `Обраб.`) — не Q&A, а извлечение,
  резюме и стилистическая обработка; их формат не меняется.

### 7.2. Реальный замер заземления на golden set

Одна команда запускает 36 размеченных вопросов через настоящий
`HippoRAG.rag_qa` — без копии QA-пути и без stub-ответов:

```bash
python scripts/grounding_retriever.py \
  --save-dir outputs/astrology_test \
  --llm-name gemma-4-e4b \
  --llm-base-url http://localhost:1234/v1
```

По умолчанию пишутся:

* `eval/grounding/report_2026-11-04.json` — метрики, пороги и построчные результаты;
* `eval/grounding/report_2026-11-04.md` — тот же отчёт для ревью;
* `eval/grounding/predictions_2026-11-04.jsonl` — ответы по мере прогона.

В отчёте режим `real-rag_qa` и `quality_measurement=true`: это уже измерение
качества, а не проверка механики. Код возврата: `0` — пороги выполнены,
`1` — есть проваленные пороги, `2` — ошибка разметки или запуска.

Чтобы сравнить `qa_top_k`, задайте отдельные пути, иначе отчёты перезапишут
друг друга:

```bash
python scripts/grounding_retriever.py --qa-top-k 5 \
  --out-json eval/grounding/report_top5.json \
  --out-md eval/grounding/report_top5.md \
  --out-predictions eval/grounding/predictions_top5.jsonl
python scripts/grounding_retriever.py --qa-top-k 8 \
  --out-json eval/grounding/report_top8.json \
  --out-md eval/grounding/report_top8.md \
  --out-predictions eval/grounding/predictions_top8.jsonl
```

Смотрите на `citation_recall` и `answer_rate_answerable`: рост `qa_top_k`
помогает, пока в промпт попадают релевантные фрагменты, и вредит, когда
добавляется шум. Ошибка на отдельном вопросе не роняет прогон — она
записывается как `runtime_error:<Type>`; `--fail-fast` останавливает
на первой ошибке, `--no-strict` отключает замену ответа без цитат на отказ.

## 8. Тесты

```bash
pytest tests/unit -m unit
```

## Типовые ошибки

| Симптом | Причина и решение |
|---|---|
| `ModuleNotFoundError: No module named 'astro_app'` | Не выполнена editable-установка: `pip install -e ".[dev]"` |
| `Connection refused` к `localhost:1234` | Не запущен сервер LM Studio или другой порт — проверьте `LLM_BASE_URL` |
| Индекс «пустой», хотя файлы есть | `SAVE_DIR` при индексации и при поиске различаются |
| `ModuleNotFoundError: rank_bm25` / `igraph` | Установлены не все зависимости: повторите `pip install -e ".[dev]"` |
| `PackageDataError: Package-data несовместимы с кодом` | Данные и код разошлись: `python scripts/update_package_data_manifest.py` (после намеренного изменения данных) или переустановите пакет |
| В логе `Формат индекса не подтверждён` | Индекс собран до H0-8: `python scripts/index_manifest.py --adopt` либо переиндексация |
| В логе `Индекс собран другим форматом` | Индекс и код несовместимы: переиндексируйте или откатитесь на версию кода, которой соответствует индекс |
| `QA-кэш: N записей не прочитаны` | Записи старой версии схемы или другой модели — они будут пересчитаны, действие не требуется |
| `OutlinesUnavailableError: Guided-декодирование … требует пакет outlines` | Не установлен `outlines` (нужен для offline-режима): `pip install -e ".[offline]"` либо `pip install "outlines>=1.0,<2"` |
| Ошибки на Python 3.13/3.14 | Используйте Python 3.10–3.12 |
| OCR не работает | Не установлен Tesseract или язык `rus` в нём |


## Аутентификация API

Ключ `API_KEY` обязателен по умолчанию (`API_KEY_REQUIRED=1`): без него сервер
**не стартует** и завершается с понятной ошибкой. Это fail-closed по умолчанию —
репозиторий публичный, а API отдаёт содержимое локального корпуса.

Запуск (по умолчанию только loopback — наружу порт не выставляется):

```bash
export API_KEY=your-secret-key
uvicorn src.hipporag.search_api.server:app --host 127.0.0.1 --port 8000
```

Все запросы требуют заголовок:

```
X-API-Key: ваш-секретный-ключ
```

Поведение режимов:

| `API_KEY` | `API_KEY_REQUIRED` | Поведение |
|---|---|---|
| задан | 1 (по умолчанию) | запросы без ключа → **401**, с ключом → 200 |
| не задан | 1 (по умолчанию) | сервер **не стартует** (fail-closed) |
| не задан | 0 | fail-open: запросы обслуживаются без ключа — только для локальной разработки |

Опционально:

* `API_KEY_REQUIRED=0` — явный opt-out для локальной разработки на loopback.
* `API_ALLOWED_ORIGINS` — список разрешённых origin через запятую
  (по умолчанию `http://localhost:3000,http://localhost:8000`). Список origin —
  единственный источник правды: CORS-слой обрамляет проверку ключа, поэтому
  ответы об отказе (401) тоже содержат корректные CORS-заголовки и доступны
  браузеру с разрешённого origin.

Наблюдаемость: `GET /health` возвращает `auth_required` (требование ключа) и
`auth_failures` (счётчик отказов авторизации с запуска); каждый отказ также
пишется в лог с указанием пути и origin.

Смена ключа требует перезапуска сервера (ключ читается один раз при импорте).
