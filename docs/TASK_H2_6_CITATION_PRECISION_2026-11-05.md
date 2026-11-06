# Задание H2-6: citation precision M7 с 0,621 до ≥ 0,80

**Дата:** 2026-11-05
**Статус:** выполнено 2026-11-05 — **отрицательный результат**: ни офлайн-политики,
ни prompt-only, ни их комбинация не достигают P ≥ 0,80 без нарушения
остальных порогов H2-6; код/gold/пороги не менялись. Детали —
`eval/grounding/m7_citation_policies.md`, §9 отчёта H2-6
**Оценка:** 3–5 часов (baseline 36 вопросов + офлайн-политики + live-прогон + отчёт)
**Исполнитель:** агент на машине, где доступны индекс
`outputs/astrology_test` и LM Studio с `google/gemma-4-e4b`

**Связанные документы:**
[ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md) (H2-6),
[H2_6_GROUNDING_REPORT_2026-11-05.md](./H2_6_GROUNDING_REPORT_2026-11-05.md),
[MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md) (M7),
[H1_6_PAGE_RETRIEVAL_2026-11-05.md](./H1_6_PAGE_RETRIEVAL_2026-11-05.md) (§5),
[TASK_M2_LEGACY_MODES_2026-11-05.md](./TASK_M2_LEGACY_MODES_2026-11-05.md)
(образец оформления задания).

> **Коротко.** M7 прошёл все пороги H2-6, кроме citation precision:
> **0,621 < 0,80**. Citation recall при этом **0,848**, page recall@5 **0,870**,
> answer rate **100 %**, accuracy **73,9 %**. Значит, нужные страницы уже
> попадают в контекст, но модель цитирует лишние фрагменты. Нужно сохранить
> baseline-предсказания M7, офлайн проверить детерминированные политики отбора
> цитат, затем live-прогоном доказать, что выбранная политика даёт **≥ 0,80**
> без падения recall и остальных метрик. Если не получается — зафиксировать
> отрицательный результат, а не подгонять разметку, пороги или отдельные id.
>
> **Обновление 2026-11-06 (аудит метрики).** Read-only аудит показал: все
> 31 extra baseline — из того же документа, что и gold; document-level
> precision 1,000; 10 extras приходятся на 3 промаха ретрива (`a10`, `a14`,
> `b03`). Оракульная арифметика: при текущем ретриве нужно убрать 14 из 31
> extra, после исправления 3 промахов — 5 из 21. Официальная метрика,
> golden set и пороги не менялись; следующий шаг — ретрив/rerank, затем
> дисциплина цитирования. Детали —
> [H2_6_CITATION_METRIC_AUDIT_2026-11-06.md](./H2_6_CITATION_METRIC_AUDIT_2026-11-06.md).

---

## 1. Зачем это отдельное задание

M7 — единственный замер, где продуктовый путь «режим 5 + page-level H1-6 +
grounding» прошёл почти все пороги. Его citation precision **0,621** ниже
цели **0,80**, и это уже не проблема ретрива: page recall@5 = **0,870**,
citation recall = **0,848**. Проблема в том, что модель добавляет к ответу
лишние ссылки: например, в `a13-gua1-shen-qi` пять цитат при одной gold,
в `a07-owen-zones` — четыре при одной.

На машине, где готовилось это задание, нет ни индекса `outputs/astrology_test`,
ни LM Studio. Поэтому здесь нельзя получить настоящие M7-предсказания и
проверить правило. Всё, что можно было сделать локально, — прогнать политики
на старом полигоне `predictions_mode5.jsonl` (это **не M7**) и отсеять
заведомо плохие варианты. На рабочей машине есть тексты, индекс и модель,
поэтому проверка должна идти там.

**Успех задания** — это не обязательно «0,80 любой ценой». Успех — это
воспроизводимые предсказания, полная таблица политик, честный per-question
аудит и либо подтверждённый порог, либо отрицательный результат с причиной.
Подгонка `gold_sources`, `gold_terms`, порогов или правил под конкретные
вопросы запрещена.

### 1.1. Что уже проверено локально (не M7)

На старом полигоне `eval/grounding/predictions_mode5.jsonl` (ретрив mode5,
page recall@5 = 0,5652, **не M7**) получены такие числа:

| Политика | P | R | Uncited | Вывод |
|---|---:|---:|---:|---|
| baseline mode5 | 0,6667 | 0,7647 | 0 % | исходная точка |
| `per_sentence_rank` | 0,7255 | 0,7647 | 0 % | precision растёт, recall держится |
| `single_lowest` | 0,7647 | 0,7353 | 27,8 % | падает recall и uncited |
| `hybrid` | 0,7647 | 0,7647 | 5,56 % | лучший локальный кандидат |
| mode5_top8 + `hybrid` | 0,7037 | 0,7500 | 15,79 % | нарушает uncited |

Это только направление, а не доказательство для M7: другой ретрив, другой
page recall, другие ответы. Лексический IDF-фильтр поддержки тоже проверен и
отклонён: precision не растёт или падает recall, потому что не-gold страницы
часто содержат тот же факт. Поэтому в задании проверяются ранговые политики,
а не лексическая фильтрация.

---

## 2. Что должно получиться

### 2.1. Обязательный минимум (baseline и офлайн-анализ)

1. `eval/grounding/predictions_m7_baseline.jsonl` — 36 предсказаний M7,
   сохранённых **до** любых изменений кода и промпта.
2. `eval/grounding/report_m7_baseline.json` и `.md` — отчёт
   `scripts/eval_grounding.py` по этим предсказаниям.
3. `eval/grounding/m7_citation_policies.json` и `.md` — сводная таблица
   политик и per-question аудит (скрипт из Приложения A).
4. `eval/grounding/m7_policy_dump.jsonl` — опционально, переписанные
   предсказания одной политики для ручной проверки текста.

### 2.2. Если политика проходит порог

5. `eval/grounding/predictions_m7_policy.jsonl` — live-прогон с выбранной
   политикой.
6. `eval/grounding/report_m7_policy.json` и `.md` — отчёт live-прогона,
   все пороги H2-6 пройдены.
7. Изменения в коде: чистая функция отбора цитат, вызов в продуктовом пути,
   unit-тесты.
8. Обновлённые документы: `docs/H2_6_GROUNDING_REPORT_2026-11-05.md`,
   `docs/ROADMAP_REVISION_DEEPSEEK.md` (H2-6), `docs/CHANGELOG_RU.md`.

### 2.3. Если ни одна политика не проходит

- `eval/grounding/m7_citation_policies.md` с полной таблицей и разбором,
  почему порог не достигнут.
- Запись в `docs/H2_6_GROUNDING_REPORT_2026-11-05.md` §9 и в H2-6 роадмапа:
  отрицательный результат, какие гипотезы проверены, что делать дальше.
- **Никаких изменений продакшн-кода, golden set и порогов.** Отрицательный
  результат — допустимый и полезный исход.

---

## 3. Метрика и бюджет M7

### 3.1. Определения

`scripts/eval_grounding.py` считает citation precision и recall **по
answerable-вопросам, на которые система ответила**, и усредняет
**по вопросам**, а не по всем цитатам:

| Понятие | Формула |
|---|---|
| `cited` | множество уникальных `(doc_title, page_num)` из `citations` |
| `gold` | множество уникальных `gold_sources` вопроса |
| `citation_precision` вопроса | `|cited ∩ gold| / |cited|`, если `cited` непусто, иначе `0` |
| `citation_recall` вопроса | `|cited ∩ gold| / |gold|` |
| `citation_precision` M7 | среднее по 23 answerable-вопросам с ответом |
| `citation_recall` M7 | среднее по тем же 23 вопросам |

Отказы не входят в знаменатель precision/recall. Поэтому «убрать ответ»
нельзя: answer rate должен остаться ≥ 0,90, а accuracy ≥ 0,70.

### 3.2. Где именно теряется precision в M7

Источник per-question чисел — `outputs/reports/grounding_eval.json` (M7,
2026-11-05 19:09); сводка — `docs/MEASUREMENT_REPORT_2026-11-05.md`.

Ниже — все answerable-вопросы, у которых precision < 1 или recall < 1.
Остальные восемь вопросов (`a04`, `a05`, `a08`, `a09`, `a11`, `a12`, `a17`,
`b01`) имеют P = 1 и R = 1; их трогать нельзя.

| id | цитат | gold | P | R | Что это значит |
|---|---:|---:|---:|---:|---|
| `a01-astro-author-edition` | 2 | 1 | 0,500 | 1,000 | одна лишняя не-gold цитата |
| `a02-psy-author` | 2 | 1 | 0,500 | 1,000 | одна лишняя не-gold цитата |
| `a03-psy-subtitle` | 2 | 1 | 0,500 | 1,000 | одна лишняя не-gold цитата |
| `a06-eso-publisher-license` | 2 | 1 | 0,500 | 1,000 | одна лишняя не-gold цитата |
| `a07-owen-zones` | 4 | 1 | 0,250 | 1,000 | три лишние; крупнейшая потеря |
| `a10-scorpio-zones` | 3 | 1 | 0,000 | 0,000 | gold-страницы нет среди цитат; селектор не поможет |
| `a13-gua1-shen-qi` | 5 | 1 | 0,200 | 1,000 | четыре лишние; крупнейшая потеря |
| `a14-bagua-squares` | 4 | 1 | 0,000 | 0,000 | gold-страницы нет среди цитат; селектор не поможет |
| `a15-house4` | 2 | 1 | 0,500 | 1,000 | одна лишняя не-gold цитата |
| `a16-traditional-style` | 2 | 1 | 0,500 | 1,000 | одна лишняя не-gold цитата |
| `a18-meta-zodiac` | 2 | 1 | 0,500 | 1,000 | одна лишняя; ответ всё равно неполный (accuracy 0) |
| `b02-shen-qi-gua1` | 1 | 2 | 1,000 | 0,500 | единственная цитата — gold; удалять нельзя |
| `b03-scorpio-taurus` | 3 | 2 | 0,000 | 0,000 | gold-страниц нет среди цитат; селектор не поможет |
| `b04-owen-planet-kitchen` | 3 | 2 | 0,667 | 1,000 | одна лишняя не-gold цитата |
| `b05-ryby-spalnya` | 3 | 2 | 0,667 | 1,000 | одна лишняя не-gold цитата |

**Важно:**

* `a10`, `a14`, `b03` — нули не из-за лишних цитат, а из-за отсутствия
  gold-страницы в цитатах. Селектор, который только удаляет, не может их
  исправить. Поэтому потолок M7 — **20/23 = 0,8696**, а не 1,0.
* `b01-owen-two-forms` — 2 цитаты, обе gold. Любое удаление одной из них
  снижает и precision (1,0 → 0,5), и recall (1,0 → 0,5). Политика должна
  оставить `b01` без изменений.
* `b02-shen-qi-gua1` — единственная цитата gold. Удаление даёт P = 0 и R = 0.
  Политика должна оставить `b02` без изменений.
* `a18-meta-zodiac` — precision можно поднять, но accuracy останется 0:
  ответ неполный. Это не задача селектора цитат; accuracy 73,9 % уже выше
  порога 70 %, и её не нужно «лечить» этим заданием.

### 3.3. Арифметика порога

Сумма precision по 23 вопросам M7: **14,2834** (среднее 0,6210).
Для порога 0,80 нужна сумма **≥ 18,4**; разрыв — **4,1166**.

Ориентир, а не разрешение подгонять конкретные id:

* `a13` 0,2 → 1,0 даёт **+0,80**;
* `a07` 0,25 → 1,0 даёт **+0,75**;
* шесть из семи вопросов с P = 0,5 → 1,0 дают **+3,00**.

Итого 14,2834 + 0,80 + 0,75 + 3,00 = **18,8334**, среднее **0,8188**.
Пять из семи дают 18,3334, среднее **0,7971** — ниже порога. То есть
арифметически нужно вычистить **a13, a07 и минимум шесть из семи** вопросов
`a01, a02, a03, a06, a15, a16, a18` (или эквивалентно добрать на `b04`/`b05`,
где 0,667 → 1,0 даёт +0,333 за вопрос).

Это только ориентир: политика не знает, какие id gold, и может как выиграть,
так и проиграть на каждом вопросе. Поэтому нужен per-question аудит, а не
только агрегат.

### 3.4. Запас по recall

Сумма recall по 23 вопросам M7: **19,5** (среднее 0,8478).
Для порога 0,80 нужна сумма **≥ 18,4**; запас — всего **1,1**.

Практические следствия:

* потеря одной единственной gold-цитаты на одновопросной gold-странице:
  −1,0 → recall 18,5/23 = **0,8043** (ещё проходит, но впритык);
* потеря двух таких цитат: −2,0 → 17,5/23 = **0,7609** (провал);
* потеря одной из двух gold на `b04`/`b05`: −0,5; две такие потери — −1,0,
  ещё проходит; три — провал.

Поэтому политика обязана быть консервативной: лучше оставить лишнюю
не-gold цитату, чем удалить gold. Аудит `lost_gold_ids` в Приложении A
показывает это по каждому вопросу.

---

## 4. Условия замера

| Параметр | Значение |
|---|---|
| ОС | Windows 10/11 или Linux |
| Python | 3.11.9 (как в проекте) |
| Репозиторий | `D:\HippoRAG` или другой клон; зафиксируйте `git rev-parse HEAD` |
| Индекс | `outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2` |
| Состав индекса | 1369 чанков, 653 страницы, 3 PDF (как в M7) |
| LLM | LM Studio, API-модель `google/gemma-4-e4b` |
| API | `http://127.0.0.1:1234/v1` |
| `qa_top_k` | **5**, не менять |
| Page-level | **1** (`ASTRO_RETRIEVER_PAGE_LEVEL=1`) |
| Температура | 0,15 (как в `grounded_qa.lmstudio_chat`) |
| Датасет | `eval/grounding/golden_qa.jsonl` |

### 4.1. `.env`

```dotenv
LLM_NAME=gemma-4-e4b
LLM_API_NAME=google/gemma-4-e4b
LLM_BASE_URL=http://127.0.0.1:1234/v1
```

`LLM_NAME` — имя каталога индекса, `LLM_API_NAME` — id модели в LM Studio.
Это разные вещи; если перепутать, LM Studio вернёт 400.

### 4.2. Проверка перед запуском

Windows-пример; на Linux активируйте venv и замените
`.\.venv\Scripts\python.exe` на `python`.

```powershell
$env:PYTHONIOENCODING="utf-8"

git rev-parse HEAD
git status --short

.\.venv\Scripts\python.exe -c "import sys; print(sys.version)"
.\.venv\Scripts\python.exe -c "import httpx; print(httpx.get('http://127.0.0.1:1234/v1/models', timeout=5).json())"

.\.venv\Scripts\python.exe scripts\index_manifest.py `
  --working-dir outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2

.\.venv\Scripts\python.exe scripts\show_index_stats.py --save_dir outputs/astrology_test
```

Ожидаемо:

- в ответе `/v1/models` есть `google/gemma-4-e4b`;
- `index_manifest.py` печатает состояние `ok` и версию формата;
- `show_index_stats.py` показывает 1369 чанков и 3 PDF.

> **Если каталог индекса другой.** На некоторых машинах встречается
> `..._nvidia_NV-Embed-v2` — это другой embedding и другой индекс. Запишите
> фактический путь и модель; **не сравнивайте** такие числа с M7 напрямую.
> Для чистого замера нужен индекс с
> `paraphrase-multilingual-mpnet-base-v2`.

---

## 5. Данные

`eval/grounding/golden_qa.jsonl` — 36 вопросов:

- 23 answerable, из них 5 borderline;
- 13 unanswerable;
- у каждого вопроса есть `gold_answer`, `gold_sources`, `gold_terms`.

Датасет **не менять**. `borderline=true` входят в answerable и учитываются
в precision/recall, если система ответила. Тексты вопросов и `gold_sources`
не дублируйте в отчёте — они уже в JSONL; в отчёте нужны id и числа.
Для удобства дословные тексты 15 проблемных вопросов и их `gold_sources`
приведены в [Приложении D](#приложение-d-тексты-вопросов-из-m7-таблицы).

---

## 6. Порядок выполнения

### Шаг 0. Зафиксировать состояние

```powershell
git rev-parse HEAD | Tee-Object -FilePath outputs/reports/m7_commit.txt
git status --short
```

Если рабочее дерево грязное, закоммитьте или отложите изменения: baseline
должен быть снят с чистого кода. Запишите commit, embedding-модель, путь
индекса, число чанков/страниц и версию LM Studio.

### Шаг 1. Сохранить baseline M7

Сначала — только baseline, **без** изменения кода и промпта. Команда
повторяет H1-6 §5, но добавляет `--out-predictions` и `--no-thresholds`:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:ASTRO_RETRIEVER_TOP_K = "5"
$env:ASTRO_RETRIEVER_PAGE_LEVEL = "1"
$env:LLM_API_NAME = "google/gemma-4-e4b"

.\.venv\Scripts\python.exe scripts\eval_grounding.py --run `
  --retriever grounding_astro_retriever:retrieve `
  --no-thresholds `
  --out-json eval/grounding/report_m7_baseline.json `
  --out-md eval/grounding/report_m7_baseline.md `
  --out-predictions eval/grounding/predictions_m7_baseline.jsonl
```

`--no-thresholds` не отключает расчёт порогов: в JSON остаётся блок
`thresholds` с `passed`. Он только не возвращает код 1 из-за известного
провала citation precision. Предсказания пишутся **до** проверки порогов,
поэтому файл сохранится в любом случае.

Проверьте:

- в JSONL **36** строк, id совпадают с `golden_qa.jsonl`;
- в отчёте `answered_answerable = 23`, `abstained_unanswerable = 12`;
- метрики близки к M7: P ≈ 0,621, R ≈ 0,848, page recall@5 ≈ 0,870,
  answer rate 1,0, accuracy ≈ 0,739, abstention ≈ 0,923, false answer ≈ 0,077,
  marker validity 1,0, uncited 0.

Точного совпадения не будет: температура 0,15, один прогон — одна выборка.
Если метрики отличаются **материально** (например, page recall@5 ниже 0,80
или answer rate ниже 0,90), остановитесь и разберитесь: другой индекс,
другая embedding-модель, другой commit, другой промпт, LM Studio вернула
не тот ответ. Не продолжайте сравнение политик на сломанном baseline.

### Шаг 2. Офлайн-проверить политики

Скопируйте скрипт из [Приложения A](#приложение-a-референсный-скрипт-офлайн-политик)
в `scripts/m7_citation_policies.py` на машине замера. Скрипт не входит в
package data и не меняет продакшн-код.

```powershell
.\.venv\Scripts\python.exe scripts\m7_citation_policies.py `
  --predictions eval/grounding/predictions_m7_baseline.jsonl `
  --out-json eval/grounding/m7_citation_policies.json `
  --out-md eval/grounding/m7_citation_policies.md
```

Скрипт считает пять политик:

| Политика | Правило | Роль |
|---|---|---|
| `identity` | оставить все цитаты | контроль, должен совпасть с baseline |
| `per_sentence_rank` | в каждом предложении оставить только минимальный id | самый простой кандидат |
| `dominant` | оставить самый частый primary-id (при равенстве — минимальный) | диагностика, ожидаемо агрессивная |
| `hybrid` | primary по предложению; схлопнуть в dominant, только если dominant совпадает с минимальным primary | основной кандидат |
| `single_lowest` | оставить только глобально минимальный id | диагностика, ожидаемо теряет recall |

`primary` — минимальный id цитаты в предложении; id соответствуют рангу
ретрива (1 — лучший фрагмент). Политики **не видят** `gold_sources`: они
работают только с текстом ответа, id и списком `retrieved`. Gold используется
только в аудите, чтобы показать, какие gold-цитаты были потеряны.

Проверьте в `m7_citation_policies.md`:

- `identity` совпадает с baseline по P, R, uncited, marker validity;
- для каждой политики есть P, R, uncited, marker, answer rate, accuracy,
  abstention, false answer, ΔP, ΔR;
- в аудите для каждого изменённого вопроса видны `removed_ids`,
  `lost_gold_ids`, `cited before→after`, `P before→after`, `R before→after`;
- `lost_gold_ids` пуст у консервативных кандидатов; если там есть gold,
  смотрите, сколько таких потерь и какой получается recall.

Для ручной проверки текста можно сохранить переписанные предсказания одной
политики:

```powershell
.\.venv\Scripts\python.exe scripts\m7_citation_policies.py `
  --predictions eval/grounding/predictions_m7_baseline.jsonl `
  --out-json eval/grounding/m7_citation_policies.json `
  --out-md eval/grounding/m7_citation_policies.md `
  --dump-policy hybrid `
  --dump-out eval/grounding/m7_policy_dump.jsonl
```

Откройте 2–3 ответа, где политика что-то удалила. Убедитесь, что маркеры
не превратились в `[]`, не повисли пробелы перед точкой и что удалённый id
действительно не был единственной опорой длинного предложения.

### Шаг 3. Выбрать политику

Правило выбора — **до** просмотра per-question результатов, по агрегатам:

1. Политика должна пройти **все** пороги H2-6:
   P ≥ 0,80, R ≥ 0,80, uncited ≤ 0,10, marker validity ≥ 1,0,
   answer rate ≥ 0,90, accuracy ≥ 0,70, abstention ≥ 0,90,
   false answer ≤ 0,10.
2. Предпочитайте более простое правило: `per_sentence_rank` → `hybrid`.
   `dominant` и `single_lowest` — диагностика; они не выбираются, если
   нарушают uncited или recall.
3. При равенстве — выше recall, затем ниже uncited, затем меньше изменённых
   вопросов.
4. Не выбирайте политику по принципу «она исправила `a13`, значит хорошая».
   Смотрите на все 23 вопроса и на `lost_gold_ids`.

Если ни одна политика не проходит, переходите к шагу 6 (prompt-only) или
к шагу 7 (отрицательный результат), не к «подкрутить правило под M7». Любая новая эвристика,
придуманная после просмотра per-question таблицы, должна быть описана как
post-hoc и проверена на всех вопросах; выдавать её за pre-registered нельзя.

### Шаг 4. Реализовать выбранную политику (только если она проходит)

Если офлайн-победитель есть, реализуйте **ту же** политику в продакшн-пути.
Минимальный вариант:

1. В `src/hipporag/qa/grounding.py` добавьте чистую функцию
   `select_citations(answer, used_ids, *, policy="hybrid") -> tuple[str, list[int]]`.
   Она не должна зависеть от LLM, индекса и gold; только текст, id и правило.
2. В `src/astro_app/grounded_qa.py` вызовите её после `parse_grounded_response`
   и после repair-retry, **до** возврата `GroundedAnswer`. Перепишите `text`
   и `used_citation_ids`; `abstained` и `errors` не меняйте. Если политика
   вернула пустой набор, оставьте исходный ответ (fail-safe).
3. Не меняйте `parse_grounded_response` для всех call-site'ов: это затронет
   `HippoRAG.rag_qa` и `StandardRAG`, которые не входят в M7. Если нужен
   единый контракт — отдельная задача после замера.
4. Добавьте unit-тесты в `tests/unit/test_qa_grounding.py` или
   `tests/unit/test_grounded_qa_helper.py`:
   - несколько цитат в одном предложении → остаётся минимальная;
   - несколько предложений с разными primary → `per_sentence_rank` сохраняет
     по одной на предложение;
   - `hybrid` не схлопывает, если dominant не совпадает с минимальным primary;
   - `b01`-подобный случай (обе цитаты gold, обе в разных предложениях)
     не теряет цитату;
   - ответ без цитат и отказ не ломаются;
   - пустой результат политики возвращает исходный ответ.
5. Обновите docstring и `docs/H2_6_GROUNDING_REPORT_2026-11-05.md` §9.

Если офлайн-победитель — `hybrid`, используйте ровно то правило, которое
проверялось: `primary = min(id)` в предложении; схлопывать в dominant только
при `len(unique) >= 2`, `max(count) >= 2` и `dominant == min(unique)`.
Не меняйте `>= 2` на `>= 3` по результатам M7 без отдельного обоснования.

### Шаг 5. Live-прогон выбранной политики

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:ASTRO_RETRIEVER_TOP_K = "5"
$env:ASTRO_RETRIEVER_PAGE_LEVEL = "1"
$env:LLM_API_NAME = "google/gemma-4-e4b"

.\.venv\Scripts\python.exe scripts\eval_grounding.py --run `
  --retriever grounding_astro_retriever:retrieve `
  --require-page-recall `
  --out-json eval/grounding/report_m7_policy.json `
  --out-md eval/grounding/report_m7_policy.md `
  --out-predictions eval/grounding/predictions_m7_policy.jsonl
```

Здесь **нельзя** `--no-thresholds`: команда должна вернуть код 0, то есть
пройти все пороги, включая page recall@5 ≥ 0,80. Если код 1 — откройте
`report_m7_policy.json`, найдите `thresholds` с `passed=false` и разберитесь,
что именно просело. Предсказания всё равно сохранятся, их не удаляйте.

Сравните с baseline:

- P ≥ 0,80 и R ≥ 0,80;
- uncited ≤ 0,10, marker validity 1,0;
- answer rate, accuracy, abstention, false answer — в порогах и без
  материального ухудшения относительно baseline;
- в per-question таблице нет потери gold на `b01`/`b02` и нет новых нулей
  там, где baseline был 1,0;
- `a10`, `a14`, `b03` остаются нулями — это ожидаемо, их не «чиним».

Если live-прогон дал другой результат, чем офлайн-симуляция, это нормально:
live — новая выборка LLM. Но если политика в коде отличается от проверенной
(например, вы поменяли правило при переносе), это уже другая политика;
вернитесь к шагу 3.

### Шаг 6. Prompt-only вариант «одна лучшая ссылка на утверждение»

Это отдельный эксперимент, а не замена офлайн-политике. Он нужен, если:

- ни одна офлайн-политика не прошла; **или**
- офлайн-политика прошла, но вы хотите проверить более простой путь без
  постобработки (тогда сначала зафиксируйте результат шага 5).

Порядок:

1. В `src/hipporag/prompts/templates/rag_qa_astrology.py` добавьте в
   `rag_qa_system` после блока формата:

   ```text
   Для каждого утверждения указывай ровно один самый релевантный фрагмент.
   Не перечисляй несколько номеров, если один фрагмент подтверждает утверждение.
   ```

2. Сделайте live-прогон с отдельными путями:

   ```powershell
   .\.venv\Scripts\python.exe scripts\eval_grounding.py --run `
     --retriever grounding_astro_retriever:retrieve `
     --require-page-recall `
     --out-json eval/grounding/report_m7_prompt_one.json `
     --out-md eval/grounding/report_m7_prompt_one.md `
     --out-predictions eval/grounding/predictions_m7_prompt_one.jsonl
   ```

3. Сравните **все** метрики, не только precision. Промпт может снизить
   recall, если модель выберет не-gold фрагмент, и может изменить accuracy.
4. Если вариант лучше и проходит пороги — оставьте изменение промпта,
   добавьте unit-тест на наличие инструкции и обновите документы. Если хуже —
   **откатите** промпт и запишите результат.

Не смешивайте prompt-only и постобработку в одном прогоне: иначе нельзя
понять, что дало эффект. Комбинацию можно проверить отдельным прогоном
только после того, как оба механизма измерены по отдельности.

### Шаг 7. Если ничего не прошло

Зафиксируйте отрицательный результат:

- полная таблица политик и prompt-only в `m7_citation_policies.md`;
- какие вопросы дали прирост, какие — потерю gold/recall;
- сколько gold-цитат потеряно и почему recall не удержался;
- вывод: селектор цитат сам по себе недостаточен; следующие гипотезы —
  изменить промпт, ограничить контекст, улучшить ретрив/rerank, или
  пересмотреть постановку citation precision.

**Не делайте** в этом случае:

- не меняйте `golden_qa.jsonl`, `gold_sources`, `gold_terms`;
- не меняйте определения метрик и пороги;
- не хардкодьте id вопросов;
- не удаляйте вопросы с нулями, чтобы поднять среднее;
- не выбирайте лучший из нескольких прогонов.

### Шаг 8. Документы, тесты, коммит

Обновите:

- `docs/H2_6_GROUNDING_REPORT_2026-11-05.md` §9 — статус, числа, ссылка на
  это задание и на артефакты;
- `docs/ROADMAP_REVISION_DEEPSEEK.md` — блок H2-6, статус citation precision;
- `docs/CHANGELOG_RU.md` — запись с датой и командами;
- `eval/dataset/RESULTS_REPORTED.md` — если там есть M7/H2-6.

Прогоните:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\test_qa_grounding.py `
  tests\unit\test_grounded_qa_helper.py tests\unit\test_eval_grounding.py -q
.\.venv\Scripts\python.exe scripts\check_docs_links.py
```

Коммит:

```powershell
git add eval/grounding/predictions_m7_baseline.jsonl `
        eval/grounding/report_m7_baseline.json `
        eval/grounding/report_m7_baseline.md `
        eval/grounding/m7_citation_policies.json `
        eval/grounding/m7_citation_policies.md
git add src/hipporag/qa/grounding.py src/astro_app/grounded_qa.py `
        tests/unit/test_qa_grounding.py tests/unit/test_grounded_qa_helper.py
git add docs/H2_6_GROUNDING_REPORT_2026-11-05.md `
        docs/ROADMAP_REVISION_DEEPSEEK.md docs/CHANGELOG_RU.md
git commit -m "meas(H2-6): citation precision M7 — политика отбора цитат"
git push origin main
```

Если политика не прошла, коммитьте только baseline, офлайн-отчёт и
документы; код не трогайте. Если прошла — добавьте
`predictions_m7_policy.jsonl`, `report_m7_policy.json`/`.md` и код.
`outputs/` в `.gitignore`, поэтому артефакты для коммита кладите в
`eval/grounding/`, а не в `outputs/reports/`.

---

## 7. Критерии приёмки

Задание выполнено, когда:

- [ ] `predictions_m7_baseline.jsonl` содержит **36** предсказаний, id
      совпадают с `golden_qa.jsonl`, нет пропусков и лишних id;
- [ ] baseline-отчёт воспроизводит M7 по ключевым метрикам или объясняет
      материальное расхождение (индекс, модель, commit, промпт);
- [ ] `m7_citation_policies.md` содержит все пять политик, агрегаты и
      per-question аудит;
- [ ] политики не используют `gold_sources`; это видно по коду и по тому,
      что `identity` не меняет метрики;
- [ ] если политика прошла: `predictions_m7_policy.jsonl` содержит 36
      предсказаний, live-отчёт проходит **все** пороги H2-6, команда
      `--require-page-recall` вернула 0;
- [ ] если политика прошла: P ≥ 0,80, R ≥ 0,80, uncited ≤ 0,10,
      marker validity 1,0, answer rate ≥ 0,90, accuracy ≥ 0,70,
      abstention ≥ 0,90, false answer ≤ 0,10;
- [ ] если политика прошла: unit-тесты добавлены и проходят, документы
      обновлены;
- [ ] если политика не прошла: отрицательный результат описан, код,
      golden set и пороги не менялись;
- [ ] команды воспроизведения и commit hash записаны;
- [ ] `check_docs_links.py` не находит битых ссылок;
- [ ] коммит и пуш выполнены, remote HEAD проверен.

---

## 8. Подводные камни

1. **Не подглядывать в gold из политики.** Политика видит только `answer`,
   `used_ids` и `retrieved`. Если она вызывает `gold_sources`, это уже
   подгонка, а не измерение.
2. **Не менять golden set и пороги.** Даже если кажется, что `a18` или `a07`
   размечены спорно: разметка уже проверена по корпусу
   ([H2_6_GROUNDING_REPORT](./H2_6_GROUNDING_REPORT_2026-11-05.md) §5.1).
   В этом задании набор заморожен.
3. **Не хардкодить id вопросов.** Правило должно быть общим. «Для `a13`
   оставить только id 1» — запрещено.
4. **Precision — макро-среднее по 23 вопросам.** Один вопрос весит ~0,043.
   Разрыв 4,1166 — это сумма, а не «чуть-чуть». Не гоняйтесь за одним
   вопросом в ущерб остальным.
5. **Запас recall — 1,1.** Потеря двух единственных gold-цитат уже
   проваливает recall. Смотрите `lost_gold_ids`, а не только итоговый P.
6. **Удаление цитаты может создать uncited-предложение.** Длинное
   предложение без маркера штрафуется. Проверяйте `uncited_sentence_rate`,
   а не только P/R.
7. **`citations` и inline-маркеры нужно обновлять вместе.** Если обновить
   только `citations`, precision изменится, а marker validity/uncited будут
   считаться по старому тексту. В офлайн-скрипте это уже сделано; в
   продакшн-коде сделайте так же.
8. **`ИСТОЧНИКИ` — не часть `answer`.** `source_mismatch` считается по
   сырому ответу и не изменится от переписывания inline-маркеров. Не
   «исправляйте» его редактированием предсказаний.
9. **`retrieved` может содержать дубли страниц.** id — это позиции, а не
   уникальные страницы. Сначала выбирайте id, потом дедуплицируйте источники
   для метрик.
10. **Baseline — одна выборка.** Температура 0,15, поэтому live-прогон
    не обязан совпасть с baseline до последнего знака. Сравнивайте политики
    на **одном** baseline-файле; live-прогон — отдельная выборка. Если
    число у границы, повторите 2–3 раза и покажите разброс, но не выбирайте
    лучший прогон.
11. **`--out-predictions` пишется до проверки порогов.** Код 1 не означает,
    что файла нет. Не удаляйте предсказания и отчёт.
12. **Разный индекс — разные числа.** Если embedding-модель или число
    чанков отличаются от M7, результаты не сравнимы напрямую. Запишите
    фактические параметры.
13. **CI.** Изменения только в `docs/**` и `*.md` не запускают pipeline
    из-за `paths-ignore`; изменения кода запускают. Перед пушем прогоните
    unit-тесты локально.
14. **Не путайте M7 и mode5.** `predictions_mode5.jsonl` — старый локальный
    полигон с другим ретривом и page recall 0,5652. Он годится только для
    отсева гипотез, не для доказательства порога.

---

## 9. Шаблон отчёта

````markdown
## H2-6: citation precision M7

**Дата:** ...
**Коммит:** ...
**Индекс:** ...
**LLM:** LM Studio, `google/gemma-4-e4b`
**`qa_top_k`:** 5
**Page-level:** 1
**Температура:** 0.15

### Baseline

| Метрика | M7 (опубликовано) | Baseline (этот прогон) |
|---|---:|---:|
| Citation precision | 0.621 | |
| Citation recall | 0.848 | |
| Page recall@5 | 0.870 | |
| Answer rate | 1.000 | |
| Accuracy | 0.739 | |
| Abstention | 0.923 | |
| False answer | 0.077 | |
| Marker validity | 1.000 | |
| Uncited | 0.000 | |

### Политики (офлайн, на baseline-предсказаниях)

| Политика | P | R | Uncited | Marker | Answer rate | Accuracy | Abstention | False answer | Gold lost | Решение |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| identity | | | | | | | | | | контроль |
| per_sentence_rank | | | | | | | | | | |
| dominant | | | | | | | | | | диагностика |
| hybrid | | | | | | | | | | |
| single_lowest | | | | | | | | | | диагностика |

### Выбранная политика и per-question аудит

| id | removed | lost gold | cited before→after | P before→after | R before→after |
|---|---|---|---:|---:|---:|
| ... | | | | | |

### Live-прогон

| Метрика | Baseline | Policy | Порог | PASS |
|---|---:|---:|---:|---|
| Citation precision | | | ≥ 0.80 | |
| Citation recall | | | ≥ 0.80 | |
| Page recall@5 | | | ≥ 0.80 | |
| Answer rate | | | ≥ 0.90 | |
| Accuracy | | | ≥ 0.70 | |
| Abstention | | | ≥ 0.90 | |
| False answer | | | ≤ 0.10 | |
| Marker validity | | | ≥ 1.00 | |
| Uncited | | | ≤ 0.10 | |

### Решение

- [ ] политика реализована в коде;
- [ ] политика не реализована, зафиксирован отрицательный результат.

**Почему:** ...

**Ограничения:** один прогон, температура 0.15, индекс ..., LM Studio ...,
commit ...

**Команды:**
```powershell
...
```
````

---

## Приложение A. Референсный скрипт офлайн-политик

Скрипт проверен на `predictions_mode5.jsonl` (локальный полигон, не M7):
`identity` совпадает с baseline, `per_sentence_rank` даёт P 0,7255 при
R 0,7647, `hybrid` — P 0,7647 при R 0,7647 и uncited 5,56 %,
`dominant`/`single_lowest` теряют recall и дают uncited 27,78 %.
Это только проверка механики; числа M7 будут другими.

Положите его в `scripts/m7_citation_policies.py`.

```python
#!/usr/bin/env python3
"""Offline citation-policy evaluation for grounding predictions.

Reads a predictions JSONL (for example, M7 baseline), applies deterministic
citation-selection policies, and reports metrics plus a per-question audit.

Policy functions never see gold sources: they use only the answer text,
citation IDs, and the retrieved fragment list. Gold is used only by the
evaluation/audit layer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set

ROOT = Path(os.getenv("M7_REPO_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import eval_grounding as eg  # noqa: E402

CITATION_RE = re.compile(r"\[\s*([0-9,\s]+?)\s*\]")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def ids_in(text: str) -> List[int]:
    ids: List[int] = []
    for group in CITATION_RE.findall(text or ""):
        ids.extend(int(part) for part in re.findall(r"\d+", group))
    return ids


def sentence_primaries(answer: str) -> List[int]:
    """Lowest citation ID in each sentence that has at least one marker."""
    primaries: List[int] = []
    for sentence in SENTENCE_RE.split(answer or ""):
        ids = list(dict.fromkeys(ids_in(sentence)))
        if ids:
            primaries.append(min(ids))
    return primaries


def rewrite_answer(answer: str, keep: Set[int]) -> str:
    """Removes citation IDs not in ``keep`` and tidies leftover punctuation."""

    def repl(match: re.Match) -> str:
        group_ids = [int(part) for part in re.findall(r"\d+", match.group(1))]
        kept = [str(cid) for cid in group_ids if cid in keep]
        return "[" + ", ".join(kept) + "]" if kept else ""

    text = CITATION_RE.sub(repl, answer)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def apply_keep(pred: eg.Prediction, keep: Set[int]) -> eg.Prediction:
    """Returns a prediction with only ``keep`` citation IDs."""
    if pred.abstained or not pred.answer.strip():
        return pred
    if not keep:
        # Fail-safe: an answer without any citation is worse than an extra one.
        return pred
    citations = [
        pred.retrieved[cid - 1]
        for cid in sorted(keep)
        if 1 <= cid <= len(pred.retrieved)
    ]
    return replace(pred, answer=rewrite_answer(pred.answer, keep), citations=citations)


@dataclass(frozen=True)
class Policy:
    name: str
    description: str
    keep: Callable[[eg.Prediction], Set[int]]


def keep_identity(pred: eg.Prediction) -> Set[int]:
    return set(ids_in(pred.answer))


def keep_per_sentence_rank(pred: eg.Prediction) -> Set[int]:
    return set(sentence_primaries(pred.answer))


def keep_dominant(pred: eg.Prediction) -> Set[int]:
    primaries = sentence_primaries(pred.answer)
    if not primaries:
        return set()
    counts: Dict[int, int] = {}
    for cid in primaries:
        counts[cid] = counts.get(cid, 0) + 1
    best = max(counts.values())
    return {min(cid for cid, count in counts.items() if count == best)}


def keep_hybrid(pred: eg.Prediction) -> Set[int]:
    primaries = sentence_primaries(pred.answer)
    if not primaries:
        return set()
    counts: Dict[int, int] = {}
    for cid in primaries:
        counts[cid] = counts.get(cid, 0) + 1
    unique = sorted(counts)
    if len(unique) >= 2:
        best = max(counts.values())
        if best >= 2:
            dominant = min(cid for cid, count in counts.items() if count == best)
            # Collapse only when the most frequent source is also the
            # highest-ranked one. This is the conservative variant tested
            # on the local mode5 polygon.
            if dominant == unique[0]:
                return {dominant}
    return set(unique)


def keep_single_lowest(pred: eg.Prediction) -> Set[int]:
    ids = ids_in(pred.answer)
    return {min(ids)} if ids else set()


POLICIES: Sequence[Policy] = (
    Policy("identity", "control: keep every cited ID", keep_identity),
    Policy(
        "per_sentence_rank",
        "one lowest-ID citation per sentence",
        keep_per_sentence_rank,
    ),
    Policy(
        "dominant",
        "most frequent per-sentence primary ID (tie: lowest)",
        keep_dominant,
    ),
    Policy(
        "hybrid",
        "per-sentence primary; collapse to dominant only if it is also the lowest",
        keep_hybrid,
    ),
    Policy(
        "single_lowest",
        "diagnostic: only the globally lowest cited ID (expected recall loss)",
        keep_single_lowest,
    ),
)


def source_of(pred: eg.Prediction, cid: int) -> eg.SourceRef | None:
    if 1 <= cid <= len(pred.retrieved):
        return pred.retrieved[cid - 1]
    return None


def gold_keys(question: eg.GoldenQuestion) -> Set[tuple[str, int]]:
    return {eg.source_key(source) for source in question.gold_sources}


def cited_gold_ids(pred: eg.Prediction, gold: Set[tuple[str, int]]) -> Set[int]:
    result: Set[int] = set()
    for cid in set(ids_in(pred.answer)):
        source = source_of(pred, cid)
        if source is not None and eg.source_key(source) in gold:
            result.add(cid)
    return result


def gold_keys_of_ids(
    pred: eg.Prediction, ids: Iterable[int], gold: Set[tuple[str, int]]
) -> Set[tuple[str, int]]:
    result: Set[tuple[str, int]] = set()
    for cid in ids:
        source = source_of(pred, cid)
        if source is not None and eg.source_key(source) in gold:
            result.add(eg.source_key(source))
    return result


def evaluate_policy(
    golden: eg.GoldenSet,
    predictions: Dict[str, eg.Prediction],
    policy: Policy,
) -> tuple[Dict[str, eg.Prediction], Dict[str, object]]:
    filtered = {
        qid: apply_keep(pred, policy.keep(pred)) for qid, pred in predictions.items()
    }
    result = eg.evaluate(golden, filtered)
    return filtered, result


def audit_policy(
    golden: eg.GoldenSet,
    predictions: Dict[str, eg.Prediction],
    filtered: Dict[str, eg.Prediction],
    policy: Policy,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    gold_sources_lost = 0
    precision_sum_delta = 0.0
    recall_sum_delta = 0.0
    for question in golden.items:
        if not question.answerable:
            continue
        before = predictions[question.id]
        after = filtered[question.id]
        if not eg.is_answered(before):
            continue
        gold = gold_keys(question)
        before_ids = set(ids_in(before.answer))
        after_ids = set(ids_in(after.answer))
        removed = sorted(before_ids - after_ids)
        before_gold = cited_gold_ids(before, gold)
        after_gold = cited_gold_ids(after, gold)
        lost_gold = sorted(before_gold - after_gold)
        if not removed and not lost_gold:
            continue
        before_precision = len(before_gold) / len(before_ids) if before_ids else 0.0
        after_precision = len(after_gold) / len(after_ids) if after_ids else 0.0
        before_recall = len(before_gold) / len(gold) if gold else 0.0
        after_recall = len(after_gold) / len(gold) if gold else 0.0
        precision_sum_delta += after_precision - before_precision
        recall_sum_delta += after_recall - before_recall
        lost_gold_keys = gold_keys_of_ids(before, lost_gold, gold)
        gold_sources_lost += len(lost_gold_keys)
        rows.append(
            {
                "id": question.id,
                "removed_ids": removed,
                "lost_gold_ids": lost_gold,
                "before_citations": len(before_ids),
                "after_citations": len(after_ids),
                "before_precision": round(before_precision, 4),
                "after_precision": round(after_precision, 4),
                "before_recall": round(before_recall, 4),
                "after_recall": round(after_recall, 4),
            }
        )
    return {
        "policy": policy.name,
        "gold_sources_lost": gold_sources_lost,
        "precision_sum_delta": round(precision_sum_delta, 4),
        "recall_sum_delta": round(recall_sum_delta, 4),
        "questions": rows,
    }


def policy_row(name: str, metrics: Dict[str, object], baseline: Dict[str, object]) -> str:
    def fmt(value: object) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    return (
        f"| {name} | {fmt(metrics.get('citation_precision'))} | "
        f"{fmt(metrics.get('citation_recall'))} | "
        f"{fmt(metrics.get('uncited_sentence_rate'))} | "
        f"{fmt(metrics.get('marker_validity'))} | "
        f"{fmt(metrics.get('answer_rate_answerable'))} | "
        f"{fmt(metrics.get('answer_accuracy'))} | "
        f"{fmt(metrics.get('abstention_rate_unanswerable'))} | "
        f"{fmt(metrics.get('false_answer_rate'))} | "
        f"{fmt((metrics.get('citation_precision') or 0) - (baseline.get('citation_precision') or 0))} | "
        f"{fmt((metrics.get('citation_recall') or 0) - (baseline.get('citation_recall') or 0))} |"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default="eval/grounding/predictions_m7_baseline.jsonl",
    )
    parser.add_argument("--golden", default="eval/grounding/golden_qa.jsonl")
    parser.add_argument(
        "--out-json",
        default="eval/grounding/m7_citation_policies.json",
    )
    parser.add_argument(
        "--out-md",
        default="eval/grounding/m7_citation_policies.md",
    )
    parser.add_argument(
        "--dump-policy",
        choices=[policy.name for policy in POLICIES],
        help="write one policy's rewritten predictions for manual inspection",
    )
    parser.add_argument(
        "--dump-out",
        default="eval/grounding/m7_policy_dump.jsonl",
    )
    args = parser.parse_args(argv)

    golden = eg.load_golden(args.golden)
    predictions = eg.load_predictions(args.predictions)
    baseline = eg.evaluate(golden, predictions)["metrics"]

    results = []
    audits = []
    for policy in POLICIES:
        filtered, result = evaluate_policy(golden, predictions, policy)
        results.append(
            {
                "policy": policy.name,
                "description": policy.description,
                "metrics": result["metrics"],
            }
        )
        audits.append(audit_policy(golden, predictions, filtered, policy))

    payload = {
        "predictions": args.predictions,
        "golden": args.golden,
        "baseline": baseline,
        "policies": results,
        "audit": audits,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.dump_policy:
        chosen = next(policy for policy in POLICIES if policy.name == args.dump_policy)
        filtered, _ = evaluate_policy(golden, predictions, chosen)
        dump_path = Path(args.dump_out)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(
            eg.predictions_to_jsonl(filtered.values()), encoding="utf-8"
        )
        print(f"dump: {args.dump_policy} -> {dump_path}")

    lines = [
        "# M7: офлайн-проверка политик отбора цитат",
        "",
        f"- Предсказания: `{args.predictions}`",
        f"- Golden set: `{args.golden}`",
        "",
        "## Сводка",
        "",
        "| Политика | P | R | Uncited | Marker | Answer rate | Accuracy | Abstention | False answer | ΔP | ΔR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        policy_row("baseline", baseline, baseline),
    ]
    for item in results:
        lines.append(policy_row(item["policy"], item["metrics"], baseline))
    lines += [
        "",
        "## Аудит по вопросам",
        "",
        "Учитываются только answerable-вопросы с ответом, где политика что-то удалила.",
        "`lost_gold_ids` — ID, которые были gold и были удалены (для аудита, не для выбора политики).",
        "",
    ]
    for audit in audits:
        lines.append(
            f"### {audit['policy']} (gold lost: {audit['gold_sources_lost']}, "
            f"ΔP sum: {audit['precision_sum_delta']}, ΔR sum: {audit['recall_sum_delta']})"
        )
        lines.append("")
        if not audit["questions"]:
            lines.append("Изменений нет.")
            lines.append("")
            continue
        lines += [
            "| id | removed | lost gold | cited before→after | P before→after | R before→after |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in audit["questions"]:
            lines.append(
                f"| `{row['id']}` | {row['removed_ids']} | {row['lost_gold_ids']} | "
                f"{row['before_citations']}→{row['after_citations']} | "
                f"{row['before_precision']}→{row['after_precision']} | "
                f"{row['before_recall']}→{row['after_recall']} |"
            )
        lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Приложение B. Эскиз продакшн-функции

Не копируйте вслепую: имена и импорты подгоните под текущий код. Смысл —
чистая функция без LLM и gold, которую можно покрыть unit-тестами.

```python
# src/hipporag/qa/grounding.py

def select_citations(
    answer: str,
    used_ids: Sequence[int],
    *,
    policy: str = "hybrid",
) -> tuple[str, list[int]]:
    """Оставляет только выбранные цитаты, не добавляя новых.

    ``used_ids`` — валидные id, уже проверенные парсером. Если политика
    не оставляет ни одного id, возвращается исходный ответ: лучше сохранить
    заземление, чем вернуть ответ без ссылок.
    """
    valid = sorted({cid for cid in used_ids if cid >= 1})
    if not answer.strip() or not valid:
        return answer, list(valid)
    # ... та же логика primary/dominant/hybrid, что в Приложении A ...
    keep = ...
    if not keep:
        return answer, list(valid)
    return _rewrite_citation_ids(answer, keep), sorted(keep)
```

Вызов в `src/astro_app/grounded_qa.py`:

```python
from dataclasses import replace

from hipporag.qa.grounding import select_citations

# после parse_grounded_response и repair-retry:
if not grounded.abstained:
    text, ids = select_citations(
        grounded.text, grounded.used_citation_ids, policy="hybrid"
    )
    grounded = replace(grounded, text=text, used_citation_ids=ids)
```

Проверьте, что после замены `grounded.is_grounded` не стал `False`:
`select_citations` не должна возвращать пустой набор при непустом исходном.

---

## Приложение C. Чек-лист исполнителя

- [x] `git rev-parse HEAD` записан, дерево чистое (`95df0af`).
- [x] LM Studio отвечает, модель `google/gemma-4-e4b` в `/v1/models`.
- [x] Индекс `ok`, 1369 чанков, 3 PDF; embedding-модель записана
      (`paraphrase-multilingual-mpnet-base-v2`).
- [x] `qa_top_k=5`, `ASTRO_RETRIEVER_PAGE_LEVEL=1`.
- [x] Baseline M7 сохранён: 36 предсказаний, отчёт; P 0,578 / R 0,848 /
      page recall 87,0 % — расхождения с M7 в пределах одной выборки.
- [x] Офлайн-скрипт прогнан, `identity` совпал с baseline.
- [x] Все пять политик в таблице, per-question аудит есть.
- [x] `lost_gold_ids` проверен: у `per_sentence_rank`/`hybrid` потеря 5 gold
      при запасе recall 1,1 — учтено в решении.
- [x] Политика выбрана по агрегатам: победителя нет, порог 0,80 не достигнут.
- [x] Live-прогон политики — не проводился (проходящей политики нет).
- [x] Код/тесты/доки политики — не менялись (отрицательный результат).
- [x] Отрицательный результат описан (Шаг 7): политики + prompt-only +
      арифметическое исключение комбинации; промпт откачен.
- [x] `check_docs_links.py` чист (57 файлов, 0 битых).
- [x] Коммит и пуш, remote HEAD проверен.

---

## Приложение D. Тексты вопросов из M7-таблицы

Ниже — дословные тексты из `eval/grounding/golden_qa.jsonl` для 15 вопросов
из §3.2. Они нужны, чтобы исполнитель понимал, что именно цитирует модель,
и мог вручную проверить 2–3 ответа. Политика **не должна** читать этот
раздел, `gold_answer` или `gold_sources`: это данные аудита, а не вход
правила. `gold_sources` менять нельзя.

### a01-astro-author-edition

- **Вопрос:** Кто автор книги «Астрофэншуй квартиры» и в каком году вышло это издание?
- **Gold-ответ:** Ирина Валериевна Максимова; переиздание 2025 года, «Издательские решения», 196 с.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 2.

### a02-psy-author

- **Вопрос:** Кто написал книгу «Психологический фэншуй»?
- **Gold-ответ:** Ирина Валериевна Максимова.
- **Gold-источники:** `Психологический фэншуй.pdf`, стр. 2.

### a03-psy-subtitle

- **Вопрос:** Какой подзаголовок у книги «Психологический фэншуй»?
- **Gold-ответ:** «Как прокачать „плохой“ фэншуй».
- **Gold-источники:** `Психологический фэншуй.pdf`, стр. 2.

### a06-eso-publisher-license

- **Вопрос:** Кто издал «Эзотерический фэншуй» и по какой лицензии?
- **Gold-ответ:** «Издательские решения», по лицензии Ridero.
- **Gold-источники:** `Эзотерический фэншуй.pdf`, стр. 2.

### a07-owen-zones

- **Вопрос:** Какие зоны квартиры соответствуют Овну и какая планета ими управляет?
- **Gold-ответ:** Овну соответствуют прихожая и плита, управитель — Марс.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 7.

### a10-scorpio-zones

- **Вопрос:** Какие зоны квартиры Скорпион считает своими «коронными» обителями?
- **Gold-ответ:** Туалет и прихожая.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 94.
- **Примечание:** gold-страница не попала в цитаты M7; селектор цитат
  этот вопрос не исправит.

### a13-gua1-shen-qi

- **Вопрос:** Какое направление Шен-ци у числа гуа 1?
- **Gold-ответ:** Юго-восток.
- **Gold-источники:** `Эзотерический фэншуй.pdf`, стр. 110.

### a14-bagua-squares

- **Вопрос:** Какие квадраты Багуа и их значения перечислены в оглавлении книги «Психологический фэншуй»?
- **Gold-ответ:** Северный — карьера, юго-западный — партнёрство, восточный — здоровье и семья, юго-восточный — богатство и везение, центральный — личностная самооценка, северо-западный — покровители, западный — творчество и дети, северо-восточный — знания, южный — слава.
- **Gold-источники:** `Психологический фэншуй.pdf`, стр. 3.
- **Примечание:** gold-страница не попала в цитаты M7; селектор цитат
  этот вопрос не исправит.

### a15-house4

- **Вопрос:** За что отвечает 4-й дом (IC) в системе домов?
- **Gold-ответ:** За модель семьи, родовые программы, помощь предков и прошлую карму; он делает дом уютным.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 139.

### a16-traditional-style

- **Вопрос:** Какой стиль фэншуй называется традиционным (академическим) и кому он подходит?
- **Gold-ответ:** Традиционный (академический) стиль — стиль поддерживающих отношений; к нему тяготеют люди, жаждущие поддержки и везения, желающие укрепить здоровье или получить образование.
- **Gold-источники:** `Психологический фэншуй.pdf`, стр. 19.

### a18-meta-zodiac

- **Вопрос:** Что делать, если человек не знает времени рождения и своего гороскопа?
- **Gold-ответ:** Использовать метод мета зодиака: взять знак Солнца и приравнять его к Асценденту.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 160.
- **Примечание:** precision можно поднять, но accuracy M7 на этом вопросе
  уже 0; это не задача селектора цитат.

### b02-shen-qi-gua1

- **Вопрос:** Что такое Шен-ци и какое направление оно занимает у числа гуа 1?
- **Gold-ответ:** Шен-ци — «производитель энергии ци», самое лучшее направление, начало цепочки удачи; у числа гуа 1 это юго-восток.
- **Gold-источники:** `Эзотерический фэншуй.pdf`, стр. 27 и 110.
- **Примечание:** единственная цитата M7 — gold; удалять её нельзя.

### b03-scorpio-taurus

- **Вопрос:** Сравни «коронные» обители Скорпиона со светской обителью Тельца.
- **Gold-ответ:** У Скорпиона это туалет и прихожая; у Тельца — кладовка, шкаф и холодильник.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 94 и 109.
- **Примечание:** gold-страницы не попали в цитаты M7; селектор цитат
  этот вопрос не исправит.

### b04-owen-planet-kitchen

- **Вопрос:** Какая планета управляет Овном и какая зона является его эзотерической обителью?
- **Gold-ответ:** Овном управляет Марс; эзотерическая обитель Овна — кухня (по другой классификации — бытовая техника).
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 7 и 110.

### b05-ryby-spalnya

- **Вопрос:** Что нужно Рыбам для ощущения своей обители и как автор использует это в споре о спальне?
- **Gold-ответ:** Рыбам нужны тайны и укромные места: спальня в нише за занавесочкой. В авторской правке спальня отдаётся Рыбам, если важна тайна.
- **Gold-источники:** `Астрофэншуй квартиры.pdf`, стр. 105 и 108.

> **Как пользоваться разделом.** После офлайн-прогона откройте 2–3
> изменённых ответа (например, `a13` и `a07`) и сверьте: удалённый id не был
> единственной опорой утверждения, в тексте не осталось `[]`, а `citations`
> в JSONL соответствует inline-маркерам. Не подбирайте правило под эти
> тексты — они приведены для проверки, а не для настройки.
