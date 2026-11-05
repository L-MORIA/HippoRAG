# Задание M2: citation coverage для legacy-режимов 12/19/21

**Дата:** 2026-11-05
**Статус:** выполнено 2026-11-05 — 0 ответов без цитаты у всех трёх режимов;
отчёт [MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md)
§10.1, сводка `outputs/reports/m2_legacy_summary.md`
**Оценка:** 2–4 часа (36 вопросов × 3 режима + настройка)
**Исполнитель:** агент на машине, где доступны индекс `outputs/astrology_test`
и LM Studio с `google/gemma-4-e4b`

**Связанные документы:**
[ROADMAP_REVISION_DEEPSEEK.md](./ROADMAP_REVISION_DEEPSEEK.md) (H2-6),
[MEASUREMENT_PLAN_2026-11-05.md](./MEASUREMENT_PLAN_2026-11-05.md) (M2),
[MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md) (§10),
[H2_6_GROUNDING_REPORT_2026-11-05.md](./H2_6_GROUNDING_REPORT_2026-11-05.md).

> **Коротко.** Нужно отдельно измерить M2 — долю ответов без цитаты — для
> трёх legacy-режимов: **12 Self-RAG**, **19 GraphRAG**, **21 DAT+LLM**.
> Сейчас M2 закрыта по данным M7 на продуктовом пути (24 из 24 ответов с
> цитатой), но старые режимы отдельно не гонялись. Это задание закрывает
> оговорку реальными предсказаниями, отчётами и командой воспроизведения.

---

## 1. Зачем это отдельное измерение

M2 в [MEASUREMENT_REPORT_2026-11-05.md](./MEASUREMENT_REPORT_2026-11-05.md)
§10 посчитана по per-question данным M7: продуктовый путь «режим 5 +
page-level H1-6 + grounding». Это корректная метрика продукта, но не
историческая сверка со старыми генеративными режимами.

H2-6 изначально требовал единый контракт цитирования для режимов 12/19/21.
Чтобы утверждение «ответ без цитаты = 0 %» относилось и к ним, нужен
отдельный прогон на том же golden set. Задание не меняет продуктовый путь и
не переписывает M7 — оно добавляет независимый замер.

**Важно:** успех задания — это не обязательно «0 % у всех». Успех — это
воспроизводимые числа, сырые предсказания и честно описанные ограничения.
Если режим даёт ответы без цитат, это фиксируется как дефект режима, а не
подгоняется под цель.

---

## 2. Что должно получиться

1. Три файла предсказаний:
   - `eval/grounding/predictions_mode12.jsonl`
   - `eval/grounding/predictions_mode19.jsonl`
   - `eval/grounding/predictions_mode21.jsonl`
2. Три отчёта `eval_grounding.py`:
   - `outputs/reports/m2_mode12.json` + `outputs/reports/m2_mode12.md`
   - `outputs/reports/m2_mode19.json` + `outputs/reports/m2_mode19.md`
   - `outputs/reports/m2_mode21.json` + `outputs/reports/m2_mode21.md`
3. Сводная таблица `outputs/reports/m2_legacy_summary.md`.
4. Сырые ответы режимов (для аудита):
   `outputs/reports/m2_mode{12,19,21}_raw.jsonl`.
5. Обновлённые документы с фактическими числами и точной командой:
   `docs/MEASUREMENT_REPORT_2026-11-05.md` §10,
   `eval/dataset/RESULTS_REPORTED.md`,
   `docs/ROADMAP_REVISION_DEEPSEEK.md` (H2-6),
   `docs/CHANGELOG_RU.md`.

---

## 3. Определение метрики

| Понятие | Определение |
|---|---|
| `answered` | `answer.strip()` непустой, `abstained=false`, ответ не равен `ABSTAIN_TEXT` |
| `citation` | источник `(doc_title, page_num)`, полученный из маркера `[n]` и привязанный к фрагменту `n` |
| Валидная цитата | `1 ≤ n ≤ fragment_count`, у фрагмента есть `page_num ≥ 1` |
| **M2 (primary)** | `answered_without_citation / answered`; цель **0 %** |
| `marker_validity` | валидные маркеры / все маркеры среди ответов; `N/A`, если маркеров нет или `fragment_count` неизвестен |
| `citation_precision` | средняя доля цитат, попавших в `gold_sources`, по answerable-вопросам с ответом |
| `citation_recall` | средняя доля `gold_sources`, покрытых цитатами |
| `answer_rate` | ответы на answerable-вопросы / все answerable |
| `abstention_rate` | отказы на unanswerable-вопросы / все unanswerable |
| `false_answer_rate` | ответы на unanswerable-вопросы / все unanswerable |

**Отказы не входят в знаменатель M2.** Вопрос, на который режим отказался
отвечать, не является «ответом без цитаты». Если режим ответил 0 вопросов,
M2 = `N/A`, а не `0 %`.

**Что считать ответом.** Если режим вернул не ответ, а список найденных
фрагментов (retrieval fallback), это **не** отказ: он показывается
пользователю как выдача. Такой случай считается ответом без цитаты и
помечается `grounding_errors: ["retrieval_fallback_without_answer"]`.
Не переклассифицируйте его в отказ, иначе метрика улучшится искусственно.

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
> `..._nvidia_NV-Embed-v2` — это другой embedding-модель и другой индекс.
> Запишите фактический путь и модель в отчёт; **не сравнивайте** такие числа
> с M7 напрямую. Для чистого замера нужен индекс с
> `paraphrase-multilingual-mpnet-base-v2`.

---

## 5. Данные

`eval/grounding/golden_qa.jsonl` — 36 вопросов:

- 23 answerable, из них 5 borderline;
- 13 unanswerable;
- у каждого вопроса есть `gold_answer`, `gold_sources`, `gold_terms`.

Датасет не менять. Вопросы с `borderline=true` входят в answerable и
учитываются в M2, если режим на них ответил.

---

## 6. Порядок выполнения

### Шаг 1. Положить адаптер

Скопируйте референсную реализацию из [Приложения A](#приложение-a-референсная-реализация-адаптера)
в `scripts/m2_legacy_modes.py` на машине замера. Скрипт не входит в
package data и не меняет продакшн-код.

Если пишете свой адаптер, он обязан:

1. инициализировать `HippoRAG` так же, как `scripts/eval_search_quality.py`
   (см. `load_hipporag` в приложении);
2. вызывать `MODES[mode_id].fn`, передавая `top_k=5` только если функция
   принимает `top_k`;
3. перехватывать фрагменты, переданные в `answer_grounded`, а не парсить
   обрезанную секцию `=== НАЙДЕННЫЕ ФРАГМЕНТЫ ===` (она обрезана до 3000
   символов);
4. писать JSONL строго с полями `id`, `answer`, `abstained`, `citations`,
   `retrieved`, `fragment_count`, `grounding_errors`, `latency_ms`;
5. сохранять сырой ответ режима отдельно: `eval_grounding.py` не сохраняет
   исходный текст ответа и перехваченные фрагменты в отчёте.

### Шаг 2. Дымовой прогон на 2 вопросах

```powershell
.\.venv\Scripts\python.exe scripts\m2_legacy_modes.py `
  --mode 12 --limit 2 `
  --out outputs/reports/m2_mode12_smoke.jsonl `
  --raw-out outputs/reports/m2_mode12_smoke_raw.jsonl
```

Проверьте:

- в JSONL две строки, `id` совпадают с первыми двумя id golden set;
- `answer` непустой или `abstained=true`;
- `citations` — список объектов `{"doc_title": ..., "page_num": ...}`;
- `grounding_errors` не содержит неожиданных ошибок.

Повторите для режимов 19 и 21. Если дымовой прогон падает, не запускайте
полный: сначала исправьте адаптер.

### Шаг 3. Полный прогон

```powershell
.\.venv\Scripts\python.exe scripts\m2_legacy_modes.py `
  --mode 12 `
  --out eval/grounding/predictions_mode12.jsonl `
  --raw-out outputs/reports/m2_mode12_raw.jsonl

.\.venv\Scripts\python.exe scripts\m2_legacy_modes.py `
  --mode 19 `
  --out eval/grounding/predictions_mode19.jsonl `
  --raw-out outputs/reports/m2_mode19_raw.jsonl

.\.venv\Scripts\python.exe scripts\m2_legacy_modes.py `
  --mode 21 `
  --out eval/grounding/predictions_mode21.jsonl `
  --raw-out outputs/reports/m2_mode21_raw.jsonl
```

Каждый прогон — 36 вопросов. Один вопрос может занимать 10–60 секунд;
полный прогон трёх режимов обычно 30–90 минут.

### Шаг 4. Оценка

```powershell
.\.venv\Scripts\python.exe scripts\eval_grounding.py `
  --predictions eval/grounding/predictions_mode12.jsonl `
  --out-json outputs/reports/m2_mode12.json `
  --out-md outputs/reports/m2_mode12.md `
  --no-thresholds
```

Повторите для 19 и 21. `--no-thresholds` нужен потому, что legacy-режимы
не обязаны проходить пороги H2-6 (page recall, citation precision); нам
нужен отчёт, а не падение команды. Метрики всё равно считаются.

### Шаг 5. Сводная таблица

```powershell
.\.venv\Scripts\python.exe - <<'PY'
import json
from pathlib import Path

rows = []
for mode in ("12", "19", "21"):
    report = json.loads(
        Path(f"outputs/reports/m2_mode{mode}.json").read_text(encoding="utf-8")
    )
    metrics = report["metrics"]
    questions = report["questions"]
    answered = [q for q in questions if q["answered"]]
    without = [q for q in answered if q["citations"] == 0]
    coverage = (
        f"{100 * (1 - len(without) / len(answered)):.1f} %"
        if answered else "N/A"
    )
    rows.append(
        {
            "mode": mode,
            "answered": len(answered),
            "with_citation": len(answered) - len(without),
            "without_citation": len(without),
            "coverage": coverage,
            "marker_validity": metrics["marker_validity"],
            "answer_rate": metrics["answer_rate_answerable"],
            "abstention": metrics["abstention_rate_unanswerable"],
            "false_answer": metrics["false_answer_rate"],
            "precision": metrics["citation_precision"],
            "recall": metrics["citation_recall"],
            "p50": metrics["p50_latency_ms"],
            "p95": metrics["p95_latency_ms"],
        }
    )

header = (
    "| Режим | Ответов | С цитатой | Без цитаты | M2 coverage | "
    "Marker validity | Answer rate | Abstention | False answer | "
    "Cit. precision | Cit. recall | p50, мс | p95, мс |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
)
lines = [header]
for row in rows:
    lines.append(
        f"| {row['mode']} | {row['answered']} | {row['with_citation']} | "
        f"{row['without_citation']} | {row['coverage']} | "
        f"{row['marker_validity']} | {row['answer_rate']} | "
        f"{row['abstention']} | {row['false_answer']} | "
        f"{row['precision']} | {row['recall']} | "
        f"{row['p50']} | {row['p95']} |\n"
    )
Path("outputs/reports/m2_legacy_summary.md").write_text(
    "".join(lines), encoding="utf-8"
)
print("".join(lines))
PY
```

На Windows heredoc может не работать в PowerShell. Тогда сохраните snippet
в `scripts/m2_legacy_summary.py` и запустите
`.\.venv\Scripts\python.exe scripts\m2_legacy_summary.py`.

### Шаг 6. Обновить документы

В `docs/MEASUREMENT_REPORT_2026-11-05.md` §10 добавьте три строки (12/19/21)
с фактическими числами и оговоркой об индексе. В
`eval/dataset/RESULTS_REPORTED.md` обновите раздел M2. В
`docs/ROADMAP_REVISION_DEEPSEEK.md` в H2-6 замените «legacy отдельно не
гонялись» на результат или ссылку на отчёт. В `docs/CHANGELOG_RU.md`
добавьте запись с датой, режимами и командами.

### Шаг 7. Закоммитить и запушить

```powershell
git add eval/grounding/predictions_mode12.jsonl `
        eval/grounding/predictions_mode19.jsonl `
        eval/grounding/predictions_mode21.jsonl
git add -f outputs/reports/m2_mode12.json outputs/reports/m2_mode12.md `
          outputs/reports/m2_mode19.json outputs/reports/m2_mode19.md `
          outputs/reports/m2_mode21.json outputs/reports/m2_mode21.md `
          outputs/reports/m2_legacy_summary.md
git add docs/MEASUREMENT_REPORT_2026-11-05.md `
        eval/dataset/RESULTS_REPORTED.md `
        docs/ROADMAP_REVISION_DEEPSEEK.md `
        docs/CHANGELOG_RU.md
git commit -m "meas(M2): citation coverage legacy-режимов 12/19/21"
git push origin main
```

Сырые `*_raw.jsonl` можно приложить к коммиту, если размер разумный
(ориентир — до 1 МБ). Если больше, оставьте их локально и укажите путь в
отчёте.

---

## 7. Критерии приёмки

Задание выполнено, когда:

- [ ] для каждого режима 12/19/21 есть JSONL с **36** предсказаниями;
- [ ] `id` предсказаний совпадают с `golden_qa.jsonl` (нет пропусков и
      лишних id);
- [ ] для каждого режима посчитана M2 (доля ответов без цитаты) и
      `marker_validity`;
- [ ] в отчёте указано, сколько вопросов отвечено, сколько отказов и
      сколько ответов без цитаты;
- [ ] сырые ответы и команды воспроизведения приложены;
- [ ] ограничения описаны: индекс, embedding-модель, версия коммита,
      температура LLM, один прогон;
- [ ] числа не заменены данными M7 и не подогнаны под 0 %.

Если у режима 0 ответов, M2 = `N/A`; это допустимый результат, но его
нужно объяснить (например, режим всегда отказывает или падает).

---

## 8. Подводные камни

1. **Режим 12 и 21** возвращают структурированный текст с секциями
   `=== ОТВЕТ ===`, `=== ИСТОЧНИКИ ===`, `=== НАЙДЕННЫЕ ФРАГМЕНТЫ ===`.
   Секция фрагментов обрезана до 3000 символов, поэтому парсить её для
   `fragment_count` нельзя — перехватывайте `fragments` в `answer_grounded`.
2. **Режим 19** может вернуть:
   - структурированный ответ Self-RAG (если выбрал этот метод);
   - обычный заземлённый ответ без секций;
   - список найденных фрагментов, если LLM не дала ответ.
   Первые два случая — ответ; третий — ответ без цитаты с пометкой
   `retrieval_fallback_without_answer`.
3. **Режим 21** при пустом ответе LLM может вернуть retrieval fallback
   (`combined[:top_k]`). Это тоже ответ без цитаты, не отказ.
4. **Режим 19** в шаге переформулировки использует `config.LLM_NAME`
   (имя каталога `gemma-4-e4b`), а не `LLM_API_NAME`
   (`google/gemma-4-e4b`). Если LM Studio отклонит запрос, исключение
   проглатывается, и переформулировка просто не выполнится. Это
   pre-existing поведение; **не исправляйте его во время замера**, но
   запишите в ограничениях, если увидели.
5. **`qa_top_k=5`.** Не поднимайте до 8: по данным H2-6 рост контекста
   добавляет шум быстрее, чем доказательства (false answer 7,7 % → 15,4 %,
   accuracy режима 5 падает).
6. **Температура 0.15.** Один прогон — это одна выборка. Если число
   близко к границе, можно повторить 2–3 раза и показать разброс, но
   нельзя выбирать лучший прогон.
7. **Разный индекс.** Если embedding-модель или число чанков отличаются от
   M7, числа не сравнимы напрямую. Запишите фактические параметры.
8. **M2 — это наличие цитаты, а не её правильность.** Citation precision и
   recall — вторичные метрики; не подменяйте ими M2.

---

## 9. Шаблон отчёта

```markdown
## M2 — citation coverage legacy-режимов 12/19/21

**Дата:** ...
**Коммит:** ...
**Индекс:** ...
**LLM:** LM Studio, `google/gemma-4-e4b`
**`qa_top_k`:** 5

| Режим | Ответов | С цитатой | Без цитаты | M2 coverage | Marker validity | Answer rate | Abstention | False answer | Cit. precision | Cit. recall | p50, мс | p95, мс |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 Self-RAG | | | | | | | | | | | | |
| 19 GraphRAG | | | | | | | | | | | | |
| 21 DAT+LLM | | | | | | | | | | | | |

**Команды:**
```powershell
...
```

**Ограничения:**
- ...
```

---

## Приложение A. Референсная реализация адаптера

Скрипт проверен на синтаксис и на синтетических ответах; живой прогон
требует LM Studio и индекс. Положите его в `scripts/m2_legacy_modes.py`.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2: предсказания legacy-режимов 12/19/21 для eval_grounding.py.

Референсная реализация из docs/TASK_M2_LEGACY_MODES_2026-11-05.md.
Скрипт не входит в package data; положите его в scripts/ на машине замера.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from astro_app.config import LLM_BASE_URL, LLM_NAME  # noqa: E402
from astro_app.modes import MODES  # noqa: E402
from astro_app.modes import graph as graph_mode  # noqa: E402
from astro_app.modes import self_rag as self_rag_mode  # noqa: E402
from astro_app.modes import sentence_dat as sentence_dat_mode  # noqa: E402
from hipporag.qa.grounding import ABSTAIN_TEXT  # noqa: E402

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
QA_TOP_K = 5
MARKER_RE = re.compile(r"\[\s*([0-9,\s]+?)\s*\]")
SECTION_RE = re.compile(r"===\s*([А-ЯЁ ]+?)\s*===")
SOURCE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?),\s*стр\.\s*(\d+)", re.M)


def read_jsonl(path: Path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: {exc}") from exc
    return items


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


def split_sections(text: str) -> dict[str, str]:
    parts = SECTION_RE.split(text or "")
    sections: dict[str, str] = {}
    for index in range(1, len(parts) - 1, 2):
        sections[parts[index].strip()] = parts[index + 1].strip()
    return sections


def raw_to_text(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        if not raw:
            return ""
        first = raw[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        return str(first)
    return str(raw)


def extract_answer(text: str) -> str:
    sections = split_sections(text)
    if "ОТВЕТ" in sections:
        return sections["ОТВЕТ"].strip()
    return (text or "").strip()


def parse_marker_ids(text: str) -> list[int]:
    ids: set[int] = set()
    for group in MARKER_RE.findall(text or ""):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
    return sorted(ids)


def parse_source_lines(text: str) -> list[dict]:
    return [
        {"doc_title": match.group(2).strip(), "page_num": int(match.group(3))}
        for match in SOURCE_LINE_RE.finditer(text or "")
    ]


def source_of(fragment) -> dict:
    return {
        "doc_title": str(getattr(fragment, "doc_title", "") or "Источник"),
        "page_num": int(getattr(fragment, "page_num", 0) or 0),
    }


def has_page(source: dict) -> bool:
    return source.get("page_num", 0) >= 1


def dedupe(sources: list[dict]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    result: list[dict] = []
    for source in sources:
        key = (source["doc_title"], source["page_num"])
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result


class GroundedSpy:
    """Перехватывает fragments, переданные в answer_grounded."""

    def __init__(self, module):
        self.module = module
        self.original = module.answer_grounded
        self.fragments = []
        self.grounded = None
        self.calls = 0

    def __enter__(self):
        spy = self

        def wrapper(question, fragments, *args, **kwargs):
            spy.fragments = list(fragments)
            result = spy.original(question, fragments, *args, **kwargs)
            spy.grounded = result
            spy.calls += 1
            return result

        self.module.answer_grounded = wrapper
        return self

    def __exit__(self, *exc_info):
        self.module.answer_grounded = self.original
        return False


def run_one(mode_id: str, hipporag, question: str):
    spec = MODES[mode_id]
    spies = [
        GroundedSpy(graph_mode),
        GroundedSpy(self_rag_mode),
        GroundedSpy(sentence_dat_mode),
    ]
    for spy in spies:
        spy.__enter__()

    started = time.perf_counter()
    error = None
    raw = None
    try:
        kwargs = {}
        if "top_k" in inspect.signature(spec.fn).parameters:
            kwargs["top_k"] = QA_TOP_K
        raw = spec.fn(hipporag, question, **kwargs)
    except Exception as exc:  # один сбойный вопрос не должен ронять прогон
        error = f"{type(exc).__name__}: {exc}"
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        for spy in reversed(spies):
            spy.__exit__(None, None, None)

    text = raw_to_text(raw)
    answer = extract_answer(text)
    by_module = {spy.module.__name__: spy for spy in spies}
    if mode_id == "19":
        preferred = by_module.get("astro_app.modes.graph")
        fallback = by_module.get("astro_app.modes.self_rag")
        spy = preferred if preferred and preferred.calls else fallback
    elif mode_id == "12":
        spy = by_module.get("astro_app.modes.self_rag")
    else:
        spy = by_module.get("astro_app.modes.sentence_dat")

    fragments = list(spy.fragments) if spy else []
    grounded = spy.grounded if spy else None
    abstained = bool(getattr(grounded, "abstained", False))
    if not answer.strip() or answer.strip().startswith(ABSTAIN_TEXT):
        abstained = True

    errors: list[str] = []
    if error:
        errors.append(error)

    sections = split_sections(text)
    list_of_tuples = (
        isinstance(raw, (list, tuple))
        and bool(raw)
        and isinstance(raw[0], (list, tuple))
    )
    first_score = None
    if list_of_tuples and len(raw[0]) > 1:
        try:
            first_score = float(raw[0][1])
        except (TypeError, ValueError):
            first_score = None
    # Ответы режимов помечены сентинелом 999.0; retrieval fallback — обычные
    # чанки с реальным score. Fallback показывается пользователю как выдача,
    # поэтому это ответ без цитаты, а не отказ.
    retrieval_fallback = (
        list_of_tuples
        and "ОТВЕТ" not in sections
        and first_score != 999.0
    )
    if retrieval_fallback:
        errors.append("retrieval_fallback_without_answer")
        abstained = False

    marker_ids = parse_marker_ids(answer)
    citations: list[dict] = []
    retrieved: list[dict] = []
    fragment_count = None

    if fragments:
        fragment_count = len(fragments)
        retrieved = [
            source_of(fragment)
            for fragment in fragments
            if has_page(source_of(fragment))
        ]
        for marker_id in marker_ids:
            if not 1 <= marker_id <= len(fragments):
                errors.append(f"marker_{marker_id}_out_of_range")
                continue
            source = source_of(fragments[marker_id - 1])
            if has_page(source):
                citations.append(source)
            else:
                errors.append(f"marker_{marker_id}_fragment_without_page")
    else:
        parsed_fragments = parse_source_lines(
            sections.get("НАЙДЕННЫЕ ФРАГМЕНТЫ", "")
        )
        parsed_sources = parse_source_lines(sections.get("ИСТОЧНИКИ", ""))
        if parsed_fragments:
            fragment_count = len(parsed_fragments)
            retrieved = parsed_fragments
            errors.append("fragments_parsed_from_text")
            for marker_id in marker_ids:
                if 1 <= marker_id <= len(parsed_fragments):
                    citations.append(parsed_fragments[marker_id - 1])
                else:
                    errors.append(f"marker_{marker_id}_out_of_range")
        elif parsed_sources:
            citations = parsed_sources
            retrieved = parsed_sources
            errors.append("fragments_parsed_from_sources")
        elif marker_ids:
            errors.append("markers_without_fragments")

    prediction = {
        "id": "",
        "answer": answer,
        "abstained": abstained,
        "citations": dedupe(citations),
        "retrieved": dedupe(retrieved),
        "fragment_count": fragment_count,
        "grounding_errors": sorted(set(errors)),
        "latency_ms": latency_ms,
    }
    raw_row = {
        "id": "",
        "raw": text,
        "captured_fragments": [source_of(fragment) for fragment in fragments],
        "captured_grounded_abstained": bool(
            getattr(grounded, "abstained", False)
        ),
        "captured_calls": spy.calls if spy else 0,
        "error": error,
    }
    return prediction, raw_row


def load_hipporag(index: str, embedding_model: str, llm_name: str):
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=index,
        llm_name=llm_name,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=embedding_model,
        linking_top_k=100,
        retrieval_top_k=200,
        passage_node_weight=0.05,
        embedding_batch_size=8,
    )
    return HippoRAG(global_config=config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="M2 predictions для режимов 12/19/21"
    )
    parser.add_argument("--mode", required=True, choices=["12", "19", "21"])
    parser.add_argument("--golden", default="eval/grounding/golden_qa.jsonl")
    parser.add_argument("--index", default="outputs/astrology_test")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--llm-name", default=LLM_NAME)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    questions = read_jsonl(Path(args.golden))
    if args.limit:
        questions = questions[: args.limit]

    hipporag = load_hipporag(args.index, args.embedding_model, args.llm_name)
    predictions = []
    raw_rows = []
    try:
        for index, question in enumerate(questions, 1):
            print(
                f"[{args.mode}] {index}/{len(questions)} {question['id']}",
                flush=True,
            )
            prediction, raw_row = run_one(
                args.mode, hipporag, question["question"]
            )
            prediction["id"] = question["id"]
            raw_row["id"] = question["id"]
            predictions.append(prediction)
            raw_rows.append(raw_row)
    finally:
        close = getattr(hipporag, "close", None)
        if callable(close):
            close()

    write_jsonl(Path(args.out), predictions)
    write_jsonl(Path(args.raw_out), raw_rows)
    print(f"Готово: {len(predictions)} предсказаний -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## Приложение B. Чек-лист исполнителя

- [x] `git rev-parse HEAD` записан (`50cafbc`).
- [x] LM Studio отвечает, модель `google/gemma-4-e4b` в `/v1/models`.
- [x] Индекс `ok`, 1369 чанков, 3 PDF; embedding-модель записана
      (`paraphrase-multilingual-mpnet-base-v2`).
- [x] `qa_top_k=5`, `.env` с `LLM_NAME` и `LLM_API_NAME`.
- [x] Дымовой прогон 2 вопросов для 12/19/21 прошёл.
- [x] Полные прогоны: по 36 предсказаний на режим.
- [x] `eval_grounding.py` отработал для трёх JSONL.
- [x] Сводная таблица собрана.
- [x] Сырые ответы сохранены.
- [x] Документы обновлены, ссылки проверены.
- [x] Коммит и пуш; в отчёте есть команда воспроизведения.
