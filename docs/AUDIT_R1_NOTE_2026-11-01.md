# Замечание по R-1′ — сужение применено к одному хуку из шести

**Дата:** 2026-11-01
**От:** Mavis (проверка финального варианта)
**К:** исполнитель сводного плана аудита
**Предмет:** закрытие R-1′ (дефект `pre-commit`-хука ruff из вердикта `docs/AUDIT_EXECUTION_VERDICT_R2_2026-11-01.md`, раздел 3)

---

## 1. Что подтверждено как сделанное

**R-1′ для хука ruff закрыт корректно.** `.pre-commit-config.yaml:28-36`:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  hooks:
    - id: ruff
      # ...поясняющий комментарий...
      files: ^(src|tests)/
```

Проверено живым прогоном:

```
pre-commit run ruff --all-files   →   ruff.....Passed   (exit 0)
```

Раньше этот же хук давал `exit 1, 59 errors`. Сужение до области CI (`src/` + `tests/`) — ровно та область, которую проверяет CI-шаг `ruff check src/ tests/`. **Выбор варианта №2 сделан правильно, и для ruff — полностью.**

Также подтверждено, что в области CI всё чисто:

| Проверка (область `src` + `tests`) | Результат |
|-----------------------------------|-----------|
| `black --check src/ tests/` | exit 0, 107 файлов unchanged |
| `isort --check-only src/ tests/` | exit 0 |
| `ruff check src tests` | All checks passed |
| `pytest tests/unit` | 62 passed |

---

## 2. Замечание 1 (блокирующее) — тот же дефект остался в трёх других хуках

**Аргумент.** R-1′ был не «ошибкой в одном хуке», а **несогласованностью ворот качества**: CI проверяет `src/` + `tests/`, а pre-commit — всё дерево. Сужение `files:` устраняет симптом только в том хуке, где симптом заметили. Пока остальные три хука не сужены, `pre-commit run --all-files` остаётся красным — то есть **блокировка коммитов, ради снятия которой правка и делалась, сохраняется**.

**Доказательства** (произведены на текущем дереве):

| Хук | Строки | Фильтр | Результат |
|-----|--------|--------|-----------|
| `ruff` | 28-36 | `files: ^(src|tests)/` | ✅ Passed |
| `black` | 15-20 | **нет** | ❌ exit 1 — «3 files would be reformatted»: `main.py`, `main_azure.py`, `main_dpr.py` |
| `isort` | 22-26 | **нет** | ❌ exit 1 — те же 3 файла: «Imports are incorrectly sorted and/or formatted» |
| `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending` | 2-13 | **нет** | ❌ exit 1 — авто-исправили 2 файла |

То есть у трёх корневых скриптов одновременно не выполняются и форматирование, и сортировка импортов. Это не «стиль по вкусу»: `black` и `isort` — ровно те же инструменты, что уже настроены в проекте и уже прогнаны по `src`/`tests`; корневые скрипты просто оказались вне их области.

**Что предлагаю (выбор за владельцем, оба варианта рабочие):**

- **Вариант A — сузить все python-хуки** (рекомендую, симметрично CI):
  ```yaml
  - id: black
    files: ^(src|tests)/
  - id: isort
    files: ^(src|tests)/
  ```
  Хуки `pre-commit-hooks` — либо `files: ^(src|tests)/`, либо исключить из них `reproduce/` и `archive/` (см. замечание 4).
- **Вариант B — расширить ворота на корневые скрипты**: тогда в CI добавляется шаг `ruff check .` / `black --check .`, и `main.py` / `main_azure.py` / `main_dpr.py` надо привести к формату (`ruff check --fix` закрывает F401/F541/F841 автоматически; B023 — 23 штуки — лямбды вызываются в той же итерации, то есть на поведение не влияют, достаточно зафиксировать или убрать предупреждение).

---

## 3. Замечание 2 (блокирующее) — `language_version: python3.10` делает pre-commit неработоспособным на dev-машине

**Доказательство.** Полный прогон не доходит до проверок:

```
$ pre-commit run --all-files
An unexpected error has occurred: CalledProcessError:
  command: ('C:\Python314\python.exe', '-mvirtualenv', '...py_env-python3.10', '-p', 'python3.10')
  return code: 1
  RuntimeError: failed to find interpreter for Builtin discover of python_spec='python3.10'
PRE-COMMIT EXIT: 3
```

На машине установлен только Python 3.14.3; `python3.10` отсутствует.

**Аргумент.** `requires-python = ">=3.10"` в `pyproject.toml` — это **минимум**, а не требование запускать инструменты под конкретной минорной версией. Совместимость с 3.10 у black/isort обеспечивается отдельно и уже настроено: `target-version = "py310"` в `[tool.black]` и `target-version = "py310"` в `[tool.ruff]`. Пин `language_version: python3.10` в хуке не даёт ничего (те же таргеты уже заданы), но блокирует всю цепочку на любой машине без 3.10 — включая машину владельца.

**Рекомендация:** убрать строку `language_version: python3.10` у black-хука (или заменить на `language_version: python3`). Проверено: black/isort/ruff корректно работают на 3.14 — `black --check src/ tests/` даёт exit 0.

---

## 4. Замечание 3 — bandit-хук не собирает окружение

**Доказательство:**

```
$ pre-commit run bandit --all-files
  .../bandit/__init__.py, line 5, in <module>
    import pbr.version
ModuleNotFoundError: No module named 'pbr'
BANDIT EXIT: 1
```

**Аргумент.** `mirrors-bandit` требует `pbr`, которого нет в образе хука. Это дефект конфигурации хука, а не кода — но в текущем виде хук не выполняет свою функцию (а bandit для проекта с API-эндпоинтом и обработкой файлов полезен).

**Рекомендация:**

```yaml
- id: bandit
  args: ['-c', '.bandit']
  additional_dependencies: ['pbr']
```

---

## 5. Замечание 4 — mypy-хук останется красным

**Доказательство.** `mypy src/hipporag --ignore-missing-imports` → **208 замечаний в 34 файлах** (exit 1). В CI это не блокирует — там стоит `continue-on-error: true`. В pre-commit такого механизма нет, поэтому хук упадёт на каждом коммите. `exclude: ^(tests/|reproduce/)` оставляет в scope ещё и корневые скрипты (`main.py` и др.), где аннотаций нет вовсе.

**Аргумент.** Хук, который падает всегда, не выполняет функцию ворот — он только создаёт раздражение и подталкивает к `--no-verify`. Раз mypy уже есть в CI как advisory, дублировать его в pre-commit в блокирующем виде нецелесообразно.

**Рекомендация:** убрать mypy из pre-commit (оставить advisory-шаг в CI), либо сузить scope до `^src/hipporag/` и принять, что до приведения типизации хук красный.

---

## 6. Замечание 5 — побочный эффект моей проверки: изменены 2 файла (раскрываю честно)

Запуск `pre-commit run --all-files` — это не только проверка, но и **авто-фикс**: хуки `trailing-whitespace`, `end-of-file-fixer` и `mixed-line-ending` переписали файлы в рабочем дереве.

| Файл | Что сделано | Риск |
|------|-------------|------|
| `src/hipporag/prompts/templates/triple_extraction.py` | black (форматирование), isort (порядок импортов), trailing-whitespace, добавлена финальная новая строка | Проверен — см. ниже |
| `reproduce/dataset/2wikimultihopqa_corpus.json` | `end-of-file-fixer` + `mixed-line-ending` | Только завершающая новая строка и нормализация переводов строк |

**Проверка влияния на промпт** (сравнение отрендеренного `prompt_template` из HEAD и из рабочего дерева):

```
message[0] role=system     identical=False  len_head=456 len_cur=454
   first diff at char 119: head=' ' cur='\n'      ← сняты 2 trailing space
message[1] role=user       identical=True
message[2] role=assistant  identical=True
message[3] role=user       identical=True
```

Разница — **два концевых пробела** в тексте system-промпта; семантически промпт идентичен. После авто-фиксов: `pytest tests/unit` → **62 passed**, `black/isort/ruff` по `src`+`tests` → зелёные.

**Последствие для владельца:** в рабочем дереве теперь +2 изменённых файла. Если изменения нежелательны — `git checkout -- src/hipporag/prompts/templates/triple_extraction.py reproduce/dataset/2wikimultihopqa_corpus.json`. Если оставить — файл промпта станет black/isort-совместимым, что как раз нужно для варианта B из замечания 1.

---

## 7. Замечание 6 (на будущее) — датасеты под авто-фиксом

`reproduce/dataset/2wikimultihopqa_corpus.json` — большой машиночитаемый датасет. Хуки `end-of-file-fixer` и `mixed-line-ending` правят его при каждом прогоне, что даёт лишние диффы в данных и риск случайной порчи разметки. Стоит исключить `reproduce/` (и `archive/`) из файловых авто-фиксов:

```yaml
- id: trailing-whitespace
  exclude: ^(reproduce|archive|outputs)/
```

---

## 8. Сводка: что нужно сделать

| # | Что | Приоритет | Оценка |
|---|-----|-----------|--------|
| 1 | Сузить `files: ^(src|tests)/` у хуков `black` и `isort` (или расширить ворота на корневые скрипты и починить их) | **блокирующее** | 20 мин |
| 2 | Убрать `language_version: python3.10` у black-хука | **блокирующее** | 2 мин |
| 3 | Добавить `additional_dependencies: ['pbr']` бандит-хуку | среднее | 2 мин |
| 4 | Убрать mypy из pre-commit (оставить advisory в CI) | среднее | 5 мин |
| 5 | Исключить `reproduce/` и `archive/` из файловых авто-фиксов | низкое | 5 мин |
| 6 | Решить по 2 изменённым файлам из п. 6 (оставить или откатить) | низкое | 1 мин |

**Приёмка:** `pre-commit run --all-files` → exit 0 на машине без Python 3.10; `pytest tests/unit` → 62 passed; `black --check src/ tests/`, `isort --check-only src/ tests/`, `ruff check src tests` → exit 0.

---

## 9. Что уже закрыто и подтверждено (перечень для протокола)

- **R-1′ (ruff)** — ✅ сужение `files: ^(src|tests)/` работает, хук проходит.
- **R-2** — ✅ критерий 0.8: «одна запись со списком источников», `Citation.sources` называет оба документа (стр. 164).
- **R-3** — ✅ приёмка S-30: «чанков > 0 **и фактов > 100**» (стр. 200).
- **YAML** — ✅ оба файла валидны, `ci-cd.yml`: jobs `lint`/`security`/`test`/`build`/`publish`; исполнитель дополнительно нашёл и починил собственный дефект (двоеточие в имени шага ruff), не попавший в мой вердикт.
- **Инструменты в области CI** — ✅ black (107 unchanged), isort, ruff, pytest 62/62, compileall exit 0.
- **D-1…D-8** — ✅ закрыты и проверены эмпирически (см. `docs/AUDIT_EXECUTION_VERDICT_R2_2026-11-01.md`).
- **Не входит в этот раунд:** переиндексация (операционный шаг) — 5 критериев приёмки зафиксированы в `docs/AUDIT_FINAL_CONCLUSION_2026-11-01.md`, раздел 5.
