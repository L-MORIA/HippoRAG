"""Бюджеты латентности и профилирование (H1-3).

Политика зафиксирована в плане (`docs/ROADMAP_REVISION_DEEPSEEK.md`, H1-3):

* режим по умолчанию — **p95 ≤ 10 с**;
* LLM-режимы (те, что формируют ответ) — **p95 ≤ 30 с**;
* всё, что медленнее 30 с, — **«экспериментальный» режим**: по умолчанию
  выключен, в меню помечается, в отчётах идёт отдельной строкой.

Модуль чистый: не ходит в сеть, не читает индексы и не запускает модели. Он
считает вердикты по уже измеренным сводкам. Замер — отдельная задача
(`scripts/benchmark_budget.py`), результат хранится в
`eval/dataset/latency_baseline.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .evaluation import ANSWER_ONLY_MODES, LLM_DEPENDENT_MODES
from .modes import DEFAULT_MODE

#: Бюджет режима по умолчанию (p95, мс).
DEFAULT_P95_BUDGET_MS = 10_000

#: Бюджет LLM-режимов (p95, мс).
LLM_P95_BUDGET_MS = 30_000

#: Всё, что медленнее этого порога (p95, мс), — экспериментальный режим.
EXPERIMENTAL_P95_MS = 30_000

#: Роли режимов и их бюджеты.
ROLE_DEFAULT = "default"
ROLE_LLM = "llm"
ROLE_EXPERT = "expert"
ROLE_BUDGETS: Dict[str, int] = {
    ROLE_DEFAULT: DEFAULT_P95_BUDGET_MS,
    ROLE_LLM: LLM_P95_BUDGET_MS,
    ROLE_EXPERT: LLM_P95_BUDGET_MS,
}

#: Вердикты.
STATUS_OK = "ok"
STATUS_OVER_BUDGET = "over-budget"
STATUS_EXPERIMENTAL = "experimental"
STATUS_UNKNOWN = "unknown"

#: Путь к машинно-читаемому снимку замеров (в репозитории; сырые отчёты — в
#: `outputs/`, который вне git).
BASELINE_PATH = Path(__file__).resolve().parents[2] / "eval" / "dataset" / "latency_baseline.json"


def mode_role(mode_id: str, default_mode: Optional[str] = None) -> str:
    """Роль режима: default / llm / expert.

    * режим по умолчанию — `default` (бюджет 10 с);
    * режимы, которые ходят в LLM или формируют ответ, — `llm` (бюджет 30 с);
    * остальные — `expert` (бюджет 30 с).
    """
    default = default_mode or DEFAULT_MODE
    if mode_id == default:
        return ROLE_DEFAULT
    if mode_id in LLM_DEPENDENT_MODES or mode_id in ANSWER_ONLY_MODES:
        return ROLE_LLM
    return ROLE_EXPERT


def budget_for_role(role: str) -> int:
    """Бюджет p95 для роли, мс."""
    try:
        return ROLE_BUDGETS[role]
    except KeyError:
        raise ValueError(f"Неизвестная роль режима: {role!r}") from None


def classify_latency(p95_ms: Optional[float], role: str) -> str:
    """Вердикт по одному числу p95 и роли режима."""
    if p95_ms is None:
        return STATUS_UNKNOWN
    budget = budget_for_role(role)
    if p95_ms > EXPERIMENTAL_P95_MS:
        return STATUS_EXPERIMENTAL
    if p95_ms > budget:
        return STATUS_OVER_BUDGET
    return STATUS_OK


def evaluate_mode(summary: Dict[str, Any], *, default_mode: Optional[str] = None) -> Dict[str, Any]:
    """Вердикт по сводке одного режима (как её отдаёт `evaluation.summarize_mode`)."""
    mode = summary.get("mode") or {}
    mode_id = str(mode.get("id") or "")
    label = mode.get("label") or ""
    role = mode_role(mode_id, default_mode)
    budget = budget_for_role(role)
    latency = summary.get("latency_ms") or {}
    retrieval = summary.get("retrieval_ms") or {}
    p50 = latency.get("p50")
    p95 = latency.get("p95")
    status = classify_latency(p95, role)
    margin = None if p95 is None else round(budget - float(p95), 1)
    return {
        "mode": mode_id,
        "label": label,
        "role": role,
        "budget_ms": budget,
        "p50_ms": p50,
        "p95_ms": p95,
        "retrieval_p95_ms": retrieval.get("p95"),
        "llm_calls_mean": summary.get("llm_calls_mean"),
        "status": status,
        "margin_ms": margin,
    }


def evaluate_budgets(
    summaries: Iterable[Dict[str, Any]], *, default_mode: Optional[str] = None
) -> Dict[str, Any]:
    """Вердикты по всем режимам прогона + общий итог."""
    verdicts = [evaluate_mode(s, default_mode=default_mode) for s in summaries]
    violations = [v for v in verdicts if v["status"] == STATUS_OVER_BUDGET]
    experimental = [v for v in verdicts if v["status"] == STATUS_EXPERIMENTAL]
    unknown = [v for v in verdicts if v["status"] == STATUS_UNKNOWN]
    return {
        "modes": verdicts,
        "violations": [v["mode"] for v in violations],
        "experimental": [v["mode"] for v in experimental],
        "unknown": [v["mode"] for v in unknown],
        "ok": not violations,
    }


def format_budget_table(verdicts: Sequence[Dict[str, Any]]) -> str:
    """Markdown-таблица вердиктов по бюджету латентности."""
    lines = [
        "| Режим | Роль | p50, мс | p95, мс | Бюджет, мс | Запас, мс | Вердикт |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in verdicts:
        lines.append(
            "| {mode} {label} | {role} | {p50} | {p95} | {budget} | {margin} | {status} |".format(
                mode=v["mode"],
                label=v["label"],
                role=v["role"],
                p50=_fmt(v.get("p50_ms")),
                p95=_fmt(v.get("p95_ms")),
                budget=v["budget_ms"],
                margin="—" if v.get("margin_ms") is None else _fmt(v.get("margin_ms")),
                status=v["status"],
            )
        )
    return "\n".join(lines)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{float(value):.0f}"


# --- машинно-читаемый снимок замеров ----------------------------------------


def load_baseline(path: Optional[Path] = None) -> Dict[str, Any]:
    """Загрузить снимок замеров (`eval/dataset/latency_baseline.json`).

    Отсутствие файла — не ошибка: снимок появляется после прогона
    `scripts/benchmark_budget.py`, а до этого меню просто не помечает
    экспериментальные режимы.
    """
    target = Path(path) if path else BASELINE_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def baseline_verdicts(
    baseline: Dict[str, Any], *, default_mode: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Вердикты по снимку замеров (без запуска моделей)."""
    modes = baseline.get("modes") or {}
    summaries = []
    for mode_id, entry in modes.items():
        if not isinstance(entry, dict):
            continue
        summaries.append(
            {
                "mode": {"id": str(mode_id), "label": entry.get("label", "")},
                "latency_ms": {"p50": entry.get("p50_ms"), "p95": entry.get("p95_ms")},
                "retrieval_ms": {
                    "p50": entry.get("retrieval_p50_ms"),
                    "p95": entry.get("retrieval_p95_ms"),
                },
                "llm_calls_mean": entry.get("llm_calls_mean"),
            }
        )
    return evaluate_budgets(summaries, default_mode=default_mode)["modes"]


def experimental_modes(baseline: Dict[str, Any], *, default_mode: Optional[str] = None) -> Set[str]:
    """Режимы, которые по замерам медленнее 30 с (помечаются в меню)."""
    return {
        v["mode"]
        for v in baseline_verdicts(baseline, default_mode=default_mode)
        if v["status"] == STATUS_EXPERIMENTAL
    }
