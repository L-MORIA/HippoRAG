#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2: сводная таблица citation coverage для legacy-режимов 12/19/21.

Сниппет из docs/TASK_M2_LEGACY_MODES_2026-11-05.md (Шаг 5), сохранён как
скрипт для запуска на Windows. Читает outputs/reports/m2_mode{12,19,21}.json
(отчёты eval_grounding.py) и пишет outputs/reports/m2_legacy_summary.md.
"""

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
