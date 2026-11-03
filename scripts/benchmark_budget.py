#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Бюджет латентности режимов (H1-3) — замер и вердикты.

Что делает: прогоняет ``scripts/eval_search_quality.py`` (тот же измеритель, что
использован для H1-1) и переводит его сводки в вердикты по бюджету:

* режим по умолчанию — p95 ≤ 10 с;
* LLM-режимы — p95 ≤ 30 с;
* всё, что медленнее 30 с, — «экспериментальный» режим.

Замер умеет и не запускать модели: ``--from-report`` переоценивает уже
сохранённый отчёт ``eval_search_quality.py`` (например, из ``outputs/``).

Примеры::

    # замер на своём индексе (нужны локальный индекс и LM Studio)
    python scripts/benchmark_budget.py --index outputs/astrology_test --modes 1,2,5,21 --warmup

    # переоценка готового отчёта без моделей
    python scripts/benchmark_budget.py --from-report outputs/reports/eval_after_llm.json

    # обновить снимок замеров в репозитории (eval/dataset/latency_baseline.json)
    python scripts/benchmark_budget.py --from-report outputs/reports/eval_after_llm.json --update-baseline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from astro_app import performance as perf  # noqa: E402

EVAL_SCRIPT = ROOT / "scripts" / "eval_search_quality.py"
BASELINE_FILE = ROOT / "eval" / "dataset" / "latency_baseline.json"

DEFAULT_MODES = "1,2,5,21"


def _parse_modes(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def run_eval(args: argparse.Namespace, out_path: Path) -> int:
    """Запустить измеритель H1-1 и вернуть код возврата."""
    cmd: List[str] = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--index",
        args.index,
        "--modes",
        args.modes,
        "--out",
        str(out_path),
    ]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.embedding_model:
        cmd += ["--embedding-model", args.embedding_model]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.top_k:
        cmd += ["--top-k", str(args.top_k)]
    if args.warmup:
        cmd += ["--warmup"]
    if args.stub_llm:
        cmd += ["--stub-llm"]
    if args.allow_unavailable_llm:
        cmd += ["--allow-unavailable-llm"]
    if args.include_draft:
        cmd += ["--include-draft"]

    print("Замер:", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


def load_report(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Отчёт {path} не является JSON-объектом")
    return data


def build_baseline(report: Dict[str, Any]) -> Dict[str, Any]:
    """Снимок замеров для репозитория (машинно-читаемый)."""
    meta = report.get("meta") or {}
    modes: Dict[str, Any] = {}
    for summary in report.get("summaries") or []:
        mode = summary.get("mode") or {}
        mode_id = str(mode.get("id") or "")
        if not mode_id:
            continue
        latency = summary.get("latency_ms") or {}
        retrieval = summary.get("retrieval_ms") or {}
        modes[mode_id] = {
            "label": mode.get("label", ""),
            "p50_ms": latency.get("p50"),
            "p95_ms": latency.get("p95"),
            "retrieval_p50_ms": retrieval.get("p50"),
            "retrieval_p95_ms": retrieval.get("p95"),
            "llm_calls_mean": summary.get("llm_calls_mean"),
            "recall@k_doc": summary.get("recall@k_doc"),
            "mrr@k_doc": summary.get("mrr@k_doc"),
            "ndcg@k_doc": summary.get("ndcg@k_doc"),
            "adequacy_mean": summary.get("adequacy_mean"),
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": (
            "живой прогон scripts/eval_search_quality.py; сырой отчёт лежит вне репозитория "
            "(outputs/ в .gitignore), здесь — машинно-читаемый снимок агрегатов"
        ),
        "index": (meta.get("index") or {}).get("path"),
        "dataset": meta.get("dataset"),
        "llm": (meta.get("llm") or {}).get("name"),
        "n_questions": (meta.get("dataset") or {}).get("n_questions"),
        "budgets_ms": {
            "default": perf.DEFAULT_P95_BUDGET_MS,
            "llm": perf.LLM_P95_BUDGET_MS,
            "experimental_threshold": perf.EXPERIMENTAL_P95_MS,
        },
        "modes": modes,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Бюджет латентности режимов (H1-3)")
    parser.add_argument("--index", default="outputs/astrology_test", help="save_dir индекса")
    parser.add_argument("--dataset", default="eval/dataset/questions.jsonl")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--modes", default=DEFAULT_MODES, help="список режимов через запятую")
    parser.add_argument("--limit", type=int, default=0, help="первые N вопросов (0 — все)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--warmup", action="store_true", help="прогрев ленивой инициализации")
    parser.add_argument("--stub-llm", action="store_true")
    parser.add_argument("--allow-unavailable-llm", action="store_true")
    parser.add_argument("--include-draft", action="store_true")
    parser.add_argument(
        "--from-report",
        default=None,
        help="переоценить готовый отчёт eval_search_quality.py вместо нового прогона",
    )
    parser.add_argument("--out", default=None, help="куда сохранить JSON с вердиктами")
    parser.add_argument("--markdown-out", default=None, help="куда сохранить Markdown-таблицу")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=f"обновить снимок замеров {BASELINE_FILE.relative_to(ROOT)}",
    )
    args = parser.parse_args(argv)

    if args.from_report:
        report_path = Path(args.from_report)
        report = load_report(report_path)
    else:
        with tempfile.TemporaryDirectory(prefix="hipporag-budget-") as tmp:
            report_path = Path(tmp) / "eval.json"
            code = run_eval(args, report_path)
            if code != 0 or not report_path.exists():
                print(f"Замер не удался (код {code}) — вердикты не посчитаны", file=sys.stderr)
                return code
            report = load_report(report_path)

    summaries = report.get("summaries") or []
    if not summaries:
        print("В отчёте нет сводок по режимам — нечего оценивать", file=sys.stderr)
        return 2

    verdicts = perf.evaluate_budgets(summaries)
    table = perf.format_budget_table(verdicts["modes"])

    print()
    print(table)
    print()
    if verdicts["violations"]:
        print(f"❌ Нарушен бюджет: режимы {', '.join(verdicts['violations'])}")
    else:
        print("✅ Бюджеты соблюдены для всех измеренных режимов")
    if verdicts["experimental"]:
        print(
            f"⏱ Экспериментальные (> {perf.EXPERIMENTAL_P95_MS} мс): {', '.join(verdicts['experimental'])}"
        )
    if verdicts["unknown"]:
        print(f"⚠ Без измерения p95: {', '.join(verdicts['unknown'])}")

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_report": str(args.from_report) if args.from_report else "новый прогон",
            "modes": _parse_modes(args.modes),
        },
        "budgets_ms": {
            "default": perf.DEFAULT_P95_BUDGET_MS,
            "llm": perf.LLM_P95_BUDGET_MS,
            "experimental_threshold": perf.EXPERIMENTAL_P95_MS,
        },
        "verdicts": verdicts,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Вердикты: {out_path}")

    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(table + "\n", encoding="utf-8")
        print(f"Таблица: {md_path}")

    if args.update_baseline:
        baseline = build_baseline(report)
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_FILE.write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Снимок замеров обновлён: {BASELINE_FILE.relative_to(ROOT)}")

    return 0 if verdicts["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
