#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Смоук по урезанному индексу-фикстуре (H1-5).

Прогоняет **лексический** retrieval (BM25) по фикстуре на размеченных вопросах
датасета и пишет отчёт. Не требует ни эмбеддинг-модели, ни LLM, ни сети —
поэтому запускается в CI.

    python scripts/index_fixture_smoke.py \
        --fixture fixtures/astrology_mini \
        --out outputs/reports/fixture_smoke.json \
        --markdown-out outputs/reports/fixture_smoke.md

Код возврата: ``0`` — фикстура цела (и recall не ниже ``--min-recall``),
``1`` — нарушен контракт фикстуры или порог recall, ``2`` — фикстуры нет.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from astro_app import evaluation as ev  # noqa: E402
from astro_app import index_fixture as fix  # noqa: E402

DEFAULT_DATASET = "eval/dataset/questions.jsonl"


def load_questions(path: str, limit: int) -> List[Dict[str, Any]]:
    """Размеченные вопросы датасета в виде словарей."""
    dataset = ev.load_dataset(path)
    questions = dataset.for_run(include_draft=False)
    rows = [
        {
            "qid": q.qid,
            "question": q.question,
            "source_file": q.source_file,
            "page_num": q.page_num,
            "kind": q.kind,
        }
        for q in questions
    ]
    return rows[:limit] if limit else rows


def format_report(report: Dict[str, Any]) -> str:
    """Markdown-отчёт смоука."""
    lines = [
        "# Смоук по индексу-фикстуре (H1-5)",
        "",
        f"- фикстура: `{report['fixture']}`",
        f"- чанков в фикстуре: {report['n_chunks']}",
        f"- вопросов: {report['n_questions']}, попаданий: {report['n_hits']}",
        f"- recall@{report['top_k']} (BM25, без эмбеддингов): **{report['recall']}**",
        f"- контракт фикстуры: {'✅ цела' if report['fixture_ok'] else '❌ нарушен'}",
        "",
        "| qid | ожидаемый файл | позиция | попал |",
        "|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['qid']} | {row['source_file']} | {row['rank'] or '—'} | "
            f"{'✅' if row['hit'] else '—'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Смоук по индексу-фикстуре (H1-5)")
    parser.add_argument(
        "--fixture", default="fixtures/astrology_mini", help="каталог или файл фикстуры"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=10, help="сколько вопросов взять (0 — все)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--min-recall", type=float, default=0.0, help="порог recall (0 — не проверять)"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args(argv)

    fixture_dir = Path(args.fixture)
    if not fix.fixture_path(fixture_dir).exists():
        print(
            f"Фикстура не найдена: {fix.fixture_path(fixture_dir)} — "
            "соберите её (scripts/build_index_fixture.py) и закоммитьте в репозиторий.",
            file=sys.stderr,
        )
        return 2

    problems = fix.verify_fixture(fixture_dir)
    fixture = fix.load_fixture(fixture_dir)
    questions = load_questions(args.dataset, args.limit)
    smoke = fix.smoke_retrieval(fixture, questions, top_k=args.top_k)

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": args.dataset,
        },
        "fixture": str(fixture_dir),
        "n_chunks": len(fixture.get("chunks") or []),
        "fixture_ok": not problems,
        "fixture_problems": problems,
        "n_questions": smoke["n_questions"],
        "n_hits": smoke["n_hits"],
        "recall": smoke["recall"],
        "top_k": smoke["top_k"],
        "rows": smoke["rows"],
    }

    print(f"Фикстура: {fixture_dir} — чанков {report['n_chunks']}")
    if problems:
        print("❌ Контракт фикстуры нарушен:")
        for problem in problems:
            print(f"   - {problem}")
    else:
        print("✅ Контракт фикстуры соблюдён (sha256 совпадает)")
    print(
        f"BM25-смоук: {report['n_hits']}/{report['n_questions']} попаданий, "
        f"recall@{report['top_k']} = {report['recall']}"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"Отчёт: {out_path}")

    if args.markdown_out:
        md_path = Path(args.markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_report(report), encoding="utf-8")
        print(f"Таблица: {md_path}")

    if problems:
        return 1
    if args.min_recall and report["recall"] < args.min_recall:
        print(f"❌ recall {report['recall']} < порога {args.min_recall}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
