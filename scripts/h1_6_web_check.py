#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Живая проверка H1-6 в UI: режим 5 через /api/ask, цитаты-страницы, латентность.

Служебный скрипт приёмки; в package data не входит.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

API_KEY = os.getenv("API_KEY_CHECK", "h1-6-ui-check-8842")
BASE = os.getenv("CHECK_BASE", "http://127.0.0.1:8000")

QUESTIONS = [
    "Кто автор книги «Психологический фэншуй»?",
    "Какое направление Шен-ци у числа гуа 1?",
    "Какие зоны квартиры соответствуют Овну и какая планета ими управляет?",
    "За что отвечает 4-й дом (IC) в системе домов?",
    "Какой подзаголовок у книги «Психологический фэншуй»?",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    for i, question in enumerate(QUESTIONS, 1):
        body = {"question": question, "mode": "5", "top_k": 5}
        started = time.perf_counter()
        response = httpx.post(
            f"{BASE}/api/ask",
            headers={"X-API-Key": API_KEY, "Content-Type": "application/json; charset=utf-8"},
            content=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            timeout=180,
        )
        wall = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        citations = data.get("citations") or []
        fragments = data.get("fragments") or []
        pages = [c.get("page_num") for c in citations]
        retrieval_ms = float(data.get("retrieval_time") or 0) * 1000
        rows.append(
            {
                "question": question,
                "abstained": data.get("abstained"),
                "n_fragments": len(fragments),
                "citation_pages": pages,
                "retrieval_ms": round(retrieval_ms, 1),
                "answer_ms": round(float(data.get("answer_time") or 0) * 1000, 1),
                "wall_ms": round(wall * 1000, 1),
                "answer_head": (data.get("answer") or "")[:110].replace("\n", " "),
            }
        )
        print(
            f"[{i}] abstained={rows[-1]['abstained']} frag={len(fragments)} "
            f"pages={pages} retrieval={retrieval_ms:.0f}мс",
            flush=True,
        )
        print(f"    answer: {rows[-1]['answer_head']}")

    retrieval = [r["retrieval_ms"] for r in rows]
    retrieval_sorted = sorted(retrieval)
    p50 = statistics.median(retrieval_sorted)
    p95 = retrieval_sorted[min(len(retrieval_sorted) - 1, round(0.95 * len(retrieval_sorted)) - 1)] if retrieval_sorted else 0.0
    summary = {
        "n": len(rows),
        "retrieval_p50_ms": round(p50, 1),
        "retrieval_p95_ms": round(p95, 1),
        "retrieval_mean_ms": round(statistics.fmean(retrieval), 1),
        "rows": rows,
    }
    Path("outputs/reports/h1_6_web_check.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nretrieval: p50={p50:.0f} мс, p95={p95:.0f} мс, mean={statistics.fmean(retrieval):.0f} мс"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
