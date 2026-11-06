# -*- coding: utf-8 -*-
"""Проверка page recall продуктового page-level пути (H1-6) без LLM.

Зачем отдельно от ``eval_grounding.py --run``: тот гоняет полный путь с LLM
(дорого и зависит от LM Studio), а для приёмки H1-6 нужна только доставка
доказательства. Скрипт вызывает тот же адаптер, что и eval
(``scripts/grounding_astro_retriever.py``), но без генерации ответа.

Запуск::

    python scripts/check_page_recall.py                 # golden set, k=5/10/20
    python scripts/check_page_recall.py --min-recall 0.8  # код возврата 1, если ниже

Индекс и модель берутся из окружения: ``ASTRO_SAVE_DIR``, ``ASTRO_LLM_NAME``,
``ASTRO_LLM_API_NAME`` (см. ``grounding_astro_retriever.py``). LLM не нужен.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import eval_grounding as eg  # noqa: E402
import grounding_astro_retriever as gar  # noqa: E402

DEFAULT_KS = (5, 10, 20)


def parse_ks(raw: str) -> List[int]:
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("--ks: нужны целые >= 1 через запятую")
    return values


def fragment_key(fragment: Dict[str, object]) -> tuple:
    return eg.source_key(
        eg.SourceRef(doc_title=str(fragment["doc_title"]), page_num=int(fragment["page_num"]))
    )


def measure(golden_path: Path, ks: Sequence[int], warmup: bool = True) -> Dict[str, object]:
    """Считает page recall@k и MRR@10 по answerable-вопросам без LLM."""
    golden = eg.load_golden(golden_path)
    questions = list(golden.answerable)
    if not questions:
        raise SystemExit("в golden set нет answerable-вопросов")
    top_k = max(ks)
    if warmup:
        gar.retrieve("прогрев", top_k=top_k)

    hits = {k: 0 for k in ks}
    mrr = 0.0
    ranks: List[Optional[int]] = []
    for question in questions:
        gold = {eg.source_key(source) for source in question.gold_sources}
        fragments = gar.retrieve(question.question, top_k=top_k)
        rank = None
        for position, fragment in enumerate(fragments, start=1):
            if fragment_key(fragment) in gold:
                rank = position
                break
        ranks.append(rank)
        for k in ks:
            if rank and rank <= k:
                hits[k] += 1
        if rank and rank <= 10:
            mrr += 1.0 / rank

    total = len(questions)
    return {
        "questions": total,
        "top_k": top_k,
        "recall": {k: hits[k] / total for k in ks},
        "mrr@10": mrr / total,
        "ranks": {question.id: rank for question, rank in zip(questions, ranks)},
        "misses": [question.id for question, rank in zip(questions, ranks) if not rank],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Page recall пути H1-6 без LLM")
    parser.add_argument("--golden", default=str(eg.DEFAULT_GOLDEN), help="разметка golden set")
    parser.add_argument("--ks", default="5,10,20", help="какие k считать (по умолчанию 5,10,20)")
    parser.add_argument("--min-recall", type=float, default=None, help="порог для k из --ks")
    parser.add_argument("--no-warmup", action="store_true", help="без прогрева page-индекса")
    parser.add_argument("--out", help="куда сохранить JSON (необязательно)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ks = parse_ks(args.ks)
    result = measure(Path(args.golden), ks, warmup=not args.no_warmup)

    print(f"вопросов: {result['questions']}, top_k: {result['top_k']}")
    for k in ks:
        hits = round(result["recall"][k] * result["questions"])
        print(f"page recall@{k}: {result['recall'][k] * 100:.1f}% ({hits}/{result['questions']})")
    print(f"MRR@10: {result['mrr@10']:.3f}")
    if result["misses"]:
        print("промахи: " + ", ".join(result["misses"]))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "golden": str(args.golden),
            "questions": result["questions"],
            "top_k": result["top_k"],
            "recall": {str(k): round(value, 4) for k, value in result["recall"].items()},
            "mrr@10": round(result["mrr@10"], 4),
            "ranks": result["ranks"],
            "misses": result["misses"],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"отчёт: {out_path}")

    if args.min_recall is not None:
        threshold = args.min_recall
        if not 0.0 <= threshold <= 1.0:
            raise SystemExit("--min-recall должен быть в диапазоне 0..1")
        base_k = min(ks)
        if result["recall"][base_k] < threshold:
            print(
                f"❌ page recall@{base_k} {result['recall'][base_k] * 100:.1f}% "
                f"< порога {threshold * 100:.1f}%"
            )
            return 1
        print(f"✅ page recall@{base_k} не ниже порога {threshold * 100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
