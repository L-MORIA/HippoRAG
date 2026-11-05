#!/usr/bin/env python3
"""Offline citation-policy evaluation for grounding predictions.

Reads a predictions JSONL (for example, M7 baseline), applies deterministic
citation-selection policies, and reports metrics plus a per-question audit.

Policy functions never see gold sources: they use only the answer text,
citation IDs, and the retrieved fragment list. Gold is used only by the
evaluation/audit layer.

Референсный скрипт из docs/TASK_H2_6_CITATION_PRECISION_2026-11-05.md
(Приложение A); не входит в package data.
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
