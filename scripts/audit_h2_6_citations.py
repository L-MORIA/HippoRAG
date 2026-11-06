#!/usr/bin/env python3
"""Read-only audit of H2-6 citation precision.

The official metric is page-level: a citation is correct only when its
``(document, page)`` pair is in ``gold_sources``.  This script does not change
the golden set, predictions, metric definitions, or thresholds.  It only
reports:

* the official precision and several sensitivity variants;
* per-question breakdown of extra citations;
* leave-one-out cross-run evidence: the same source cited for the same
  question in another prediction file, with a correct answer;
* an oracle estimate of how many extra citations must be removed to reach
  the 0.80 threshold while keeping all gold citations.

The script is intentionally outside the package and has no side effects other
than writing the requested JSON/Markdown reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(os.getenv("HIPPORAG_REPO_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import eval_grounding as eg  # noqa: E402

SourceKey = Tuple[str, int]
SourceMap = Dict[SourceKey, List[str]]

CITATION_RE = re.compile(r"\[\s*([0-9,\s]+?)\s*\]")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
DEFAULT_GOLDEN = ROOT / "eval" / "grounding" / "golden_qa.jsonl"
DEFAULT_PREDICTIONS = ROOT / "eval" / "grounding" / "predictions_m7_baseline.jsonl"
DEFAULT_RUN_GLOB = "predictions_*.jsonl"
DEFAULT_JSON = ROOT / "outputs" / "reports" / "h2_6_metric_audit.json"
DEFAULT_MD = ROOT / "outputs" / "reports" / "h2_6_metric_audit.md"


def ids_in(text: str) -> List[int]:
    """Return unique citation IDs in their first-appearance order."""
    result: List[int] = []
    for group in CITATION_RE.findall(text or ""):
        result.extend(int(part) for part in re.findall(r"\d+", group))
    return list(dict.fromkeys(result))


def sentences_with_ids(answer: str) -> List[Tuple[str, List[int]]]:
    """Split an answer into sentences and keep the citation IDs of each."""
    result: List[Tuple[str, List[int]]] = []
    for sentence in SENTENCE_RE.split(answer or ""):
        ids = ids_in(sentence)
        if ids:
            result.append((sentence.strip(), ids))
    return result


def mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def display_path(path: Path) -> str:
    """Relative path for reports when the file is inside the repository."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def page_distance(source: SourceKey, gold: Set[SourceKey]) -> Optional[int]:
    """Distance in pages to the nearest gold page of the same document."""
    doc, page = source
    pages = [gold_page for gold_doc, gold_page in gold if gold_doc == doc]
    return min(abs(page - gold_page) for gold_page in pages) if pages else None


def load_runs(
    predictions_path: Path,
    run_glob: str,
) -> Dict[str, Dict[str, eg.Prediction]]:
    """Load the audited predictions and every other run for cross-evidence."""
    runs: Dict[str, Dict[str, eg.Prediction]] = {
        predictions_path.stem: eg.load_predictions(predictions_path)
    }
    for path in sorted(predictions_path.parent.glob(run_glob)):
        if path.resolve() == predictions_path.resolve():
            continue
        runs[path.stem] = eg.load_predictions(path)
    return runs


def build_cross_run_support(
    golden: eg.GoldenSet,
    runs: Dict[str, Dict[str, eg.Prediction]],
) -> Tuple[SourceMap, SourceMap]:
    """Map ``(question, source)`` to runs that cite it.

    ``cited_any`` ignores answer correctness; ``cited_correct`` keeps only
    runs whose answer contains every gold term.  The audited run is excluded
    by the caller (it is not in ``runs`` when this is called).
    """
    cited_any: SourceMap = defaultdict(list)
    cited_correct: SourceMap = defaultdict(list)
    for name, predictions in runs.items():
        for question in golden.items:
            if not question.answerable:
                continue
            prediction = predictions.get(question.id)
            if prediction is None or not eg.is_answered(prediction):
                continue
            correct = eg.answer_is_correct(prediction, question)
            for source in {eg.source_key(citation) for citation in prediction.citations}:
                key = (question.id, source)
                cited_any[key].append(name)
                if correct:
                    cited_correct[key].append(name)
    return cited_any, cited_correct


def build_gold_elsewhere(golden: eg.GoldenSet) -> SourceMap:
    """Map a source to other answerable questions where it is gold."""
    gold_elsewhere: SourceMap = defaultdict(list)
    for question in golden.items:
        if not question.answerable:
            continue
        for source in question.gold_sources:
            gold_elsewhere[eg.source_key(source)].append(question.id)
    return gold_elsewhere


def sentence_roles(
    prediction: eg.Prediction,
) -> Dict[SourceKey, List[Tuple[str, List[SourceKey]]]]:
    """Map each cited source to the sentences where it appears.

    The answer markers are resolved through ``prediction.retrieved``; this is
    the same mapping the model used when it produced the citation.
    """
    roles: Dict[SourceKey, List[Tuple[str, List[SourceKey]]]] = defaultdict(list)
    for sentence, ids in sentences_with_ids(prediction.answer):
        sources = [
            eg.source_key(prediction.retrieved[identifier - 1])
            for identifier in ids
            if 1 <= identifier <= len(prediction.retrieved)
        ]
        for source in sources:
            roles[source].append((sentence, sources))
    return roles


def evaluate_question(
    question: eg.GoldenQuestion,
    prediction: eg.Prediction,
    cited_any: SourceMap,
    cited_correct: SourceMap,
    gold_elsewhere: SourceMap,
) -> Optional[Dict[str, Any]]:
    """Return per-question audit data for an answered answerable question."""
    if not question.answerable or not eg.is_answered(prediction):
        return None

    gold = {eg.source_key(source) for source in question.gold_sources}
    gold_docs = {doc for doc, _ in gold}
    cited = {eg.source_key(citation) for citation in prediction.citations}
    if not cited:
        return {
            "id": question.id,
            "question": question.question,
            "gold": sorted(gold),
            "cited": [],
            "extras": [],
            "gold_cited": 0,
            "cited_count": 0,
            "strict_precision": 0.0,
            "doc_precision": 0.0,
            "evidence_precision": 0.0,
            "distance_le_2": 0,
            "cross_run_any": 0,
            "cross_run_correct": 0,
            "gold_term_in_sentence": 0,
            "shares_sentence_with_gold": 0,
            "gold_elsewhere": 0,
            "evidence_lenient": 0,
        }

    roles = sentence_roles(prediction)

    def has_gold_term(source: SourceKey) -> bool:
        for sentence, _sources in roles.get(source, []):
            if any(term.lower() in sentence.lower() for term in question.gold_terms):
                return True
        return False

    def shares_sentence_with_gold(source: SourceKey) -> bool:
        for _sentence, sources in roles.get(source, []):
            if any(other in gold and other != source for other in sources):
                return True
        return False

    def cross_run_correct(source: SourceKey) -> bool:
        return bool(cited_correct.get((question.id, source)))

    def distance_le_2(source: SourceKey) -> bool:
        distance = page_distance(source, gold)
        return distance is not None and distance <= 2

    def evidence_lenient(source: SourceKey) -> bool:
        if source in gold:
            return True
        if source[0] not in gold_docs:
            return False
        return (
            distance_le_2(source)
            or cross_run_correct(source)
            or shares_sentence_with_gold(source)
            or bool(
                [other_id for other_id in gold_elsewhere.get(source, []) if other_id != question.id]
            )
        )

    extras: List[Dict[str, Any]] = []
    for source in sorted(cited - gold):
        extras.append(
            {
                "id": question.id,
                "source": list(source),
                "distance": page_distance(source, gold),
                "cross_run_any": cited_any.get((question.id, source), []),
                "cross_run_correct": cited_correct.get((question.id, source), []),
                "gold_elsewhere": [
                    other_id
                    for other_id in gold_elsewhere.get(source, [])
                    if other_id != question.id
                ],
                "gold_term_in_sentence": has_gold_term(source),
                "shares_sentence_with_gold": shares_sentence_with_gold(source),
                "evidence_lenient": evidence_lenient(source),
            }
        )

    intersection = cited & gold
    cited_docs = {doc for doc, _ in cited}
    return {
        "id": question.id,
        "question": question.question,
        "gold": sorted(gold),
        "cited": sorted(cited),
        "extras": extras,
        "gold_cited": len(intersection),
        "cited_count": len(cited),
        "strict_precision": len(intersection) / len(cited),
        "doc_precision": len(cited_docs & gold_docs) / len(cited_docs),
        "evidence_precision": sum(1 for source in cited if evidence_lenient(source)) / len(cited),
        "distance_le_2": sum(1 for item in extras if distance_le_2(tuple(item["source"]))),
        "cross_run_any": sum(1 for item in extras if item["cross_run_any"]),
        "cross_run_correct": sum(1 for item in extras if item["cross_run_correct"]),
        "gold_term_in_sentence": sum(1 for item in extras if item["gold_term_in_sentence"]),
        "shares_sentence_with_gold": sum(1 for item in extras if item["shares_sentence_with_gold"]),
        "gold_elsewhere": sum(1 for item in extras if item["gold_elsewhere"]),
        "evidence_lenient": sum(1 for item in extras if item["evidence_lenient"]),
    }


def evaluate_variants(
    prediction: eg.Prediction,
    question: eg.GoldenQuestion,
    cited_correct: SourceMap,
    gold_elsewhere: SourceMap,
) -> Dict[str, float]:
    """Sensitivity variants for one answered answerable question."""
    gold = {eg.source_key(source) for source in question.gold_sources}
    gold_docs = {doc for doc, _ in gold}
    cited = {eg.source_key(citation) for citation in prediction.citations}
    if not cited:
        return {
            "strict": 0.0,
            "same_doc_page": 0.0,
            "doc_level": 0.0,
            "distance_le_2": 0.0,
            "cross_run_correct": 0.0,
            "cross_run_or_distance": 0.0,
            "sentence_gold_term": 0.0,
            "shares_sentence_with_gold": 0.0,
            "evidence_lenient": 0.0,
        }

    roles = sentence_roles(prediction)

    def has_gold_term(source: SourceKey) -> bool:
        return any(
            any(term.lower() in sentence.lower() for term in question.gold_terms)
            for sentence, _sources in roles.get(source, [])
        )

    def shares_sentence_with_gold(source: SourceKey) -> bool:
        return any(
            any(other in gold and other != source for other in sources)
            for _sentence, sources in roles.get(source, [])
        )

    def cross_run_correct(source: SourceKey) -> bool:
        return bool(cited_correct.get((question.id, source)))

    def distance_le_2(source: SourceKey) -> bool:
        distance = page_distance(source, gold)
        return distance is not None and distance <= 2

    def evidence_lenient(source: SourceKey) -> bool:
        if source in gold:
            return True
        if source[0] not in gold_docs:
            return False
        return (
            distance_le_2(source)
            or cross_run_correct(source)
            or shares_sentence_with_gold(source)
            or bool(
                [other_id for other_id in gold_elsewhere.get(source, []) if other_id != question.id]
            )
        )

    def share(predicate: Callable[[SourceKey], bool]) -> float:
        return sum(1 for source in cited if predicate(source)) / len(cited)

    return {
        "strict": share(lambda source: source in gold),
        "same_doc_page": share(lambda source: source[0] in gold_docs),
        "doc_level": len({doc for doc, _ in cited} & gold_docs) / len({doc for doc, _ in cited}),
        "distance_le_2": share(lambda source: source in gold or distance_le_2(source)),
        "cross_run_correct": share(lambda source: source in gold or cross_run_correct(source)),
        "cross_run_or_distance": share(
            lambda source: source in gold or cross_run_correct(source) or distance_le_2(source)
        ),
        "sentence_gold_term": share(lambda source: source in gold or has_gold_term(source)),
        "shares_sentence_with_gold": share(
            lambda source: source in gold or shares_sentence_with_gold(source)
        ),
        "evidence_lenient": share(evidence_lenient),
    }


def oracle_analysis(
    per_question: Sequence[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Estimate how many extras may remain while precision stays >= 0.80.

    The dynamic program assumes gold citations are kept and answers do not
    change.  It is an upper bound for a perfect citation selector, not a
    measured policy.
    """
    total_answered = len(per_question)
    total_extras = sum(len(item["extras"]) for item in per_question)

    def max_kept(questions: Sequence[Tuple[int, int]]) -> Tuple[int, float]:
        negative = -1e9
        dp = [negative] * (sum(extras for _, extras in questions) + 1)
        dp[0] = 0.0
        for gold_cited, extras in questions:
            next_dp = [negative] * len(dp)
            for kept_total in range(len(dp)):
                if dp[kept_total] == negative:
                    continue
                for kept_here in range(extras + 1):
                    denominator = gold_cited + kept_here
                    value = gold_cited / denominator if denominator else 0.0
                    candidate = dp[kept_total] + value
                    index = kept_total + kept_here
                    if candidate > next_dp[index]:
                        next_dp[index] = candidate
            dp = next_dp
        feasible = [
            (kept, value / total_answered)
            for kept, value in enumerate(dp)
            if value / total_answered >= 0.80
        ]
        kept, precision = max(feasible, key=lambda item: item[0])
        return kept, round(precision, 4)

    current = [(item["gold_cited"], len(item["extras"])) for item in per_question]
    kept, precision = max_kept(current)
    all_removed = (
        sum(
            item["gold_cited"] / item["gold_cited"] if item["gold_cited"] else 0.0
            for item in per_question
        )
        / total_answered
    )

    # Hypothetical: the three questions with no gold citation are fixed by
    # retrieval (one gold page cited, no extras).  This isolates the value of
    # retrieval work from citation selection.
    fixed = [
        (
            1 if item["gold_cited"] == 0 else item["gold_cited"],
            0 if item["gold_cited"] == 0 else len(item["extras"]),
        )
        for item in per_question
    ]
    fixed_kept, fixed_precision = max_kept(fixed)

    return {
        "total_answered": total_answered,
        "total_extras": total_extras,
        "extras_without_gold": sum(
            len(item["extras"]) for item in per_question if item["gold_cited"] == 0
        ),
        "max_extras_kept": kept,
        "min_extras_removed": total_extras - kept,
        "precision_at_max_kept": precision,
        "precision_with_all_extras_removed": round(all_removed, 4),
        "retrieval_fixed_max_extras_kept": fixed_kept,
        "retrieval_fixed_min_extras_removed": sum(
            0 if item["gold_cited"] == 0 else len(item["extras"]) for item in per_question
        )
        - fixed_kept,
        "retrieval_fixed_precision": fixed_precision,
    }


def build_report(
    golden: eg.GoldenSet,
    golden_path: Path,
    predictions_path: Path,
    run_glob: str,
) -> Dict[str, Any]:
    audited = eg.load_predictions(predictions_path)
    other_runs = load_runs(predictions_path, run_glob)
    other_runs.pop(predictions_path.stem, None)
    cited_any, cited_correct = build_cross_run_support(golden, other_runs)
    gold_elsewhere = build_gold_elsewhere(golden)

    official = eg.evaluate(golden, audited)["metrics"]["citation_precision"]

    variants: Dict[str, List[float]] = defaultdict(list)
    per_question: List[Dict[str, Any]] = []
    for question in golden.items:
        prediction = audited[question.id]
        if not question.answerable or not eg.is_answered(prediction):
            continue
        question_variants = evaluate_variants(
            prediction,
            question,
            cited_correct,
            gold_elsewhere,
        )
        for name, value in question_variants.items():
            variants[name].append(value)
        audit = evaluate_question(
            question,
            prediction,
            cited_any,
            cited_correct,
            gold_elsewhere,
        )
        if audit is not None:
            per_question.append(audit)

    extras = [extra for item in per_question for extra in item["extras"]]
    support_counts = {
        "extras_total": len(extras),
        "same_document": sum(1 for item in extras if item["distance"] is not None),
        "distance_le_2": sum(
            1 for item in extras if item["distance"] is not None and item["distance"] <= 2
        ),
        "cross_run_any": sum(1 for item in extras if item["cross_run_any"]),
        "cross_run_correct": sum(1 for item in extras if item["cross_run_correct"]),
        "gold_elsewhere": sum(1 for item in extras if item["gold_elsewhere"]),
        "gold_term_in_sentence": sum(1 for item in extras if item["gold_term_in_sentence"]),
        "shares_sentence_with_gold": sum(1 for item in extras if item["shares_sentence_with_gold"]),
        "evidence_lenient": sum(1 for item in extras if item["evidence_lenient"]),
    }

    return {
        "inputs": {
            "golden": display_path(golden_path),
            "predictions": display_path(predictions_path),
            "cross_runs": sorted(other_runs),
        },
        "official_citation_precision": official,
        "sensitivity": {name: mean(values) for name, values in variants.items()},
        "support_counts": support_counts,
        "oracle": oracle_analysis(per_question),
        "questions": per_question,
        "extras": extras,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# H2-6: read-only citation precision audit\n")
    lines.append(
        "Официальная метрика: page-level precision, макро-среднее по отвеченным "
        "answerable-вопросам; цитата верна только при точном совпадении "
        "`(документ, страница)` с `gold_sources`.\n"
    )
    lines.append(
        f"- Предсказания: `{report['inputs']['predictions']}`\n"
        f"- Golden set: `{report['inputs']['golden']}`\n"
        f"- Кросс-прогоны: {len(report['inputs']['cross_runs'])}\n"
    )

    lines.append("\n## 1. Официальная метрика и sensitivity\n")
    lines.append("| Вариант | Precision |\n|---|---:|\n")
    lines.append(f"| strict (официальная) | {report['official_citation_precision']:.4f} |\n")
    labels = {
        "same_doc_page": "same-document, page ignored",
        "doc_level": "document-level",
        "distance_le_2": "same doc, page distance <= 2",
        "cross_run_correct": "same doc + cross-run correct citation",
        "cross_run_or_distance": "cross-run or distance <= 2",
        "sentence_gold_term": "sentence contains a gold term",
        "shares_sentence_with_gold": "shares sentence with a gold citation",
        "evidence_lenient": "lenient evidence union",
    }
    for name, label in labels.items():
        value = report["sensitivity"].get(name)
        if value is not None:
            lines.append(f"| {label} | {value:.4f} |\n")

    lines.append("\n## 2. Extra citations: support counts\n")
    lines.append("| Признак | Кол-во |\n|---|---:|\n")
    for name, value in report["support_counts"].items():
        lines.append(f"| {name} | {value} |\n")

    lines.append("\n## 3. Per-question breakdown\n")
    lines.append(
        "| id | gold cited | cited | extras | strict P | doc P | evidence P | "
        "cross-run correct | distance <= 2 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    for item in report["questions"]:
        lines.append(
            f"| {item['id']} | {item['gold_cited']} | {item['cited_count']} | "
            f"{len(item['extras'])} | {item['strict_precision']:.3f} | "
            f"{item['doc_precision']:.3f} | {item['evidence_precision']:.3f} | "
            f"{item['cross_run_correct']} | {item['distance_le_2']} |\n"
        )

    lines.append("\n## 4. Oracle estimate (gold citations kept)\n")
    oracle = report["oracle"]
    lines.append(f"- Answered answerable questions: {oracle['total_answered']}\n")
    lines.append(f"- Extra citations: {oracle['total_extras']}\n")
    lines.append(
        f"- Extras in questions without a gold citation: {oracle['extras_without_gold']}\n"
    )
    lines.append(
        f"- Max extras kept at P >= 0.80: {oracle['max_extras_kept']} "
        f"(P {oracle['precision_at_max_kept']:.4f}); "
        f"min extras removed: {oracle['min_extras_removed']}\n"
    )
    lines.append(
        f"- P with all extras removed: " f"{oracle['precision_with_all_extras_removed']:.4f}\n"
    )
    lines.append(
        f"- If the three retrieval-miss questions are fixed: max extras kept "
        f"{oracle['retrieval_fixed_max_extras_kept']}, min removed "
        f"{oracle['retrieval_fixed_min_extras_removed']}, "
        f"P {oracle['retrieval_fixed_precision']:.4f}\n"
    )

    lines.append("\n## 5. Extra citations detail\n")
    lines.append(
        "| id | source | distance | cross-run any | cross-run correct | "
        "gold elsewhere | gold term | shares sentence | lenient |\n"
        "|---|---|---:|---|---|---|---|---|---|\n"
    )
    for item in report["extras"]:
        doc, page = item["source"]
        lines.append(
            f"| {item.get('id', '')} | {doc} p.{page} | "
            f"{item['distance'] if item['distance'] is not None else '-'} | "
            f"{', '.join(item['cross_run_any']) or '-'} | "
            f"{', '.join(item['cross_run_correct']) or '-'} | "
            f"{', '.join(item['gold_elsewhere']) or '-'} | "
            f"{'yes' if item['gold_term_in_sentence'] else 'no'} | "
            f"{'yes' if item['shares_sentence_with_gold'] else 'no'} | "
            f"{'yes' if item['evidence_lenient'] else 'no'} |\n"
        )

    lines.append("\n## 6. Limitations\n")
    lines.append(
        "- Нет текста страниц: семантическая проверка «та ли это страница» "
        "невозможна локально; cross-run evidence косвенное.\n"
        "- `answer_is_correct` проверяет только gold-термины, а не полную "
        "семантическую корректность.\n"
        "- Oracle estimate предполагает, что gold-цитаты сохраняются, а "
        "ответы не меняются; это верхняя граница, а не результат политики.\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--run-glob", default=DEFAULT_RUN_GLOB)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    golden = eg.load_golden(args.golden)
    report = build_report(golden, args.golden, args.predictions, args.run_glob)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "official_citation_precision": report["official_citation_precision"],
                "sensitivity": report["sensitivity"],
                "support_counts": report["support_counts"],
                "oracle": report["oracle"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
