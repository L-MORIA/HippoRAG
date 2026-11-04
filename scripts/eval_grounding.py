#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оценка заземления Q&A на golden set (H2-6, D-9/D-10).

Скрипт отвечает на один вопрос: **можно ли доверять ответу** — то есть
отказывается ли система, когда ответа нет, и ссылается ли она на те
фрагменты, где ответ действительно есть.

Что считается:

* abstention rate и false answer rate на вопросах без ответа;
* answer rate на отвечаемых вопросах;
* citation precision/recall по размеченным источникам;
* marker validity — все ли ``[n]`` попадают в переданные фрагменты;
* uncited sentence rate — доля ответов с предложениями без ссылки;
* answer accuracy — покрытие ключевых терминов эталона;
* p50/p95 latency, если предсказания её содержат.

Режимы:

    # 1. Проверить разметку без моделей и сети
    python scripts/eval_grounding.py --validate-only

    # 2. Оценить готовые предсказания (основной офлайн-режим)
    python scripts/eval_grounding.py --predictions eval/grounding/after.jsonl

    # 3. Сравнить baseline и after
    python scripts/eval_grounding.py --baseline eval/grounding/before.jsonl \
        --predictions eval/grounding/after.jsonl

    # 4. Проверить механику без LM Studio (ответы строятся по эталону;
    #    это НЕ измерение качества)
    python scripts/eval_grounding.py --stub-llm --stub-mode grounded \
        --out-json /tmp/grounding_stub.json --out-md /tmp/grounding_stub.md

Формат предсказания (JSONL, по строке на вопрос)::

    {"id": "a01-astro-author-edition",
     "answer": "Ирина Максимова, 2025 [1].",
     "abstained": false,
     "citations": [{"doc_title": "Астрофэншуй квартиры.pdf", "page_num": 2}],
     "fragment_count": 3,
     "grounding_errors": [],
     "latency_ms": 812.0}

``retrieved`` необязателен: если он есть, ``fragment_count`` берётся из него.
Предсказания обязаны покрывать все вопросы выбранного набора.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from astro_app.evaluation import answer_body, normalize_text, term_matches  # noqa: E402

DEFAULT_GOLDEN = ROOT / "eval" / "grounding" / "golden_qa.jsonl"
DEFAULT_REPORT_JSON = ROOT / "eval" / "grounding" / "report_2026-11-04.json"
DEFAULT_REPORT_MD = ROOT / "eval" / "grounding" / "report_2026-11-04.md"

ABSTAIN_TEXT = "В предоставленных фрагментах нет ответа на этот вопрос."

_REQUIRED_FIELDS = {
    "id",
    "question",
    "answerable",
    "gold_answer",
    "gold_sources",
    "notes",
}
_OPTIONAL_FIELDS = {"borderline", "gold_terms"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{2,64}$")
_MARKER_RE = re.compile(r"\[\s*([0-9,\s]+?)\s*\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
_MIN_UNCITED_SENTENCE_LEN = 40
_STOPWORDS = {
    "или",
    "как",
    "что",
    "это",
    "для",
    "при",
    "над",
    "под",
    "без",
    "его",
    "её",
    "она",
    "они",
    "оно",
    "так",
    "все",
    "весь",
    "где",
    "когда",
    "который",
    "которая",
    "которые",
    "нужно",
    "можно",
    "есть",
    "быть",
    "были",
    "было",
}


class GoldenSetError(ValueError):
    """Разметка или предсказания не соответствуют контракту."""


@dataclass(frozen=True)
class SourceRef:
    """Источник ответа: документ и страница."""

    doc_title: str
    page_num: int


@dataclass
class GoldenQuestion:
    """Один вопрос golden set."""

    id: str
    question: str
    answerable: bool
    gold_answer: str
    gold_sources: List[SourceRef] = field(default_factory=list)
    gold_terms: List[str] = field(default_factory=list)
    notes: str = ""
    borderline: bool = False


@dataclass
class Prediction:
    """Ответ системы на один вопрос golden set."""

    id: str
    answer: str
    abstained: bool
    citations: List[SourceRef] = field(default_factory=list)
    fragment_count: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    retrieved: List[SourceRef] = field(default_factory=list)


@dataclass
class GoldenSet:
    """Набор вопросов и производные счётчики."""

    items: List[GoldenQuestion]

    @property
    def by_id(self) -> Dict[str, GoldenQuestion]:
        return {item.id: item for item in self.items}

    @property
    def answerable(self) -> List[GoldenQuestion]:
        return [item for item in self.items if item.answerable]

    @property
    def unanswerable(self) -> List[GoldenQuestion]:
        return [item for item in self.items if not item.answerable]

    @property
    def borderline(self) -> List[GoldenQuestion]:
        return [item for item in self.items if item.borderline]


def normalize_doc_title(title: Any) -> str:
    """Нормализует имя документа: регистр, пробелы и расширение .pdf."""
    text = normalize_text(title)
    if text.endswith(".pdf"):
        text = text[:-4].strip()
    return text


def source_key(source: SourceRef) -> Tuple[str, int]:
    """Ключ сравнения источников: документ без .pdf + страница."""
    return normalize_doc_title(source.doc_title), int(source.page_num)


def _source_from_dict(raw: Any, where: str) -> SourceRef:
    if not isinstance(raw, dict):
        raise GoldenSetError(f"{where}: источник должен быть объектом")
    unknown = set(raw) - {"doc_title", "page_num", "source_file"}
    if unknown:
        raise GoldenSetError(f"{where}: неизвестные поля источника {sorted(unknown)}")
    title = raw.get("doc_title") or raw.get("source_file")
    if not isinstance(title, str) or not title.strip():
        raise GoldenSetError(f"{where}: пустой doc_title")
    page = raw.get("page_num")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise GoldenSetError(f"{where}: page_num должен быть целым >= 1")
    return SourceRef(doc_title=title.strip(), page_num=page)


def _validate_terms(raw: Any, where: str) -> List[str]:
    if not isinstance(raw, list):
        raise GoldenSetError(f"{where}: gold_terms должен быть списком")
    terms: List[str] = []
    for term in raw:
        if not isinstance(term, str) or not term.strip():
            raise GoldenSetError(f"{where}: пустой термин в gold_terms")
        terms.append(term.strip().lower())
    return terms


def _derive_terms(answer: str, limit: int = 6) -> List[str]:
    """Запасной вариант, если gold_terms не заданы вручную."""
    tokens = re.findall(r"[a-zа-яё0-9][a-zа-яё0-9-]{2,}", normalize_text(answer))
    terms: List[str] = []
    for token in tokens:
        if token in _STOPWORDS or token in terms:
            continue
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def parse_golden_entry(raw: Any, where: str) -> GoldenQuestion:
    """Строгий разбор одной записи golden set."""
    if not isinstance(raw, dict):
        raise GoldenSetError(f"{where}: запись должна быть объектом")
    unknown = set(raw) - _ALLOWED_FIELDS
    missing = _REQUIRED_FIELDS - set(raw)
    if unknown:
        raise GoldenSetError(f"{where}: неизвестные поля {sorted(unknown)}")
    if missing:
        raise GoldenSetError(f"{where}: нет обязательных полей {sorted(missing)}")

    qid = raw["id"]
    if not isinstance(qid, str) or not _ID_RE.match(qid):
        raise GoldenSetError(f"{where}: id должен быть 2..64 символа [A-Za-z0-9_-]")
    question = raw["question"]
    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError(f"{where}: пустой question")
    if not isinstance(raw["answerable"], bool):
        raise GoldenSetError(f"{where}: answerable должен быть bool")
    answerable = raw["answerable"]
    gold_answer = raw["gold_answer"]
    if not isinstance(gold_answer, str):
        raise GoldenSetError(f"{where}: gold_answer должен быть строкой")
    if answerable and not gold_answer.strip():
        raise GoldenSetError(f"{where}: для answerable нужен непустой gold_answer")
    if not answerable and gold_answer.strip():
        raise GoldenSetError(f"{where}: у unanswerable gold_answer должен быть пустым")
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        raise GoldenSetError(f"{where}: notes должен быть строкой")
    borderline = raw.get("borderline", False)
    if not isinstance(borderline, bool):
        raise GoldenSetError(f"{where}: borderline должен быть bool")
    if borderline and not answerable:
        raise GoldenSetError(f"{where}: borderline может быть только answerable")

    raw_sources = raw["gold_sources"]
    if not isinstance(raw_sources, list):
        raise GoldenSetError(f"{where}: gold_sources должен быть списком")
    sources = [_source_from_dict(item, where) for item in raw_sources]
    if len({source_key(item) for item in sources}) != len(sources):
        raise GoldenSetError(f"{where}: повторяющиеся gold_sources")
    if answerable and not sources:
        raise GoldenSetError(f"{where}: для answerable нужен хотя бы один gold_source")
    if not answerable and sources:
        raise GoldenSetError(f"{where}: у unanswerable не должно быть gold_sources")
    if borderline and len(sources) < 2:
        raise GoldenSetError(f"{where}: borderline требует минимум два gold_sources")

    terms = _validate_terms(raw.get("gold_terms", []), where)
    if answerable and not terms:
        terms = _derive_terms(gold_answer)
    if answerable and not terms:
        raise GoldenSetError(f"{where}: не удалось определить gold_terms")
    if not answerable and terms:
        raise GoldenSetError(f"{where}: у unanswerable gold_terms должен быть пустым")

    return GoldenQuestion(
        id=qid,
        question=question.strip(),
        answerable=answerable,
        gold_answer=gold_answer.strip(),
        gold_sources=sources,
        gold_terms=terms,
        notes=notes.strip(),
        borderline=borderline,
    )


def load_golden(path: str | Path = DEFAULT_GOLDEN) -> GoldenSet:
    """Читает JSONL/JSON и проверяет схему."""
    path = Path(path)
    if not path.exists():
        raise GoldenSetError(f"нет файла разметки: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw_items = json.loads(text)
        if not isinstance(raw_items, list):
            raise GoldenSetError("JSON-разметка должна быть массивом")
    else:
        raw_items = []
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw_items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise GoldenSetError(f"{path}:{line_no}: битый JSON: {exc}") from exc

    items = [parse_golden_entry(raw, f"{path}:{index}") for index, raw in enumerate(raw_items, 1)]
    ids = [item.id for item in items]
    duplicates = sorted({qid for qid in ids if ids.count(qid) > 1})
    if duplicates:
        raise GoldenSetError(f"повторяющиеся id: {duplicates}")
    if not items:
        raise GoldenSetError("пустой golden set")
    return GoldenSet(items=items)


def validate_golden(
    golden: GoldenSet,
    *,
    enforce_distribution: bool = False,
) -> Tuple[List[str], List[str]]:
    """Проверяет состав набора; возвращает (ошибки, предупреждения)."""
    errors: List[str] = []
    warnings: List[str] = []
    plain_answerable = [item for item in golden.answerable if not item.borderline]
    if not golden.answerable:
        errors.append("в наборе нет answerable-вопросов")
    if not golden.unanswerable:
        errors.append("в наборе нет unanswerable-вопросов")

    if enforce_distribution:
        if not 30 <= len(golden.items) <= 40:
            errors.append(f"всего вопросов {len(golden.items)}, нужно 30–40")
        if not 15 <= len(plain_answerable) <= 20:
            errors.append(f"обычных answerable {len(plain_answerable)}, нужно 15–20")
        if not 10 <= len(golden.unanswerable) <= 15:
            errors.append(f"unanswerable {len(golden.unanswerable)}, нужно 10–15")
        if not 3 <= len(golden.borderline) <= 5:
            errors.append(f"пограничных {len(golden.borderline)}, нужно 3–5")

    for item in golden.answerable:
        if not item.gold_sources:
            errors.append(f"{item.id}: answerable без gold_sources")
        if not item.gold_terms:
            errors.append(f"{item.id}: answerable без gold_terms")
    for item in golden.unanswerable:
        if item.gold_sources:
            errors.append(f"{item.id}: unanswerable с gold_sources")
    for item in golden.borderline:
        if len({source_key(source) for source in item.gold_sources}) < 2:
            errors.append(f"{item.id}: borderline должен ссылаться на два фрагмента")

    if len(golden.items) < 30:
        warnings.append("набор меньше 30 вопросов: метрики приёмки неполные")
    return errors, warnings


def _prediction_source(raw: Any, where: str) -> SourceRef:
    if isinstance(raw, dict):
        return _source_from_dict(raw, where)
    raise GoldenSetError(f"{where}: цитата должна быть объектом doc_title/page_num")


def parse_prediction(raw: Any, where: str) -> Prediction:
    """Разбирает одно предсказание; лишние поля допустимы."""
    if not isinstance(raw, dict):
        raise GoldenSetError(f"{where}: предсказание должно быть объектом")
    qid = raw.get("id")
    if not isinstance(qid, str) or not _ID_RE.match(qid):
        raise GoldenSetError(f"{where}: нет корректного id")
    answer = raw.get("answer", "")
    if not isinstance(answer, str):
        raise GoldenSetError(f"{where}: answer должен быть строкой")
    abstained = raw.get("abstained")
    if abstained is None:
        abstained = answer.strip() == ABSTAIN_TEXT
    if not isinstance(abstained, bool):
        raise GoldenSetError(f"{where}: abstained должен быть bool")
    citations_raw = raw.get("citations", [])
    if not isinstance(citations_raw, list):
        raise GoldenSetError(f"{where}: citations должен быть списком")
    citations = [
        _prediction_source(item, f"{where}.citations[{index}]")
        for index, item in enumerate(citations_raw)
    ]
    retrieved_raw = raw.get("retrieved", [])
    if not isinstance(retrieved_raw, list):
        raise GoldenSetError(f"{where}: retrieved должен быть списком")
    retrieved = [
        _prediction_source(item, f"{where}.retrieved[{index}]")
        for index, item in enumerate(retrieved_raw)
    ]
    fragment_count = raw.get("fragment_count")
    if fragment_count is None and retrieved:
        fragment_count = len(retrieved)
    if fragment_count is not None and (
        isinstance(fragment_count, bool)
        or not isinstance(fragment_count, int)
        or fragment_count < 0
    ):
        raise GoldenSetError(f"{where}: fragment_count должен быть целым >= 0")
    errors_raw = raw.get("grounding_errors", raw.get("errors", []))
    if not isinstance(errors_raw, list) or any(not isinstance(item, str) for item in errors_raw):
        raise GoldenSetError(f"{where}: grounding_errors должен быть списком строк")
    latency = raw.get("latency_ms")
    if latency is not None and not isinstance(latency, (int, float)):
        raise GoldenSetError(f"{where}: latency_ms должен быть числом")
    return Prediction(
        id=qid,
        answer=answer,
        abstained=abstained,
        citations=citations,
        fragment_count=fragment_count,
        errors=list(errors_raw),
        latency_ms=float(latency) if latency is not None else None,
        retrieved=retrieved,
    )


def load_predictions(path: str | Path) -> Dict[str, Prediction]:
    """Читает JSONL/JSON с предсказаниями и проверяет уникальность id."""
    path = Path(path)
    if not path.exists():
        raise GoldenSetError(f"нет файла предсказаний: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw_items = json.loads(text)
        if not isinstance(raw_items, list):
            raise GoldenSetError("JSON-предсказания должны быть массивом")
    else:
        raw_items = []
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                raw_items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise GoldenSetError(f"{path}:{line_no}: битый JSON: {exc}") from exc
    predictions: Dict[str, Prediction] = {}
    for index, raw in enumerate(raw_items, 1):
        prediction = parse_prediction(raw, f"{path}:{index}")
        if prediction.id in predictions:
            raise GoldenSetError(f"{path}: повторяющийся id {prediction.id}")
        predictions[prediction.id] = prediction
    return predictions


def is_answered(prediction: Prediction) -> bool:
    """Ответ содержательный, а не отказ и не пустая строка."""
    text = prediction.answer.strip()
    return bool(text) and not prediction.abstained and text != ABSTAIN_TEXT


def marker_counts(prediction: Prediction) -> Optional[Tuple[int, int]]:
    """(всего маркеров, валидных) или None, если границы фрагментов неизвестны."""
    if prediction.fragment_count is None:
        return None
    total = 0
    valid = 0
    for group in _MARKER_RE.findall(prediction.answer):
        for part in re.findall(r"\d+", group):
            total += 1
            if 1 <= int(part) <= prediction.fragment_count:
                valid += 1
    return total, valid


def has_uncited_sentence(answer: str) -> bool:
    """Есть ли длинное предложение без inline-маркера."""
    for sentence in _SENTENCE_RE.split(answer):
        if len(sentence.strip()) >= _MIN_UNCITED_SENTENCE_LEN and not _MARKER_RE.search(sentence):
            return True
    return False


def _term_present(term: str, text_norm: str) -> bool:
    """Термин есть в тексте; числа проверяются по границе цифр."""
    term_norm = normalize_text(term)
    if not term_norm:
        return False
    if re.fullmatch(r"[0-9][0-9-]*", term_norm):
        return re.search(rf"(?<!\d){re.escape(term_norm)}(?!\d)", text_norm) is not None
    return term_matches(term_norm, text_norm)


def answer_is_correct(prediction: Prediction, question: GoldenQuestion) -> bool:
    """Все ключевые термины эталона присутствуют в теле ответа."""
    if not question.answerable or not is_answered(prediction):
        return False
    body = normalize_text(answer_body(prediction.answer))
    return all(_term_present(term, body) for term in question.gold_terms)


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(statistics.fmean(values), 4) if values else None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def evaluate(
    golden: GoldenSet,
    predictions: Dict[str, Prediction],
) -> Dict[str, Any]:
    """Считает метрики и построчную таблицу."""
    known_ids = set(golden.by_id)
    unknown = sorted(set(predictions) - known_ids)
    missing = sorted(known_ids - set(predictions))
    if unknown:
        raise GoldenSetError(f"предсказания для неизвестных id: {unknown}")
    if missing:
        raise GoldenSetError(f"нет предсказаний для id: {missing}")

    rows: List[Dict[str, Any]] = []
    answered_answerable = 0
    abstained_unanswerable = 0
    false_answers = 0
    correct_answers = 0
    precisions: List[float] = []
    recalls: List[float] = []
    marker_total = 0
    marker_valid = 0
    answered_with_uncited = 0
    answered_count = 0
    latencies: List[float] = []

    for question in golden.items:
        prediction = predictions[question.id]
        answered = is_answered(prediction)
        cited = {source_key(source) for source in prediction.citations}
        gold = {source_key(source) for source in question.gold_sources}
        intersection = cited & gold

        marker = marker_counts(prediction)
        if prediction.latency_ms is not None:
            latencies.append(prediction.latency_ms)

        row: Dict[str, Any] = {
            "id": question.id,
            "answerable": question.answerable,
            "borderline": question.borderline,
            "answered": answered,
            "abstained": not answered,
            "citations": len(cited),
            "gold_sources": len(gold),
            "errors": list(prediction.errors),
        }

        if question.answerable:
            if answered:
                answered_answerable += 1
                answered_count += 1
                precision = len(intersection) / len(cited) if cited else 0.0
                recall = len(intersection) / len(gold) if gold else None
                precisions.append(precision)
                if recall is not None:
                    recalls.append(recall)
                row["citation_precision"] = round(precision, 4)
                row["citation_recall"] = round(recall, 4) if recall is not None else None
                row["correct"] = answer_is_correct(prediction, question)
                if row["correct"]:
                    correct_answers += 1
                if has_uncited_sentence(prediction.answer):
                    answered_with_uncited += 1
                    row["uncited_sentences"] = True
                else:
                    row["uncited_sentences"] = False
            else:
                row["citation_precision"] = None
                row["citation_recall"] = None
                row["correct"] = False
                row["uncited_sentences"] = None
        else:
            if answered:
                false_answers += 1
                answered_count += 1
                if has_uncited_sentence(prediction.answer):
                    answered_with_uncited += 1
                    row["uncited_sentences"] = True
                else:
                    row["uncited_sentences"] = False
            else:
                abstained_unanswerable += 1
                row["uncited_sentences"] = None

        if answered:
            if marker is None:
                row["marker_validity"] = None
            elif marker[0] == 0:
                row["marker_validity"] = 0.0
            else:
                marker_total += marker[0]
                marker_valid += marker[1]
                row["marker_validity"] = round(marker[1] / marker[0], 4)
        else:
            row["marker_validity"] = None

        rows.append(row)

    n_answerable = len(golden.answerable)
    n_unanswerable = len(golden.unanswerable)
    marker_validity = (
        round(marker_valid / marker_total, 4) if marker_total else (0.0 if answered_count else None)
    )
    metrics: Dict[str, Any] = {
        "n_total": len(golden.items),
        "n_answerable": n_answerable,
        "n_plain_answerable": len(golden.answerable) - len(golden.borderline),
        "n_borderline": len(golden.borderline),
        "n_unanswerable": n_unanswerable,
        "answered_answerable": answered_answerable,
        "abstained_unanswerable": abstained_unanswerable,
        "answer_rate_answerable": (
            round(answered_answerable / n_answerable, 4) if n_answerable else None
        ),
        "abstention_rate_unanswerable": (
            round(abstained_unanswerable / n_unanswerable, 4) if n_unanswerable else None
        ),
        "false_answer_rate": (round(false_answers / n_unanswerable, 4) if n_unanswerable else None),
        "citation_precision": _mean(precisions),
        "citation_recall": _mean(recalls),
        "marker_validity": marker_validity,
        "uncited_sentence_rate": (
            round(answered_with_uncited / answered_count, 4) if answered_count else None
        ),
        "answer_accuracy": (round(correct_answers / n_answerable, 4) if n_answerable else None),
        "p50_latency_ms": _percentile(latencies, 0.5),
        "p95_latency_ms": _percentile(latencies, 0.95),
    }
    return {"metrics": metrics, "questions": rows}


#: Пороги из плана DEEPSEEK §6.4.
THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "abstention_rate_unanswerable": (">=", 0.90),
    "false_answer_rate": ("<=", 0.10),
    "answer_rate_answerable": (">=", 0.90),
    "citation_precision": (">=", 0.80),
    "citation_recall": (">=", 0.80),
    "marker_validity": (">=", 1.00),
    "uncited_sentence_rate": ("<=", 0.10),
    "answer_accuracy": (">=", 0.70),
}


def check_thresholds(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Сверяет метрики с порогами; None считается провалом."""
    result: Dict[str, Dict[str, Any]] = {}
    for name, (operator, threshold) in THRESHOLDS.items():
        value = metrics.get(name)
        if value is None:
            passed = False
        elif operator == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        result[name] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }
    return result


def _question_ok(row: Dict[str, Any]) -> bool:
    if row["answerable"]:
        precision = row.get("citation_precision")
        recall = row.get("citation_recall")
        return bool(
            row["answered"]
            and row.get("correct")
            and precision is not None
            and precision >= 0.8
            and recall is not None
            and recall >= 0.8
            and not row.get("uncited_sentences")
        )
    return bool(row["abstained"])


def compare_runs(
    baseline: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    """Сравнивает две сводки: метрики и построчные изменения."""
    metric_delta: Dict[str, Any] = {}
    for name in THRESHOLDS:
        before = baseline["metrics"].get(name)
        current = after["metrics"].get(name)
        metric_delta[name] = {
            "before": before,
            "after": current,
            "delta": (
                round(current - before, 4)
                if isinstance(before, (int, float)) and isinstance(current, (int, float))
                else None
            ),
        }

    before_rows = {row["id"]: row for row in baseline["questions"]}
    after_rows = {row["id"]: row for row in after["questions"]}
    improved: List[str] = []
    regressed: List[str] = []
    for qid in sorted(before_rows):
        if qid not in after_rows:
            continue
        was = _question_ok(before_rows[qid])
        now = _question_ok(after_rows[qid])
        if now and not was:
            improved.append(qid)
        elif was and not now:
            regressed.append(qid)
    return {
        "metrics": metric_delta,
        "improved": improved,
        "regressed": regressed,
        "n_improved": len(improved),
        "n_regressed": len(regressed),
    }


def _round_source(source: SourceRef) -> Dict[str, Any]:
    return {"doc_title": source.doc_title, "page_num": source.page_num}


def predictions_to_jsonl(predictions: Iterable[Prediction]) -> str:
    """Сериализует предсказания в JSONL для повторного прогона."""
    lines = []
    for prediction in predictions:
        lines.append(
            json.dumps(
                {
                    "id": prediction.id,
                    "answer": prediction.answer,
                    "abstained": prediction.abstained,
                    "citations": [_round_source(item) for item in prediction.citations],
                    "retrieved": [_round_source(item) for item in prediction.retrieved],
                    "fragment_count": prediction.fragment_count,
                    "grounding_errors": prediction.errors,
                    "latency_ms": prediction.latency_ms,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _fragment_from_raw(raw: Any, where: str) -> Any:
    """Превращает результат retriever'а в astro_app.grounded_qa.Fragment."""
    from astro_app.grounded_qa import Fragment

    if isinstance(raw, Fragment):
        return raw
    if isinstance(raw, dict):
        text = raw.get("text", "")
        if not isinstance(text, str):
            raise GoldenSetError(f"{where}: text должен быть строкой")
        return Fragment(
            text=text,
            doc_title=str(raw.get("doc_title") or raw.get("source_file") or "Источник"),
            page_num=int(raw.get("page_num") or 0),
            section=str(raw.get("section") or ""),
            source_file=str(raw.get("source_file") or ""),
        )
    if isinstance(raw, str):
        return Fragment(text=raw)
    raise GoldenSetError(f"{where}: фрагмент должен быть dict, str или Fragment")


def _load_retriever(spec: str):
    if ":" not in spec:
        raise GoldenSetError("--retriever должен быть в формате module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise GoldenSetError(f"в {module_name} нет функции {function_name}")
    return function


def _stub_chat(question: GoldenQuestion, fragment_count: int):
    """Фейковая LLM для проверки механики, а не качества."""

    def chat(messages: List[Dict[str, str]]) -> str:  # noqa: ARG001
        if not question.answerable:
            return f"ОТВЕТ: {ABSTAIN_TEXT}\nИСТОЧНИКИ:\nОТКАЗ: да"
        ids = ",".join(str(index) for index in range(1, fragment_count + 1))
        markers = "".join(f"[{index}]" for index in range(1, fragment_count + 1))
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_RE.split(question.gold_answer)
            if sentence.strip()
        ]
        cited_sentences = []
        for sentence in sentences:
            match = re.match(r"^(.*?)([.!?…]+)$", sentence)
            if match:
                cited_sentences.append(f"{match.group(1).strip()} {markers}{match.group(2)}")
            else:
                cited_sentences.append(f"{sentence} {markers}")
        cited_answer = " ".join(cited_sentences)
        return f"ОТВЕТ: {cited_answer}\n" f"ИСТОЧНИКИ: {ids}\n" f"ОТКАЗ: нет"

    return chat


def run_questions(
    golden: GoldenSet,
    *,
    retriever=None,
    stub_mode: Optional[str] = None,
) -> List[Prediction]:
    """Прогоняет вопросы через retriever + answer_grounded.

    ``stub_mode``:
      * ``grounded`` — правильный ответ с цитатами по gold_sources;
      * ``legacy`` — ответ без цитат, как старый формат (baseline-заглушка);
      * ``abstain`` — все вопросы получают отказ.
    """
    from astro_app.grounded_qa import Fragment, answer_grounded

    if stub_mode and stub_mode not in {"grounded", "legacy", "abstain"}:
        raise GoldenSetError(f"неизвестный stub-mode: {stub_mode}")
    if not stub_mode and retriever is None:
        raise GoldenSetError("нужен retriever или stub_mode")

    predictions: List[Prediction] = []
    for question in golden.items:
        started = time.perf_counter()

        if stub_mode:
            fragments = [
                Fragment(
                    text=question.gold_answer or "нет данных",
                    doc_title=source.doc_title,
                    page_num=source.page_num,
                )
                for source in question.gold_sources
            ]
        else:
            raw_fragments = retriever(question.question) or []
            fragments = [
                _fragment_from_raw(raw, f"{question.id}.retrieved[{index}]")
                for index, raw in enumerate(raw_fragments)
            ]

        if stub_mode == "legacy":
            answer = question.gold_answer if question.answerable else ABSTAIN_TEXT
            grounded = None
            prediction = Prediction(
                id=question.id,
                answer=answer,
                abstained=not question.answerable,
                citations=[],
                retrieved=[SourceRef(item.doc_title, item.page_num) for item in fragments],
                fragment_count=len(fragments),
                errors=[],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            predictions.append(prediction)
            continue

        if stub_mode == "abstain":
            prediction = Prediction(
                id=question.id,
                answer=ABSTAIN_TEXT,
                abstained=True,
                citations=[],
                retrieved=[SourceRef(item.doc_title, item.page_num) for item in fragments],
                fragment_count=len(fragments),
                errors=[],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            predictions.append(prediction)
            continue

        chat = _stub_chat(question, len(fragments)) if stub_mode == "grounded" else None
        grounded = answer_grounded(question.question, fragments, chat=chat)
        used = [
            SourceRef(fragments[index - 1].doc_title, fragments[index - 1].page_num)
            for index in grounded.used_citation_ids
            if 1 <= index <= len(fragments)
        ]
        predictions.append(
            Prediction(
                id=question.id,
                answer=grounded.text,
                abstained=grounded.abstained,
                citations=used,
                retrieved=[SourceRef(item.doc_title, item.page_num) for item in fragments],
                fragment_count=len(fragments),
                errors=grounded.errors,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )
    return predictions


def _fmt(value: Any, *, percent: bool = False, digits: int = 3) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{value * 100:.1f}%"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_markdown(
    golden: GoldenSet,
    report: Dict[str, Any],
    *,
    title: str = "Заземление Q&A: отчёт golden set",
) -> str:
    """Человекочитаемый отчёт для коммита и ревью."""
    metrics = report["metrics"]
    mode = report.get("mode", "predictions")
    lines = [
        f"# {title}",
        "",
        f"- Дата: {report.get('generated_at', '—')}",
        f"- Режим: `{mode}`",
        f"- Разметка: `{report.get('dataset', '—')}`",
        f"- Вопросов: {metrics['n_total']} "
        f"(answerable {metrics['n_answerable']}, "
        f"borderline {metrics['n_borderline']}, "
        f"unanswerable {metrics['n_unanswerable']})",
        "",
    ]
    baseline = report.get("baseline") or {}
    if baseline.get("dataset"):
        lines.insert(5, f"- Baseline: `{baseline['dataset']}`")
    if mode.startswith("stub"):
        lines += [
            "> **Это проверка механики, а не качества.** Ответы построены по "
            "эталону, без индекса и LM Studio. Для оценки модели запустите "
            "`--predictions` на реальных ответах.",
            "",
        ]

    lines += [
        "## Метрики",
        "",
        "| Метрика | Значение | Порог | Статус |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "abstention_rate_unanswerable": "Abstention rate (unanswerable)",
        "false_answer_rate": "False answer rate",
        "answer_rate_answerable": "Answer rate (answerable)",
        "citation_precision": "Citation precision",
        "citation_recall": "Citation recall",
        "marker_validity": "Marker validity",
        "uncited_sentence_rate": "Uncited sentence rate",
        "answer_accuracy": "Answer accuracy",
    }
    for name, check in report["thresholds"].items():
        value = check["value"]
        percent = name not in {"citation_precision", "citation_recall", "marker_validity"}
        display = _fmt(value, percent=percent)
        if name in {"citation_precision", "citation_recall", "marker_validity"}:
            display = _fmt(value)
        threshold = check["threshold"]
        threshold_text = (
            f"{check['operator']} {_fmt(threshold, percent=percent)}"
            if name not in {"citation_precision", "citation_recall", "marker_validity"}
            else f"{check['operator']} {_fmt(threshold)}"
        )
        lines.append(
            f"| {labels[name]} | {display} | {threshold_text} | "
            f"{'✅' if check['passed'] else '❌'} |"
        )

    if metrics.get("p50_latency_ms") is not None:
        lines += [
            "",
            f"Латентность: p50 {metrics['p50_latency_ms']} мс, "
            f"p95 {metrics['p95_latency_ms']} мс.",
        ]

    comparison = report.get("comparison")
    if comparison:
        lines += [
            "",
            "## Baseline → after",
            "",
            "| Метрика | Baseline | After | Δ |",
            "|---|---:|---:|---:|",
        ]
        for name, delta in comparison["metrics"].items():
            lines.append(
                f"| {labels.get(name, name)} | {_fmt(delta['before'])} | "
                f"{_fmt(delta['after'])} | {_fmt(delta['delta'])} |"
            )
        lines += [
            "",
            f"Улучшились: {comparison['n_improved']} "
            f"({', '.join(comparison['improved']) or '—'}).",
            "",
            f"Ухудшились: {comparison['n_regressed']} "
            f"({', '.join(comparison['regressed']) or '—'}).",
        ]

    lines += [
        "",
        "## Вопросы",
        "",
        "| id | answerable | ответ | цитаты P/R | accuracy | ошибки |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["questions"]:
        precision = row.get("citation_precision")
        recall = row.get("citation_recall")
        citation_text = (
            f"{_fmt(precision)} / {_fmt(recall)}"
            if precision is not None or recall is not None
            else "—"
        )
        accuracy = row.get("correct")
        lines.append(
            f"| `{row['id']}` | {'да' if row['answerable'] else 'нет'} | "
            f"{'ответ' if row['answered'] else 'отказ'} | {citation_text} | "
            f"{'✅' if accuracy else ('❌' if accuracy is not None else '—')} | "
            f"{', '.join(row['errors']) or '—'} |"
        )

    lines += [
        "",
        "## Как читать",
        "",
        "- Citation precision/recall считаются по размеченным `gold_sources`; "
        "для ответа без цитат precision = 0.",
        "- Answer accuracy — все ключевые термины эталона найдены в теле ответа. "
        "Это не LLM-судья и не проверка истинности.",
        "- Marker validity = валидные `[n]` / все `[n]`; ответ без маркеров "
        "при наличии ответа даёт 0.",
        "- Uncited sentence rate — доля ответов, где есть предложение длиннее "
        "40 символов без `[n]`.",
        "",
    ]
    return "\n".join(lines)


def _display_path(path: str | Path) -> str:
    """Путь относительно корня репозитория, если это возможно."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_report(
    golden: GoldenSet,
    result: Dict[str, Any],
    *,
    dataset_path: str | Path,
    mode: str,
    baseline: Optional[Dict[str, Any]] = None,
    baseline_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Собирает JSON-отчёт, пригодный для сравнения прогонов."""
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": _display_path(dataset_path),
        "mode": mode,
        "quality_measurement": not mode.startswith("stub"),
        "metrics": result["metrics"],
        "thresholds": check_thresholds(result["metrics"]),
        "questions": result["questions"],
    }
    if baseline is not None:
        report["baseline"] = {
            "dataset": _display_path(baseline_path) if baseline_path else None,
            "metrics": baseline["metrics"],
        }
        report["comparison"] = compare_runs(baseline, result)
    return report


def _print_summary(report: Dict[str, Any]) -> None:
    metrics = report["metrics"]
    print(
        f"Вопросов: {metrics['n_total']} "
        f"(answerable {metrics['n_answerable']}, "
        f"unanswerable {metrics['n_unanswerable']})"
    )
    print(
        f"Answer rate: {_fmt(metrics['answer_rate_answerable'], percent=True)} | "
        f"Abstention rate: {_fmt(metrics['abstention_rate_unanswerable'], percent=True)} | "
        f"False answer rate: {_fmt(metrics['false_answer_rate'], percent=True)}"
    )
    print(
        f"Citation P/R: {_fmt(metrics['citation_precision'])} / "
        f"{_fmt(metrics['citation_recall'])} | "
        f"Marker validity: {_fmt(metrics['marker_validity'])} | "
        f"Uncited: {_fmt(metrics['uncited_sentence_rate'], percent=True)} | "
        f"Accuracy: {_fmt(metrics['answer_accuracy'], percent=True)}"
    )
    failed = [name for name, check in report["thresholds"].items() if not check["passed"]]
    if failed:
        print("Пороги не выполнены: " + ", ".join(failed))
    else:
        print("Все пороги выполнены.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="JSONL/JSON")
    parser.add_argument("--predictions", help="JSONL/JSON с ответами after")
    parser.add_argument("--baseline", help="JSONL/JSON с ответами before")
    parser.add_argument(
        "--baseline-label",
        help="подпись baseline в отчёте (по умолчанию путь к файлу)",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="прогнать retriever + LLM")
    parser.add_argument("--retriever", help="module:function для --run")
    parser.add_argument(
        "--stub-llm",
        action="store_true",
        help="фейковая LLM: проверка механики, не качества",
    )
    parser.add_argument(
        "--stub-mode",
        default="grounded",
        choices=["grounded", "legacy", "abstain"],
        help="что имитировать в stub-режиме",
    )
    parser.add_argument("--limit", type=int, help="первые N вопросов")
    parser.add_argument("--out-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_REPORT_MD))
    parser.add_argument(
        "--out-predictions",
        help="куда сохранить предсказания режима --run/--stub-llm",
    )
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="не возвращать код 1 при провале порогов",
    )
    parser.add_argument(
        "--no-distribution-check",
        action="store_true",
        help="не проверять состав 30–40 / 15–20 / 10–15 / 3–5",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        golden = load_golden(args.golden)
        if args.limit is not None:
            if args.limit < 1:
                raise GoldenSetError("--limit должен быть >= 1")
            golden = GoldenSet(items=golden.items[: args.limit])

        errors, warnings = validate_golden(
            golden,
            enforce_distribution=(not args.no_distribution_check and args.limit is None),
        )
        for warning in warnings:
            print(f"предупреждение: {warning}", file=sys.stderr)
        if errors:
            for error in errors:
                print(f"ошибка: {error}", file=sys.stderr)
            return 2
        print(
            f"golden set: {len(golden.items)} вопросов "
            f"(answerable {len(golden.answerable)}, "
            f"borderline {len(golden.borderline)}, "
            f"unanswerable {len(golden.unanswerable)})"
        )
        if args.validate_only:
            print("разметка корректна")
            return 0

        baseline_result = None
        if args.baseline:
            baseline_predictions = load_predictions(args.baseline)
            baseline_result = evaluate(golden, baseline_predictions)

        if args.predictions:
            predictions = load_predictions(args.predictions)
            mode = "predictions"
        elif args.stub_llm:
            predictions_list = run_questions(golden, stub_mode=args.stub_mode)
            predictions = {item.id: item for item in predictions_list}
            mode = f"stub-{args.stub_mode}"
            if args.out_predictions:
                Path(args.out_predictions).write_text(
                    predictions_to_jsonl(predictions_list), encoding="utf-8"
                )
        elif args.run:
            if not args.retriever:
                raise GoldenSetError("для --run нужен --retriever module:function")
            retriever = _load_retriever(args.retriever)
            predictions_list = run_questions(golden, retriever=retriever)
            predictions = {item.id: item for item in predictions_list}
            mode = "run"
            if args.out_predictions:
                Path(args.out_predictions).write_text(
                    predictions_to_jsonl(predictions_list), encoding="utf-8"
                )
        else:
            raise GoldenSetError("нужен --predictions, --stub-llm или --run")

        result = evaluate(golden, predictions)
        report = build_report(
            golden,
            result,
            dataset_path=args.golden,
            mode=mode,
            baseline=baseline_result,
            baseline_path=args.baseline_label or args.baseline,
        )
        _print_summary(report)

        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.out_md:
            Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_md).write_text(format_markdown(golden, report), encoding="utf-8")

        if args.no_thresholds:
            return 0
        return 0 if all(check["passed"] for check in report["thresholds"].values()) else 1
    except GoldenSetError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
