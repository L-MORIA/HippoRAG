#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2: предсказания legacy-режимов 12/19/21 для eval_grounding.py.

Референсная реализация из docs/TASK_M2_LEGACY_MODES_2026-11-05.md.
Скрипт не входит в package data; положите его в scripts/ на машине замера.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from astro_app.config import LLM_BASE_URL, LLM_NAME  # noqa: E402
from astro_app.modes import MODES  # noqa: E402
from astro_app.modes import graph as graph_mode  # noqa: E402
from astro_app.modes import self_rag as self_rag_mode  # noqa: E402
from astro_app.modes import sentence_dat as sentence_dat_mode  # noqa: E402
from hipporag.qa.grounding import ABSTAIN_TEXT  # noqa: E402

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
QA_TOP_K = 5
MARKER_RE = re.compile(r"\[\s*([0-9,\s]+?)\s*\]")
SECTION_RE = re.compile(r"===\s*([А-ЯЁ ]+?)\s*===")
SOURCE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?),\s*стр\.\s*(\d+)", re.M)


def read_jsonl(path: Path):
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: {exc}") from exc
    return items


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


def split_sections(text: str) -> dict[str, str]:
    parts = SECTION_RE.split(text or "")
    sections: dict[str, str] = {}
    for index in range(1, len(parts) - 1, 2):
        sections[parts[index].strip()] = parts[index + 1].strip()
    return sections


def raw_to_text(raw) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        if not raw:
            return ""
        first = raw[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        return str(first)
    return str(raw)


def extract_answer(text: str) -> str:
    sections = split_sections(text)
    if "ОТВЕТ" in sections:
        return sections["ОТВЕТ"].strip()
    return (text or "").strip()


def parse_marker_ids(text: str) -> list[int]:
    ids: set[int] = set()
    for group in MARKER_RE.findall(text or ""):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
    return sorted(ids)


def parse_source_lines(text: str) -> list[dict]:
    return [
        {"doc_title": match.group(2).strip(), "page_num": int(match.group(3))}
        for match in SOURCE_LINE_RE.finditer(text or "")
    ]


def source_of(fragment) -> dict:
    return {
        "doc_title": str(getattr(fragment, "doc_title", "") or "Источник"),
        "page_num": int(getattr(fragment, "page_num", 0) or 0),
    }


def has_page(source: dict) -> bool:
    return source.get("page_num", 0) >= 1


def dedupe(sources: list[dict]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    result: list[dict] = []
    for source in sources:
        key = (source["doc_title"], source["page_num"])
        if key not in seen:
            seen.add(key)
            result.append(source)
    return result


class GroundedSpy:
    """Перехватывает fragments, переданные в answer_grounded."""

    def __init__(self, module):
        self.module = module
        self.original = module.answer_grounded
        self.fragments = []
        self.grounded = None
        self.calls = 0

    def __enter__(self):
        spy = self

        def wrapper(question, fragments, *args, **kwargs):
            spy.fragments = list(fragments)
            result = spy.original(question, fragments, *args, **kwargs)
            spy.grounded = result
            spy.calls += 1
            return result

        self.module.answer_grounded = wrapper
        return self

    def __exit__(self, *exc_info):
        self.module.answer_grounded = self.original
        return False


def run_one(mode_id: str, hipporag, question: str):
    spec = MODES[mode_id]
    spies = [
        GroundedSpy(graph_mode),
        GroundedSpy(self_rag_mode),
        GroundedSpy(sentence_dat_mode),
    ]
    for spy in spies:
        spy.__enter__()

    started = time.perf_counter()
    error = None
    raw = None
    try:
        kwargs = {}
        if "top_k" in inspect.signature(spec.fn).parameters:
            kwargs["top_k"] = QA_TOP_K
        raw = spec.fn(hipporag, question, **kwargs)
    except Exception as exc:  # один сбойный вопрос не должен ронять прогон
        error = f"{type(exc).__name__}: {exc}"
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        for spy in reversed(spies):
            spy.__exit__(None, None, None)

    text = raw_to_text(raw)
    answer = extract_answer(text)
    by_module = {spy.module.__name__: spy for spy in spies}
    if mode_id == "19":
        preferred = by_module.get("astro_app.modes.graph")
        fallback = by_module.get("astro_app.modes.self_rag")
        spy = preferred if preferred and preferred.calls else fallback
    elif mode_id == "12":
        spy = by_module.get("astro_app.modes.self_rag")
    else:
        spy = by_module.get("astro_app.modes.sentence_dat")

    fragments = list(spy.fragments) if spy else []
    grounded = spy.grounded if spy else None
    abstained = bool(getattr(grounded, "abstained", False))
    if not answer.strip() or answer.strip().startswith(ABSTAIN_TEXT):
        abstained = True

    errors: list[str] = []
    if error:
        errors.append(error)

    sections = split_sections(text)
    list_of_tuples = (
        isinstance(raw, (list, tuple))
        and bool(raw)
        and isinstance(raw[0], (list, tuple))
    )
    first_score = None
    if list_of_tuples and len(raw[0]) > 1:
        try:
            first_score = float(raw[0][1])
        except (TypeError, ValueError):
            first_score = None
    # Ответы режимов помечены сентинелом 999.0; retrieval fallback — обычные
    # чанки с реальным score. Fallback показывается пользователю как выдача,
    # поэтому это ответ без цитаты, а не отказ.
    retrieval_fallback = (
        list_of_tuples
        and "ОТВЕТ" not in sections
        and first_score != 999.0
    )
    if retrieval_fallback:
        errors.append("retrieval_fallback_without_answer")
        abstained = False

    marker_ids = parse_marker_ids(answer)
    citations: list[dict] = []
    retrieved: list[dict] = []
    fragment_count = None

    if fragments:
        fragment_count = len(fragments)
        retrieved = [
            source_of(fragment)
            for fragment in fragments
            if has_page(source_of(fragment))
        ]
        for marker_id in marker_ids:
            if not 1 <= marker_id <= len(fragments):
                errors.append(f"marker_{marker_id}_out_of_range")
                continue
            source = source_of(fragments[marker_id - 1])
            if has_page(source):
                citations.append(source)
            else:
                errors.append(f"marker_{marker_id}_fragment_without_page")
    else:
        parsed_fragments = parse_source_lines(
            sections.get("НАЙДЕННЫЕ ФРАГМЕНТЫ", "")
        )
        parsed_sources = parse_source_lines(sections.get("ИСТОЧНИКИ", ""))
        if parsed_fragments:
            fragment_count = len(parsed_fragments)
            retrieved = parsed_fragments
            errors.append("fragments_parsed_from_text")
            for marker_id in marker_ids:
                if 1 <= marker_id <= len(parsed_fragments):
                    citations.append(parsed_fragments[marker_id - 1])
                else:
                    errors.append(f"marker_{marker_id}_out_of_range")
        elif parsed_sources:
            citations = parsed_sources
            retrieved = parsed_sources
            errors.append("fragments_parsed_from_sources")
        elif marker_ids:
            errors.append("markers_without_fragments")

    prediction = {
        "id": "",
        "answer": answer,
        "abstained": abstained,
        "citations": dedupe(citations),
        "retrieved": dedupe(retrieved),
        "fragment_count": fragment_count,
        "grounding_errors": sorted(set(errors)),
        "latency_ms": latency_ms,
    }
    raw_row = {
        "id": "",
        "raw": text,
        "captured_fragments": [source_of(fragment) for fragment in fragments],
        "captured_grounded_abstained": bool(
            getattr(grounded, "abstained", False)
        ),
        "captured_calls": spy.calls if spy else 0,
        "error": error,
    }
    return prediction, raw_row


def load_hipporag(index: str, embedding_model: str, llm_name: str):
    from hipporag import HippoRAG
    from hipporag.utils.config_utils import BaseConfig

    config = BaseConfig(
        save_dir=index,
        llm_name=llm_name,
        llm_base_url=LLM_BASE_URL,
        embedding_model_name=embedding_model,
        linking_top_k=100,
        retrieval_top_k=200,
        passage_node_weight=0.05,
        embedding_batch_size=8,
    )
    return HippoRAG(global_config=config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="M2 predictions для режимов 12/19/21"
    )
    parser.add_argument("--mode", required=True, choices=["12", "19", "21"])
    parser.add_argument("--golden", default="eval/grounding/golden_qa.jsonl")
    parser.add_argument("--index", default="outputs/astrology_test")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--llm-name", default=LLM_NAME)
    parser.add_argument("--out", required=True)
    parser.add_argument("--raw-out", required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    questions = read_jsonl(Path(args.golden))
    if args.limit:
        questions = questions[: args.limit]

    hipporag = load_hipporag(args.index, args.embedding_model, args.llm_name)
    predictions = []
    raw_rows = []
    try:
        for index, question in enumerate(questions, 1):
            print(
                f"[{args.mode}] {index}/{len(questions)} {question['id']}",
                flush=True,
            )
            prediction, raw_row = run_one(
                args.mode, hipporag, question["question"]
            )
            prediction["id"] = question["id"]
            raw_row["id"] = question["id"]
            predictions.append(prediction)
            raw_rows.append(raw_row)
    finally:
        close = getattr(hipporag, "close", None)
        if callable(close):
            close()

    write_jsonl(Path(args.out), predictions)
    write_jsonl(Path(args.raw_out), raw_rows)
    print(f"Готово: {len(predictions)} предсказаний -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
