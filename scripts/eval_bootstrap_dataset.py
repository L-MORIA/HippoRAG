#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Черновик датасета для eval: по одному кандидату на страницу документа.

Скрипт не придумывает вопросы и ответы — он только показывает, что лежит в
индексе, чтобы разметка шла по реальным страницам, а не по памяти. Выходной
файл — заготовки с пустыми ``question``/``reference_answer``: их заполняет
владелец/предметник, после чего записи переносятся в
``eval/dataset/questions.jsonl`` со ``status: "verified"``.

Примеры:

    # по 4 страницы из каждого документа, равномерно по книге
    python scripts/eval_bootstrap_dataset.py --per-source 4

    # конкретные страницы (после просмотра оглавления)
    python scripts/eval_bootstrap_dataset.py --pages "Астрофэншуй квартиры.pdf:7,171"

Заготовки не проходят валидацию и в метрики не попадают: у них пустые вопрос
и эталонный ответ. Это защита от «прогон прошёл, а размечено 3 вопроса».
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from astro_app.config import LLM_NAME  # noqa: E402
from astro_app.evaluation import normalize_text  # noqa: E402

DEFAULT_INDEX = "outputs/astrology_test"
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_OUT = "eval/dataset/candidates.jsonl"


def slug(text: str) -> str:
    """Читаемый qid-префикс: «Астрофэншуй квартиры.pdf» → «астрофэншуй-квартиры»."""
    name = text.rsplit(".", 1)[0].lower()
    name = re.sub(r"[^\w]+", "-", name, flags=re.UNICODE).strip("-")
    return name or "doc"


def parse_pages(spec: str) -> dict[str, set[int]]:
    """``"файл.pdf:7,171"`` → ``{"файл.pdf": {7, 171}}``."""
    wanted: dict[str, set[int]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"--pages: ожидалось 'файл:страницы', получено '{item}'")
        source, pages = item.rsplit(":", 1)
        try:
            numbers = {int(p) for p in pages.replace(" ", "").split(",") if p}
        except ValueError as exc:
            raise SystemExit(f"--pages: не числа в '{pages}'") from exc
        wanted.setdefault(source.strip(), set()).update(numbers)
    return wanted


def spread(pages: list[int], count: int) -> list[int]:
    """Равномерно выбрать ``count`` страниц: середина книги, а не титул."""
    if count <= 0 or not pages:
        return []
    if len(pages) <= count:
        return list(pages)
    last = len(pages) - 1
    positions = {round((i + 0.5) / count * last) for i in range(count)}
    return [pages[position] for position in sorted(positions)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Заготовки вопросов по страницам индекса")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--pages", default="", help="'файл.pdf:7,171;другой.pdf:5'")
    parser.add_argument("--min-chars", type=int, default=200, help="минимальная длина чанка")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    from hipporag.utils.config_utils import build_working_dir

    working = Path(build_working_dir(args.index, LLM_NAME, args.embedding_model))
    parquet = working / "chunk_embeddings" / "vdb_chunk.parquet"
    if not parquet.exists():
        raise SystemExit(f"нет {parquet}")

    frame = pd.read_parquet(parquet, columns=["content", "metadata"])
    wanted = parse_pages(args.pages) if args.pages else {}

    # (source, page) → самый длинный чанк страницы: в нём больше фактов,
    # по нему проще придумать вопрос и найти эталонный ответ.
    best: dict[tuple[str, int], str] = {}
    for content, meta in zip(frame["content"], frame["metadata"]):
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                continue
        if isinstance(meta, list):
            meta = meta[0] if meta else {}
        if not isinstance(meta, dict):
            continue
        source = str(meta.get("source_file") or "").strip()
        try:
            page = int(meta.get("page_num"))
        except (TypeError, ValueError):
            continue
        if not source or page < 1 or len(content) < args.min_chars:
            continue
        key = (source, page)
        if len(content) > len(best.get(key, "")):
            best[key] = content

    by_source: dict[str, list[int]] = {}
    for source, page in best:
        by_source.setdefault(source, []).append(page)

    selected: list[tuple[str, int]] = []
    for source in sorted(by_source):
        pages = sorted(by_source[source])
        if wanted:
            chosen = [page for page in pages if page in wanted.get(source, set())]
        else:
            chosen = spread(pages, args.per_source)
        selected.extend((source, page) for page in chosen)

    if not selected:
        raise SystemExit("не выбрано ни одной страницы: проверьте --pages и индекс")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for source, page in selected:
            entry = {
                "qid": f"{slug(source)}-p{page}",
                "question": "",
                "kind": "domain",
                "subtype": "other",
                "source_file": source,
                "page_num": page,
                "reference_answer": "",
                "expected_terms": [],
                "status": "draft",
                "notes": normalize_text(best[(source, page)])[:300],
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"{source} стр.{page}: {normalize_text(best[(source, page)])[:110]}…")

    print(f"\nЗаготовок: {len(selected)} → {out_path}")
    print("Заполните question/reference_answer/expected_terms и перенесите готовые")
    print('записи в eval/dataset/questions.jsonl со status="verified".')
    return 0


if __name__ == "__main__":
    sys.exit(main())
