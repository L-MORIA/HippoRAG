#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Собрать урезанный индекс-фикстуру для CI (H1-5).

Фикстура — снимок части индекса (чанки + метаданные + манифест с sha256),
по которому в CI можно гонять **лексический** retrieval (BM25) без
эмбеддинг-модели, без LLM и без сети. Это воспроизводимая часть H1-5:
полный retrieval с эмбеддингами в CI невозможен (индекса в репозитории нет).

Запускать нужно владельцу, у которого есть собранный индекс::

    python scripts/build_index_fixture.py \
        --index outputs/astrology_test \
        --out fixtures/astrology_mini \
        --max-chunks 300

Что попадает в фикстуру:

* все чанки файлов, на которые ссылаются размеченные вопросы датасета
  (иначе смоук не нашёл бы ожидаемые документы);
* затем — остальные чанки, пока не наберётся ``--max-chunks``.

**Внимание:** фикстура содержит дословный текст корпуса. Публикуя её как
CI-артефакт, вы публикуете этот текст. Если это нежелательно — не коммитьте
фикстуру в репозиторий, а артефактом выкладывайте только отчёт смоука.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from astro_app import evaluation as ev  # noqa: E402
from astro_app import index_fixture as fix  # noqa: E402

DEFAULT_INDEX = "outputs/astrology_test"
DEFAULT_DATASET = "eval/dataset/questions.jsonl"
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_LLM = "google/gemma-4-e4b"
DEFAULT_OUT = "fixtures/astrology_mini"


def resolve_working_dir(args: argparse.Namespace) -> Path:
    """Каталог с чанками индекса (тот же, что использует измеритель H1-1)."""
    if args.working_dir:
        return Path(args.working_dir)
    from hipporag.utils.config_utils import build_working_dir

    return Path(build_working_dir(args.index, args.llm_name, args.embedding_model))


def dataset_sources(dataset_path: str) -> List[str]:
    """Файлы, на которые ссылаются размеченные вопросы."""
    try:
        dataset = ev.load_dataset(dataset_path)
    except ev.DatasetError as exc:
        print(f"Датасет недоступен ({exc}) — беру первые чанки подряд", file=sys.stderr)
        return []
    sources = []
    for question in dataset.for_run(include_draft=False):
        if question.source_file and question.source_file not in sources:
            sources.append(question.source_file)
    return sources


def select_chunks(
    records: Sequence[Dict[str, Any]],
    *,
    preferred_sources: Sequence[str],
    max_chunks: int,
    only_sources: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Отобрать чанки: сначала нужные источники, затем остальные."""
    preferred = set(preferred_sources)
    only = set(only_sources) if only_sources else None

    def _sources_of(record: Dict[str, Any]) -> set:
        return {source for source, _ in fix.chunk_entries(record)}

    selected: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    for record in records:
        sources = _sources_of(record)
        if only and not (sources & only):
            continue
        if sources & preferred:
            selected.append(record)
        else:
            rest.append(record)
    selected.extend(rest)
    return selected[:max_chunks]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка урезанного индекса-фикстура (H1-5)")
    parser.add_argument("--index", default=DEFAULT_INDEX, help="save_dir индекса")
    parser.add_argument(
        "--working-dir", default=None, help="каталог индекса напрямую (в обход build_working_dir)"
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--llm-name", default=DEFAULT_LLM)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", default=DEFAULT_OUT, help="каталог фикстуры")
    parser.add_argument("--max-chunks", type=int, default=300)
    parser.add_argument(
        "--sources", default=None, help="через запятую: оставить только эти источники"
    )
    args = parser.parse_args(argv)

    working = resolve_working_dir(args)
    parquet = working / "chunk_embeddings" / "vdb_chunk.parquet"
    if not parquet.exists():
        print(
            f"нет {parquet}: индекс не собран или другое имя модели "
            "(проверьте --index и --embedding-model)",
            file=sys.stderr,
        )
        return 2

    import pandas as pd

    frame = pd.read_parquet(parquet, columns=["hash_id", "content", "metadata"])
    records = frame.to_dict("records")
    print(f"Индекс: {working} — чанков {len(records)}")

    preferred = dataset_sources(args.dataset)
    print(f"Источники из датасета ({len(preferred)}): {', '.join(preferred) or '—'}")

    only_sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    chunks = select_chunks(
        records,
        preferred_sources=preferred,
        max_chunks=args.max_chunks,
        only_sources=only_sources,
    )
    if not chunks:
        print("Не выбрано ни одного чанка", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    manifest = fix.save_fixture(
        chunks,
        out_dir,
        meta={
            "source_index": str(args.index),
            "working_dir": str(working),
            "embedding_model": args.embedding_model,
            "llm_name": args.llm_name,
            "n_chunks_total": len(records),
            "n_chunks_kept": len(chunks),
            "dataset_sources": preferred,
            "note": (
                "урезанный снимок индекса для CI (H1-5): лексический retrieval (BM25) "
                "без эмбеддинг-модели и без LLM; содержит дословный текст корпуса"
            ),
        },
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    print(f"Фикстура: {out_dir} — чанков {len(chunks)} (из {len(records)})")
    print(f"Манифест: {manifest['files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
