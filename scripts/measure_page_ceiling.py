# -*- coding: utf-8 -*-
"""Замер потолка page recall для H1-6: страница как единица поиска.

Зачем: на golden set page recall@5 у чанкового поиска 56,5 %, а порог H1-6 —
80 %. Прежде чем переделывать продукт, надо понять, помогает ли сама смена
единицы поиска: страница целиком (BM25 по тексту страницы, dense по векторам
чанков, RRF и RU cross-encoder) даёт ≥ 80 % или нет. Если нет — дело не в
единице поиска, а в чанкинге, OCR, эмбеддингах или разметке.

Скрипт ничего не меняет в индексе и не вызывает LLM: читает
``chunk_embeddings/vdb_chunk.parquet``, склеивает чанки в страницы и считает
recall@k по паре «документ + страница».

Запуск (сначала bounded-проба)::

    python scripts/measure_page_ceiling.py --limit 3
    python scripts/measure_page_ceiling.py            # все answerable вопросы

Ретриверы:
* ``bm25``            — BM25 по тексту страницы;
* ``bm25_ce``         — тот же BM25, затем RU-CE по тексту страницы;
* ``bm25_ce_chunk``   — RU-CE по чанкам страницы, страница = максимум чанков;
* ``dense_mean``      — косинус вопроса и среднего вектора чанков страницы;
* ``dense_max``       — косинус вопроса и лучшего чанка страницы;
* ``rrf_ce``          — RRF(BM25, dense_max), затем RU-CE по тексту страницы.

``--no-expansion`` отключает расширение запроса синонимами: на этом корпусе
слово «планета» тянет в запрос список всех планет и уводит BM25 на таблицы.

Метрики: recall@5/10/20/50 и MRR@10 по страницам. Порог H1-6 — recall@5 ≥ 80 %.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import eval_grounding as eg  # noqa: E402

DEFAULT_SAVE_DIR = ROOT / "outputs" / "astrology_test"
DEFAULT_LLM_NAME = "gemma-4-e4b"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_CE_POOL = 100
KS = (5, 10, 20, 50)
RRF_K = 60
CE_PAGE_CHARS = 2000
CE_CHUNK_CHARS = 512

PageKey = Tuple[str, int]


@dataclass
class Page:
    """Страница книги как единица поиска."""

    doc_title: str
    page_num: int
    text: str
    vectors: List[np.ndarray] = field(default_factory=list)
    chunk_texts: List[str] = field(default_factory=list)

    @property
    def chunks(self) -> int:
        return len(self.chunk_texts)

    @property
    def vector(self) -> np.ndarray:
        """Средний вектор чанков страницы (нормированный)."""
        mean = np.vstack(self.vectors).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm else mean


def discover_index_dir(save_dir: Path, llm_name: str, embedding_model: str) -> Path:
    """Каталог индекса: ожидаемое имя, иначе первый с chunk_embeddings."""
    expected = save_dir / f"{llm_name}_{embedding_model.replace('/', '_')}"
    if (expected / "chunk_embeddings" / "vdb_chunk.parquet").exists():
        return expected
    found = sorted(p.parent.parent for p in save_dir.glob("*/chunk_embeddings/vdb_chunk.parquet"))
    if not found:
        raise FileNotFoundError(
            f"в {save_dir} нет ни одного индекса с chunk_embeddings/vdb_chunk.parquet"
        )
    if len(found) > 1:
        print(
            f"  индекс не указан явно, беру первый из {len(found)}: {found[0].name}",
            file=sys.stderr,
        )
    return found[0]


def load_pages(index_dir: Path) -> List[Page]:
    """Склеивает чанки в страницы: текст по порядку, векторы — по чанкам."""
    import pandas as pd

    frame = pd.read_parquet(index_dir / "chunk_embeddings" / "vdb_chunk.parquet")
    grouped: Dict[PageKey, Page] = {}
    for content, metadata, embedding in zip(
        frame["content"], frame["metadata"], frame["embedding"]
    ):
        entries = metadata if isinstance(metadata, list) else json.loads(metadata)
        if not entries:
            continue
        primary = entries[0]
        page_num = int(primary.get("page_num") or 0)
        if page_num < 1:
            continue
        doc_title = str(primary.get("doc_title") or primary.get("source_file") or "")
        key = eg.source_key(eg.SourceRef(doc_title=doc_title, page_num=page_num))
        page = grouped.get(key)
        if page is None:
            page = Page(doc_title=doc_title, page_num=page_num, text="")
            grouped[key] = page
        page.chunk_texts.append(str(content))
        page.vectors.append(np.asarray(embedding, dtype=np.float32))

    pages = list(grouped.values())
    for page in pages:
        page.text = " ".join(page.chunk_texts)
    pages.sort(key=lambda page: (page.doc_title, page.page_num))
    return pages


def page_keys(pages: Sequence[Page]) -> List[PageKey]:
    """Ключ сравнения в том же виде, что и у golden set."""
    return [
        eg.source_key(eg.SourceRef(doc_title=page.doc_title, page_num=page.page_num))
        for page in pages
    ]


def query_tokens(question: str, expand: bool) -> List[str]:
    """Токены вопроса; при ``expand`` — плюс синонимы (поведение режима 5)."""
    from astro_app.textutils import _tokenize

    tokens = _tokenize(question)
    if not expand:
        return tokens
    from astro_app.query import _expand_query, split_query_terms

    terms, q_lower = split_query_terms(question)
    if not terms:
        terms = [word for word in question.lower().split() if len(word) > 2]
    return list(dict.fromkeys(tokens + _expand_query(q_lower, terms)))


def bm25_ranking(question: str, bm25, expand: bool) -> List[int]:
    scores = bm25.get_scores(query_tokens(question, expand))
    return list(np.argsort(scores)[::-1])


def dense_ranking(question: str, encoder, matrix: np.ndarray, prefix: str = "") -> List[int]:
    query = encoder.encode([prefix + question], show_progress_bar=False, normalize_embeddings=True)
    scores = matrix @ np.asarray(query, dtype=np.float32).reshape(-1)
    return list(np.argsort(scores)[::-1])


def rrf_ranking(*rankings: Sequence[int], k: int = RRF_K) -> List[int]:
    """Reciprocal rank fusion: сумма 1/(k + место) по всем спискам."""
    scores: Dict[int, float] = {}
    for ranking in rankings:
        for position, index in enumerate(ranking, start=1):
            scores[int(index)] = scores.get(int(index), 0.0) + 1.0 / (k + position)
    return [index for index, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _crossencoder():
    """RU cross-encoder режима 5 (или None, если модель недоступна)."""
    try:
        from astro_app.rerankers import _get_crossencoder_ru
    except Exception as exc:  # pragma: no cover - зависит от окружения
        print(f"  RU-CE недоступен: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    try:
        return _get_crossencoder_ru()
    except Exception as exc:  # pragma: no cover - зависит от окружения
        print(f"  RU-CE недоступен: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def _ce_scores(ce, question: str, texts: Sequence[str]) -> Optional[np.ndarray]:
    if ce is None or not texts:
        return None
    try:
        values = ce.predict([(question, text) for text in texts], show_progress_bar=False)
    except Exception as exc:  # pragma: no cover - зависит от окружения
        print(f"  RU-CE упал: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    return np.asarray(values, dtype=np.float32)


def ce_page_ranking(
    question: str, ranking: Sequence[int], pages: Sequence[Page], pool: int, ce
) -> Optional[List[int]]:
    """RU-CE по тексту страницы для первых ``pool`` кандидатов."""
    candidates = list(ranking[:pool])
    if ce is None or not candidates:
        return None
    scores = _ce_scores(ce, question, [pages[index].text[:CE_PAGE_CHARS] for index in candidates])
    if scores is None:
        return None
    order = np.argsort(scores)[::-1]
    return [candidates[int(position)] for position in order] + [index for index in ranking[pool:]]


def ce_chunk_ranking(
    question: str, ranking: Sequence[int], pages: Sequence[Page], pool: int, ce
) -> Optional[List[int]]:
    """RU-CE по чанкам: страница получает максимум по своим чанкам."""
    candidates = list(ranking[:pool])
    if ce is None or not candidates:
        return None
    flat: List[str] = []
    owner: List[int] = []
    for index in candidates:
        for text in pages[index].chunk_texts:
            flat.append(text[:CE_CHUNK_CHARS])
            owner.append(index)
    scores = _ce_scores(ce, question, flat)
    if scores is None:
        return None
    best: Dict[int, float] = {}
    for position, index in enumerate(owner):
        value = float(scores[position])
        if index not in best or value > best[index]:
            best[index] = value
    order = sorted(candidates, key=lambda index: best[index], reverse=True)
    return order + [index for index in ranking[pool:]]


def rank_of_gold(ranking: Sequence[int], keys: Sequence[PageKey], gold: set) -> Optional[int]:
    for position, index in enumerate(ranking, start=1):
        if keys[index] in gold:
            return position
    return None


def evaluate(
    questions: Sequence[eg.GoldenQuestion],
    pages: Sequence[Page],
    bm25,
    encoder,
    matrix_mean: np.ndarray,
    matrix_chunks: Tuple[np.ndarray, np.ndarray],
    ce,
    ce_pool: int,
    expand: bool,
    query_prefix: str = "",
) -> Dict[str, object]:
    keys = page_keys(pages)
    chunk_matrix, chunk_owner = matrix_chunks
    names = ["bm25", "bm25_ce", "bm25_ce_chunk", "dense_mean", "dense_max", "rrf", "rrf_ce"]
    ranks: Dict[str, List[Optional[int]]] = {name: [] for name in names}
    rows: List[Dict[str, object]] = []

    for position, question in enumerate(questions, start=1):
        gold = {eg.source_key(source) for source in question.gold_sources}
        bm25_order = bm25_ranking(question.question, bm25, expand)
        query = encoder.encode(
            [query_prefix + question.question],
            show_progress_bar=False,
            normalize_embeddings=True,
        ).reshape(-1)
        mean_scores = matrix_mean @ query
        chunk_scores = chunk_matrix @ query
        dense_mean_order = list(np.argsort(mean_scores)[::-1])
        dense_max_order = max_pool(chunk_scores, chunk_owner, len(pages))
        rrf_order = rrf_ranking(bm25_order, dense_max_order)

        orders: Dict[str, Sequence[int]] = {
            "bm25": bm25_order,
            "bm25_ce": ce_page_ranking(question.question, bm25_order, pages, ce_pool, ce)
            or bm25_order,
            "bm25_ce_chunk": ce_chunk_ranking(question.question, bm25_order, pages, ce_pool, ce)
            or bm25_order,
            "dense_mean": dense_mean_order,
            "dense_max": dense_max_order,
            "rrf": rrf_order,
            "rrf_ce": ce_page_ranking(question.question, rrf_order, pages, ce_pool, ce)
            or rrf_order,
        }
        row: Dict[str, object] = {
            "id": question.id,
            "gold": sorted(f"{key[0]}:{key[1]}" for key in gold),
        }
        for name, order in orders.items():
            rank = rank_of_gold(order, keys, gold)
            ranks[name].append(rank)
            row[name] = rank
        rows.append(row)
        print(f"  [{position}/{len(questions)}] {question.id}", flush=True)

    summary: Dict[str, Dict[str, object]] = {}
    for name in names:
        name_ranks = ranks[name]
        summary[name] = {
            f"recall@{k}": round(
                sum(1 for rank in name_ranks if rank and rank <= k) / len(name_ranks), 4
            )
            for k in KS
        }
        summary[name]["mrr@10"] = round(
            sum(1.0 / rank for rank in name_ranks if rank and 1 <= rank <= 10) / len(name_ranks), 4
        )
    return {"summary": summary, "rows": rows, "ranks": ranks}


def max_pool(chunk_scores: np.ndarray, chunk_owner: np.ndarray, pages: int) -> List[int]:
    """Страница = максимум по её чанкам (dense)."""
    best = np.full(pages, -np.inf, dtype=np.float32)
    np.maximum.at(best, chunk_owner, chunk_scores)
    return list(np.argsort(best)[::-1])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Замер потолка page recall (H1-6)")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="каталог с индексами")
    parser.add_argument("--index-dir", help="конкретный каталог индекса (иначе авто)")
    parser.add_argument("--golden", default=str(eg.DEFAULT_GOLDEN), help="разметка golden set")
    parser.add_argument("--limit", type=int, help="первые N answerable вопросов (bounded-проба)")
    parser.add_argument("--ids", help="только эти id через запятую (диагностика)")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--llm-name", default=DEFAULT_LLM_NAME, help="префикс каталога индекса")
    parser.add_argument(
        "--ce-pool", type=int, default=DEFAULT_CE_POOL, help="0 — без cross-encoder"
    )
    parser.add_argument(
        "--no-expansion", action="store_true", help="не расширять запрос синонимами"
    )
    parser.add_argument(
        "--dense-model",
        help=(
            "модель эмбеддингов для страниц и запроса (иначе — та, под которой собран "
            "индекс); для e5-семейства префиксы query:/passage: добавляются сами"
        ),
    )
    parser.add_argument("--out", default=str(ROOT / "outputs" / "reports" / "page_ceiling.json"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    index_dir = (
        Path(args.index_dir)
        if args.index_dir
        else discover_index_dir(save_dir, args.llm_name, args.embedding_model)
    )
    print(f"индекс: {index_dir}")

    golden = eg.load_golden(args.golden)
    questions = list(golden.answerable)
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        questions = [item for item in questions if item.id in wanted]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit должен быть >= 1")
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("нет вопросов после фильтра --ids/--limit")
    print(f"вопросов: {len(questions)} (answerable из golden set)")

    pages = load_pages(index_dir)
    print(
        f"страниц: {len(pages)}, чанков на страницу: "
        f"{min(page.chunks for page in pages)}–{max(page.chunks for page in pages)}"
    )

    from rank_bm25 import BM25Okapi

    from astro_app.textutils import _tokenize

    bm25 = BM25Okapi([_tokenize(page.text) for page in pages])

    from sentence_transformers import SentenceTransformer

    dense_model = args.dense_model or args.embedding_model
    encoder = SentenceTransformer(dense_model)
    use_index_vectors = dense_model == args.embedding_model
    if use_index_vectors:
        matrix_mean = np.vstack([page.vector for page in pages]).astype(np.float32)
        chunk_vectors: List[np.ndarray] = []
        chunk_owner: List[int] = []
        for index, page in enumerate(pages):
            for vector in page.vectors:
                norm = float(np.linalg.norm(vector))
                chunk_vectors.append(vector / norm if norm else vector)
                chunk_owner.append(index)
        matrix_chunks = (np.vstack(chunk_vectors).astype(np.float32), np.asarray(chunk_owner))
    else:
        print(f"dense-модель: {dense_model} (пересчитываю векторы чанков)")
        texts: List[str] = []
        chunk_owner = []
        for index, page in enumerate(pages):
            for text in page.chunk_texts:
                texts.append(text)
                chunk_owner.append(index)
        prefix = "passage: " if "e5" in dense_model.lower() else ""
        encoded = encoder.encode(
            [prefix + text for text in texts],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        matrix_chunks = (np.asarray(encoded, dtype=np.float32), np.asarray(chunk_owner))
        sums = np.zeros((len(pages), matrix_chunks[0].shape[1]), dtype=np.float32)
        np.add.at(sums, matrix_chunks[1], matrix_chunks[0])
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        matrix_mean = np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 0)

    ce = _crossencoder() if args.ce_pool else None
    query_prefix = "query: " if "e5" in dense_model.lower() else ""
    result = evaluate(
        questions,
        pages,
        bm25,
        encoder,
        matrix_mean,
        matrix_chunks,
        ce,
        args.ce_pool,
        expand=not args.no_expansion,
        query_prefix=query_prefix,
    )

    summary = result["summary"]
    print()
    print(f"{'ретривер':14} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'R@50':>7} {'MRR@10':>7}")
    for name in (
        "bm25",
        "bm25_ce",
        "bm25_ce_chunk",
        "dense_mean",
        "dense_max",
        "rrf",
        "rrf_ce",
    ):
        row = summary[name]
        print(
            f"{name:14} "
            f"{row['recall@5'] * 100:6.1f}% {row['recall@10'] * 100:6.1f}% "
            f"{row['recall@20'] * 100:6.1f}% {row['recall@50'] * 100:6.1f}% "
            f"{row['mrr@10']:7.3f}"
        )
    best = max(summary, key=lambda name: summary[name]["recall@5"])
    print(f"лучший по R@5: {best} — {summary[best]['recall@5'] * 100:.1f}% (порог H1-6: 80 %)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_dir": str(index_dir),
        "golden": str(args.golden),
        "questions": len(questions),
        "pages": len(pages),
        "ce_pool": args.ce_pool,
        "expansion": not args.no_expansion,
        "dense_model": dense_model,
        "summary": summary,
        "rows": result["rows"],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
