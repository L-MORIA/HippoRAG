#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Оценка качества поиска на размеченном датасете (H1-1).

Что делает: берёт вопросы с ручной разметкой (``source_file``/``page_num``/
эталонный ответ), прогоняет выбранные режимы, сопоставляет выдачу с чанками
индекса и считает recall@5 (документы и страницы), MRR@10, nDCG@10, отказы,
пустые ответы, adequacy и p50/p95 latency.

Почему отдельный скрипт, а не ``eval/evaluate.py``: тот сравнивает DPR и
HippoRAG на ``expected_terms`` и не знает про страницы, режимы меню и
LLM-условия — для H1-1 («до/после» по режимам) он не годится.

Примеры:

    # проверка разметки против индекса, без прогона (быстро, без моделей)
    python scripts/eval_search_quality.py --validate-only

    # smoke: режим 2 на трёх первых вопросах (включая черновики)
    python scripts/eval_search_quality.py --modes 2 --limit 3 --include-draft

    # сравнение кандидатов на всех размеченных вопросах
    python scripts/eval_search_quality.py --modes 2,5,20

    # LLM-режимы без LM Studio: считается только механика, не качество
    python scripts/eval_search_quality.py --modes 12 --stub-llm --limit 3

Методика для LLM-режимов (план §9.2): фиксируются модель, base_url и
temperature из перехваченных запросов, считается число вызовов и время внутри
них, поэтому «время вне LLM» (``retrieval_ms``) отделено от генерации. Прогон
со ``--stub-llm`` помечается в отчёте как непригодный для сравнения качества.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Кириллица и «⚠» в cp1251-консоли: без этого таблица падает с UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from astro_app import evaluation as ev  # noqa: E402
from astro_app import config as astro_config  # noqa: E402
from astro_app.config import LLM_BASE_URL, LLM_NAME  # noqa: E402
from astro_app.modes import MODES  # noqa: E402

DEFAULT_INDEX = "outputs/astrology_test"
DEFAULT_DATASET = "eval/dataset/questions.jsonl"
DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

#: Режимы, где LLM используется только для переформулировки/фильтра до поиска.
#: Их retrieval-метрики осмысленны даже при заглушке, но помечаются как stub.
STUB_NOTES = (
    "stub-LLM: ответы модели подменены, retrieval-метрики HyDE/Self-RAG и "
    "качество ответов НЕ сравнимы с реальным прогоном"
)


# ── Перехват LLM-вызовов ─────────────────────────────────────────────────────


class _StubResponse:
    """Минимальный ответ OpenAI-совместимого API для прогона без LM Studio."""

    status_code = 200

    def __init__(self, content: str):
        self._content = content
        self._payload = {"choices": [{"message": {"content": content}}], "model": "stub"}

    def json(self):
        return self._payload

    @property
    def text(self) -> str:
        return json.dumps(self._payload, ensure_ascii=False)

    def raise_for_status(self):
        return None


class _StubModelsResponse(_StubResponse):
    """Ответ ``GET /models``: режимы сначала проверяют доступность модели."""

    def __init__(self):
        super().__init__("")

    def json(self):
        return {"data": [{"id": "stub", "object": "model"}], "model": "stub"}


class LlmRecorder:
    """Считает и (опционально) подменяет вызовы ``httpx.post`` к chat/completions.

    Режимы зовут LLM напрямую через ``httpx.post``, а не через
    ``llm_model.infer``, поэтому перехват делается на уровне модуля httpx.
    """

    def __init__(self, stub_text: str | None = None):
        self.stub_text = stub_text
        self.calls: list[dict] = []
        self._original = None
        self._original_get = None

    def install(self) -> None:
        if self._original is not None:
            return
        self._original = httpx.post
        original = self._original
        recorder = self

        def post(url, *args, **kwargs):
            if "/chat/completions" not in str(url):
                return original(url, *args, **kwargs)
            payload = kwargs.get("json") or {}
            started = time.perf_counter()
            error = None
            try:
                if recorder.stub_text is not None:
                    response = _StubResponse(recorder.stub_text)
                else:
                    response = original(url, *args, **kwargs)
            except Exception as exc:  # таймаут/сеть — важно записать, а не уронить
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                recorder.calls.append(
                    {
                        "url": str(url),
                        "model": payload.get("model"),
                        "temperature": payload.get("temperature"),
                        "messages": len(payload.get("messages") or []),
                        "stub": recorder.stub_text is not None,
                        "ms": round((time.perf_counter() - started) * 1000, 1),
                        "error": error,
                    }
                )
            return response

        httpx.post = post

        if self.stub_text is not None:
            # Режимы сначала проверяют доступность модели (GET /models) и без
            # 200 не зовут LLM — иначе --stub-llm не доходил бы до генерации.
            self._original_get = httpx.get
            original_get = self._original_get

            def get(url, *args, **kwargs):
                if "/models" in str(url):
                    return _StubModelsResponse()
                return original_get(url, *args, **kwargs)

            httpx.get = get

    def uninstall(self) -> None:
        if self._original is not None:
            httpx.post = self._original
            self._original = None
        if self._original_get is not None:
            httpx.get = self._original_get
            self._original_get = None

    def reset(self) -> None:
        self.calls = []

    @property
    def total_ms(self) -> float:
        return sum(call["ms"] for call in self.calls)

    def conditions(self) -> dict:
        models = sorted({str(c["model"]) for c in self.calls if c["model"]})
        temperatures = sorted(
            {float(c["temperature"]) for c in self.calls if c["temperature"] is not None}
        )
        return {
            "model": models,
            "temperature": temperatures,
            "calls": len(self.calls),
            "failures": sum(1 for c in self.calls if c["error"]),
        }


# ── Индекс ───────────────────────────────────────────────────────────────────


def working_dir_for(index: str, llm_name: str, embedding_model: str) -> Path:
    from hipporag.utils.config_utils import build_working_dir

    return Path(build_working_dir(index, llm_name, embedding_model))


def load_lookup(working_dir: Path) -> tuple[ev.ChunkLookup, dict]:
    """Lookup по parquet чанков: тот же текст, что видит HippoRAG в памяти."""
    import pandas as pd

    parquet = working_dir / "chunk_embeddings" / "vdb_chunk.parquet"
    if not parquet.exists():
        raise SystemExit(
            f"нет {parquet}: индекс не собран или другое имя модели "
            "(проверьте --index и --embedding-model)"
        )
    frame = pd.read_parquet(parquet, columns=["hash_id", "content", "metadata"])
    lookup = ev.build_lookup_from_records(frame.to_dict("records"))
    stats = {
        "chunks": len(lookup.chunks),
        "sources": sorted(lookup.sources()),
        "pages": {source: len(pages) for source, pages in lookup.pages_by_source().items()},
    }
    return lookup, stats


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


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=10
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def describe_index_status(status) -> dict:
    """``IndexStatus`` → JSON-совместимый dict (сам dataclass не сериализуется)."""
    if status is None:
        return {"state": "unknown"}
    return {
        "state": str(getattr(status, "state", "unknown")),
        "expected_version": getattr(status, "expected_version", None),
        "found_version": getattr(status, "found_version", None),
        "message": str(getattr(status, "message", "") or ""),
    }


# ── Режимы ───────────────────────────────────────────────────────────────────


def resolve_modes(spec: str) -> list[str]:
    known = list(MODES)
    spec = (spec or "").strip().lower()
    if spec in ("all", "*", "все"):
        return known
    if spec in ("retrieval", "поиск"):
        return [m for m in known if m not in ev.ANSWER_ONLY_MODES]
    if spec in ("answers", "ответы"):
        return [m for m in known if m in ev.ANSWER_ONLY_MODES]
    ids = [part.strip() for part in spec.split(",") if part.strip()]
    if not ids:
        raise SystemExit("--modes пуст")
    unknown = [m for m in ids if m not in MODES]
    if unknown:
        raise SystemExit(f"неизвестные режимы: {', '.join(unknown)} (есть: {', '.join(known)})")
    return ids


def extract_texts(raw) -> list[str]:
    """Достать тексты из любой формы выдачи режима.

    Формы: список 5-кортежей (text, score, ...), 6-кортежей (small→big),
    строк и (для режимов-ответов) одна строка/кортеж.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        return [str(raw.get("text") or raw.get("content") or "")]
    if isinstance(raw, (list, tuple)):
        texts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, (list, tuple)) and item:
                texts.append(item[0] if isinstance(item[0], str) else str(item[0]))
            elif isinstance(item, dict):
                texts.append(str(item.get("text") or item.get("content") or ""))
            else:
                texts.append(str(item))
        return texts
    return [str(raw)]


def call_mode(spec, hipporag, query: str, top_k: int | None):
    """Вызвать режим, передав ``top_k`` только если он его принимает."""
    if top_k and "top_k" in inspect.signature(spec.fn).parameters:
        raw = spec.fn(hipporag, query, top_k=top_k)
    else:
        raw = spec.fn(hipporag, query)
    if spec.extra and isinstance(raw, tuple) and len(raw) == 2:
        raw = raw[0]
    return raw


# ── Прогон ───────────────────────────────────────────────────────────────────


def run_question(
    question: ev.Question,
    mode_id: str,
    hipporag,
    lookup: ev.ChunkLookup,
    recorder: LlmRecorder,
    top_k: int | None,
    keep_output: bool = True,
) -> dict:
    spec = MODES[mode_id]
    recorder.reset()
    captured = io.StringIO()
    started = time.perf_counter()
    error = None
    raw = None
    try:
        with contextlib.redirect_stdout(captured):
            raw = call_mode(spec, hipporag, question.question, top_k)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    texts = extract_texts(raw)
    matched = [lookup.match(text) for text in texts]
    is_answer_mode = mode_id in ev.ANSWER_ONLY_MODES

    def _doc_hit(chunks):
        return any(
            source == question.source_file for chunk in chunks for source, _ in chunk.entries
        )

    def _page_hit(chunks):
        if question.page_num is None:
            return False
        return any(
            source == question.source_file and page == question.page_num
            for chunk in chunks
            for source, page in chunk.entries
        )

    answer_text = "\n".join(texts) if is_answer_mode else None

    return {
        "qid": question.qid,
        "query": question.question,
        "kind": question.kind,
        "subtype": question.subtype,
        "mode": mode_id,
        "retrieval": (not is_answer_mode) and not error,
        "page_expected": question.page_num is not None,
        "doc_rank": ev.first_hit_rank([_doc_hit(c) for c in matched]),
        "page_rank": ev.first_hit_rank([_page_hit(c) for c in matched]),
        "n_candidates": len(texts),
        "matched_chunks": [chunk.hash_id for chunks in matched for chunk in chunks][:10],
        "top1_source": matched[0][0].source_file if matched and matched[0] else "",
        "top1_page": matched[0][0].page_num if matched and matched[0] else None,
        "latency_ms": latency_ms,
        "llm_calls": len(recorder.calls),
        "llm_ms": round(recorder.total_ms, 1),
        "answer_text": (answer_text[:600] if answer_text else None),
        "adequacy": (
            ev.answer_adequacy(answer_text, question.expected_terms) if answer_text else None
        ),
        "error": error,
        "mode_output": captured.getvalue()[-400:] if keep_output and captured.getvalue() else "",
    }


def print_progress(row: dict) -> None:
    def mark(rank):
        return f"@{rank}" if rank else "—"

    status = (
        "ERROR" if row["error"] else f"doc {mark(row['doc_rank'])} page {mark(row['page_rank'])}"
    )
    print(
        f"  [{row['mode']:>2}] {row['qid']:<24} {status:<22} n={row['n_candidates']:<3} "
        f"{row['latency_ms']:>8.1f} мс",
        flush=True,
    )
    if row["error"]:
        print(f"       {row['error'][:160]}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Оценка качества поиска HippoRAG на размеченном датасете (H1-1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--index", default=DEFAULT_INDEX, help="save_dir индекса")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument(
        "--llm-name",
        default=None,
        help=(
            "id модели в API (LM Studio отдаёт его с префиксом, например "
            "google/gemma-4-e4b). Каталог индекса по-прежнему строится из "
            "LLM_NAME: индекс собирался под именем без префикса"
        ),
    )
    parser.add_argument("--modes", default="2", help="'2', '2,5,20', 'retrieval', 'answers', 'all'")
    parser.add_argument("--limit", type=int, default=0, help="первые N вопросов (0 — все)")
    parser.add_argument("--include-draft", action="store_true", help="включить черновики")
    parser.add_argument("--top-k", type=int, default=None, help="переопределить top_k режима")
    parser.add_argument("--k", type=int, default=5, help="k для recall")
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--no-index-check", action="store_true", help="при --validate-only не читать индекс"
    )
    parser.add_argument("--stub-llm", action="store_true", help="подменить ответы LLM")
    parser.add_argument("--stub-text", default="Нет данных.")
    parser.add_argument("--allow-unavailable-llm", action="store_true")
    parser.add_argument(
        "--warmup", action="store_true", help="прогреть режимы до замера (убрать cold start из p95)"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def llm_available(base_url: str, timeout: float = 3.0) -> bool:
    try:
        response = httpx.get(base_url.rstrip("/") + "/models", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def main(argv=None) -> int:
    args = parse_args(argv)
    started_at = datetime.now()

    dataset = ev.load_dataset(args.dataset)
    working = working_dir_for(args.index, LLM_NAME, args.embedding_model)
    # Каталог индекса строится из LLM_NAME (под ним индекс собирали), а в API
    # может уходить другой id: LM Studio отдаёт модели с префиксом издателя
    # («google/gemma-4-e4b»), и без префикса запрос падает с 400.
    llm_name = args.llm_name or LLM_NAME
    if args.llm_name:
        astro_config.LLM_NAME = args.llm_name
    lookup = None
    index_stats = {}
    if not (args.validate_only and args.no_index_check):
        lookup, index_stats = load_lookup(working)

    report = ev.validate_dataset(dataset, lookup)
    if not args.quiet:
        print("=" * 78)
        print(f"Датасет: {dataset.path}")
        print(f"Индекс:  {working}")
        print("-" * 78)
        print(report.format_text())
        print("=" * 78)

    if args.validate_only:
        if args.out:
            _write_json(args.out, {"validation": report.as_dict(), "index": index_stats})
        return 0 if report.ok else 1

    questions = dataset.for_run(include_draft=args.include_draft)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print(
            "Нет вопросов для прогона: размечено 0 (status=verified). "
            "Заполните eval/dataset/questions.jsonl или добавьте --include-draft.",
            file=sys.stderr,
        )
        return 2

    modes = resolve_modes(args.modes)
    available = llm_available(LLM_BASE_URL)
    needs_llm = [m for m in modes if m in ev.LLM_DEPENDENT_MODES]
    if needs_llm and not args.stub_llm:
        if not available and not args.allow_unavailable_llm:
            print(
                f"LM Studio недоступен ({LLM_BASE_URL}), а режимы {', '.join(needs_llm)} "
                "ходят в LLM (таймаут 300 с на вызов).\n"
                "Варианты: --stub-llm (механика без качества), --modes без LLM "
                "(2,5,6,20) или --allow-unavailable-llm, если LLM поднимется по ходу.",
                file=sys.stderr,
            )
            return 2

    hipporag = load_hipporag(args.index, args.embedding_model, LLM_NAME)
    index_status = describe_index_status(getattr(hipporag, "index_status", None))
    if not args.quiet:
        print(
            f"Индекс загружен: {len(lookup.chunks)} чанков, статус {index_status['state']}, "
            f"режимы {', '.join(modes)}"
        )
        if index_status["state"] not in ("ok", "unknown"):
            print(f"  ⚠ {index_status['message']}")

    recorder = LlmRecorder(stub_text=args.stub_text if args.stub_llm else None)
    recorder.install()
    rows_by_mode: dict[str, list[dict]] = {mode: [] for mode in modes}
    try:
        if args.warmup:
            # Первый вызов режима включает ленивую инициализацию (BM25-корпус,
            # PPR-структуры) и даёт p95 в секундах — это шум замера, а не поиск.
            for mode in modes:
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        call_mode(MODES[mode], hipporag, questions[0].question, args.top_k)
                    except Exception as exc:  # прогрев не должен ронять прогон
                        print(f"прогрев {mode}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if not args.quiet:
                    print(f"Прогрев режима {mode}: готово", flush=True)
        for number, question in enumerate(questions, 1):
            if not args.quiet:
                print(f"\n[{number}/{len(questions)}] {question.question}")
            for mode in modes:
                row = run_question(
                    question,
                    mode,
                    hipporag,
                    lookup,
                    recorder,
                    args.top_k,
                    keep_output=not args.quiet,
                )
                rows_by_mode[mode].append(row)
                if not args.quiet:
                    print_progress(row)
    finally:
        recorder.uninstall()

    summaries = [
        ev.summarize_mode(
            rows_by_mode[mode],
            mode_id=mode,
            label=MODES[mode].label,
            k=args.k,
            mrr_k=args.mrr_k,
            ndcg_k=args.ndcg_k,
        )
        for mode in modes
    ]
    table = ev.format_markdown_table(summaries)

    print()
    print(table)
    if args.stub_llm:
        print(f"\n⚠ {STUB_NOTES}")

    payload = {
        "meta": {
            "generated_at": started_at.isoformat(timespec="seconds"),
            "duration_s": round((datetime.now() - started_at).total_seconds(), 1),
            "git_commit": git_commit(),
            "dataset": {
                "path": str(dataset.path),
                "sha256": dataset.sha256(),
                "n_questions": len(questions),
                "n_incomplete": len(dataset.incomplete),
                "include_draft": args.include_draft,
            },
            "index": {
                "path": args.index,
                "working_dir": str(working),
                "status": index_status,
                "embedding_model": args.embedding_model,
                **index_stats,
            },
            "llm": {
                "name": llm_name,
                "base_url": LLM_BASE_URL,
                "available": available,
                "stubbed": args.stub_llm,
                "stub_text": args.stub_text if args.stub_llm else None,
                **recorder.conditions(),
            },
            "params": {
                "modes": modes,
                "k": args.k,
                "mrr_k": args.mrr_k,
                "ndcg_k": args.ndcg_k,
                "top_k": args.top_k,
                "limit": args.limit,
            },
        },
        "validation": report.as_dict(),
        "summaries": summaries,
        "rows": [row for mode in modes for row in rows_by_mode[mode]],
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path("outputs/reports")
        / (f"eval_{working.name}_{started_at.strftime('%Y%m%d_%H%M%S')}.json")
    )
    _write_json(out_path, payload)
    print(f"\nОтчёт: {out_path}")

    if args.markdown_out:
        markdown = Path(args.markdown_out)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(table + "\n", encoding="utf-8")
        print(f"Таблица: {markdown}")

    failed = sum(1 for rows in rows_by_mode.values() for row in rows if row["error"])
    return 0 if failed < sum(len(rows) for rows in rows_by_mode.values()) else 1


def _write_json(path, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    sys.exit(main())
