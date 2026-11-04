#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Реальный прогон golden set через HippoRAG.rag_qa (H2-6).

Скрипт — адаптер между настоящим QA-путём HippoRAG и метриками
``scripts/eval_grounding.py``. Он не копирует логику ответа: каждый вопрос
уходит в ``HippoRAG.rag_qa([question])``, а результат раскладывается в
``Prediction`` ровно так, как его ожидает оценка.

Запуск на машине с индексом и LM Studio::

    python scripts/grounding_retriever.py \
        --save-dir outputs/astrology_test \
        --llm-name gemma-4-e4b \
        --llm-base-url http://localhost:1234/v1

Сравнение ``qa_top_k``::

    python scripts/grounding_retriever.py --qa-top-k 5 \
        --out-json eval/grounding/report_top5.json \
        --out-md eval/grounding/report_top5.md \
        --out-predictions eval/grounding/predictions_top5.jsonl
    python scripts/grounding_retriever.py --qa-top-k 8 \
        --out-json eval/grounding/report_top8.json \
        --out-md eval/grounding/report_top8.md \
        --out-predictions eval/grounding/predictions_top8.jsonl

Ошибка на одном вопросе не роняет весь прогон: она попадает в
``grounding_errors`` как ``runtime_error:<Type>``. Флаг ``--fail-fast``
возвращает прежнее поведение — остановиться на первой ошибке.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT / "src"))

import eval_grounding as eg  # noqa: E402

DEFAULT_QA_TOP_K = 5
DEFAULT_PREDICTIONS = ROOT / "eval" / "grounding" / "predictions_2026-11-04.jsonl"


def source_from_citation(citation: Any) -> Optional[eg.SourceRef]:
    """Превращает Citation в SourceRef; невалидные страницы пропускает.

    Страница 0/None означает «источник не определён». Такая цитата не может
    совпасть с размеченным ``gold_sources`` и только занижает precision,
    поэтому в метрики она не попадает.
    """
    page = getattr(citation, "page_num", None)
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        return None
    title = getattr(citation, "doc_title", "") or getattr(citation, "source_file", "")
    if not isinstance(title, str) or not title.strip():
        return None
    return eg.SourceRef(doc_title=title.strip(), page_num=page)


def _sources(citations: Sequence[Any], limit: Optional[int] = None) -> List[eg.SourceRef]:
    if limit is not None:
        citations = citations[:limit]
    result: List[eg.SourceRef] = []
    for citation in citations:
        source = source_from_citation(citation)
        if source is not None:
            result.append(source)
    return result


def prediction_from_solution(
    question_id: str,
    solution: Any,
    *,
    qa_top_k: int,
) -> eg.Prediction:
    """Раскладывает QuerySolution в Prediction для eval_grounding.

    ``fragment_count`` — столько фрагментов реально попало в промпт:
    ``min(len(docs), qa_top_k)``. ``retrieved`` — первые fragment_count
    цитат, ``citations`` — только те, на которые модель сослалась.
    """
    docs = list(getattr(solution, "docs", None) or [])
    citations = list(getattr(solution, "citations", None) or [])
    used = list(getattr(solution, "used_citations", None) or [])
    fragment_count = min(len(docs), qa_top_k)

    return eg.Prediction(
        id=question_id,
        answer=str(getattr(solution, "answer", "") or ""),
        abstained=bool(getattr(solution, "abstained", False)),
        citations=_sources(used),
        fragment_count=fragment_count,
        errors=list(getattr(solution, "grounding_errors", None) or []),
        retrieved=_sources(citations, limit=fragment_count),
    )


def runtime_error_prediction(question_id: str, exc: Exception) -> eg.Prediction:
    """Предсказание для упавшего вопроса: видно в отчёте, прогон идёт дальше."""
    return eg.Prediction(
        id=question_id,
        answer="",
        abstained=False,
        citations=[],
        fragment_count=None,
        errors=[f"runtime_error:{type(exc).__name__}"],
        retrieved=[],
    )


def resolve_qa_top_k(engine: Any, override: Optional[int] = None) -> int:
    """Берёт qa_top_k из конфига движка, если он не задан флагом."""
    if override is not None:
        return override
    config = getattr(engine, "global_config", None)
    value = getattr(config, "qa_top_k", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    return DEFAULT_QA_TOP_K


def _extract_solutions(result: Any) -> Sequence[Any]:
    """rag_qa возвращает (solutions, messages, metadata) или список решений."""
    solutions = result[0] if isinstance(result, tuple) else result
    if not solutions:
        raise RuntimeError("rag_qa вернул пустой список решений")
    return solutions


def run_questions(
    golden: eg.GoldenSet,
    engine: Any,
    *,
    qa_top_k: int,
    fail_fast: bool = False,
    on_prediction: Optional[Callable[[eg.Prediction], None]] = None,
) -> List[eg.Prediction]:
    """Прогоняет вопросы по одному через реальный ``rag_qa``."""
    predictions: List[eg.Prediction] = []
    total = len(golden.items)

    for index, question in enumerate(golden.items, 1):
        print(f"[{index}/{total}] {question.id}", flush=True)
        started = time.perf_counter()
        try:
            result = engine.rag_qa([question.question])
            solution = _extract_solutions(result)[0]
            prediction = prediction_from_solution(question.id, solution, qa_top_k=qa_top_k)
        except Exception as exc:  # noqa: BLE001 — ошибка вопроса не должна ронять прогон
            if fail_fast:
                raise
            print(
                f"  ошибка: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            prediction = runtime_error_prediction(question.id, exc)

        prediction.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        predictions.append(prediction)
        if on_prediction is not None:
            on_prediction(prediction)

    return predictions


def load_engine(args: argparse.Namespace) -> Any:
    """Создаёт HippoRAG с параметрами запуска (импорт тяжёлых зависимостей ленивый)."""
    from astro_app.config import load_hipporag

    return load_hipporag(
        llm_name=args.llm_name,
        save_dir=args.save_dir,
        llm_base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model,
        qa_top_k=args.qa_top_k,
        qa_strict_grounding=False if args.no_strict else None,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def engine_run_config(engine: Any, args: argparse.Namespace, qa_top_k: int) -> Dict[str, Any]:
    """Параметры прогона для отчёта: по ним видно, что именно сравнивали."""
    config = getattr(engine, "global_config", None)
    return {
        "engine": "HippoRAG.rag_qa",
        "save_dir": _json_value(getattr(config, "save_dir", args.save_dir)),
        "llm_name": _json_value(getattr(config, "llm_name", args.llm_name)),
        "llm_base_url": _json_value(getattr(config, "llm_base_url", args.llm_base_url)),
        "embedding_model_name": _json_value(
            getattr(config, "embedding_model_name", args.embedding_model)
        ),
        "qa_top_k": qa_top_k,
        "qa_strict_grounding": _json_value(
            getattr(config, "qa_strict_grounding", not args.no_strict)
        ),
        "limit": args.limit,
    }


def _write_prediction(handle: Any, prediction: eg.Prediction) -> None:
    handle.write(eg.predictions_to_jsonl([prediction]))
    handle.flush()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=str(eg.DEFAULT_GOLDEN), help="JSONL/JSON с разметкой")
    parser.add_argument("--limit", type=int, help="первые N вопросов")
    parser.add_argument("--save-dir", help="каталог индекса HippoRAG")
    parser.add_argument("--llm-name", help="имя модели в LM Studio")
    parser.add_argument("--llm-base-url", help="OpenAI-совместимый endpoint LM Studio")
    parser.add_argument("--embedding-model", help="модель эмбеддингов индекса")
    parser.add_argument(
        "--qa-top-k",
        type=int,
        help=f"сколько фрагментов получает LLM (по умолчанию из конфига, {DEFAULT_QA_TOP_K})",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="не заменять ответ без валидных цитат на отказ",
    )
    parser.add_argument("--baseline", help="JSONL/JSON с предсказаниями baseline")
    parser.add_argument(
        "--out-predictions",
        default=str(DEFAULT_PREDICTIONS),
        help="куда сохранять предсказания (пишутся по мере прогона)",
    )
    parser.add_argument("--out-json", default=str(eg.DEFAULT_REPORT_JSON))
    parser.add_argument("--out-md", default=str(eg.DEFAULT_REPORT_MD))
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="не возвращать код 1 при провале порогов",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="остановиться на первой ошибке выполнения",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        golden = eg.load_golden(args.golden)
        if args.limit is not None:
            if args.limit < 1:
                raise eg.GoldenSetError("--limit должен быть >= 1")
            golden = eg.GoldenSet(items=golden.items[: args.limit])
        if args.qa_top_k is not None and args.qa_top_k < 1:
            raise eg.GoldenSetError("--qa-top-k должен быть >= 1")

        errors, warnings = eg.validate_golden(
            golden,
            enforce_distribution=(args.limit is None),
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

        baseline_result = None
        if args.baseline:
            baseline_result = eg.evaluate(golden, eg.load_predictions(args.baseline))

        try:
            engine = load_engine(args)
        except Exception as exc:  # noqa: BLE001 — CLI должен объяснить, а не упасть трейсбеком
            print(
                f"ошибка загрузки HippoRAG: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2

        qa_top_k = resolve_qa_top_k(engine, args.qa_top_k)
        run_config = engine_run_config(engine, args, qa_top_k)
        print(
            f"движок: HippoRAG.rag_qa | qa_top_k={qa_top_k} | "
            f"strict={run_config['qa_strict_grounding']}",
            flush=True,
        )

        handle = None
        try:
            if args.out_predictions:
                predictions_path = Path(args.out_predictions)
                predictions_path.parent.mkdir(parents=True, exist_ok=True)
                handle = predictions_path.open("w", encoding="utf-8")
            predictions_list = run_questions(
                golden,
                engine,
                qa_top_k=qa_top_k,
                fail_fast=args.fail_fast,
                on_prediction=(
                    (lambda prediction: _write_prediction(handle, prediction))
                    if handle is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 — --fail-fast или сбой записи
            print(f"ошибка выполнения: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        finally:
            if handle is not None:
                handle.close()

        result = eg.evaluate(golden, {item.id: item for item in predictions_list})
        report = eg.build_report(
            golden,
            result,
            dataset_path=args.golden,
            mode="real-rag_qa",
            baseline=baseline_result,
            baseline_path=args.baseline,
        )
        report["run_config"] = run_config
        eg._print_summary(report)

        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if args.out_md:
            Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_md).write_text(
                eg.format_markdown(golden, report),
                encoding="utf-8",
            )

        if args.no_thresholds:
            return 0
        return 0 if all(check["passed"] for check in report["thresholds"].values()) else 1
    except eg.GoldenSetError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
