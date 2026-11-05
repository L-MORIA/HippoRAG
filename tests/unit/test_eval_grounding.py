"""Тесты golden set и метрик заземления (H2-6, D-9/D-10)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_grounding as eg  # noqa: E402


def _entry(
    qid: str,
    *,
    answerable: bool = True,
    sources=None,
    terms=None,
    borderline: bool = False,
    answer: str = "Эталонный ответ.",
):
    return {
        "id": qid,
        "question": f"Вопрос {qid}?",
        "answerable": answerable,
        "gold_answer": answer if answerable else "",
        "gold_sources": sources or [],
        "gold_terms": terms or [],
        "notes": "тест",
        "borderline": borderline,
    }


def _golden(*entries) -> eg.GoldenSet:
    return eg.GoldenSet(
        items=[eg.parse_golden_entry(raw, f"test:{index}") for index, raw in enumerate(entries, 1)]
    )


def _prediction(
    qid: str,
    *,
    answer: str,
    citations=None,
    abstained: bool = False,
    fragment_count: int | None = 1,
    errors=None,
    retrieved=None,
):
    return eg.Prediction(
        id=qid,
        answer=answer,
        abstained=abstained,
        citations=citations or [],
        retrieved=retrieved or [],
        retrieved_provided=retrieved is not None,
        fragment_count=fragment_count,
        errors=errors or [],
    )


def test_golden_file_has_planned_composition():
    golden = eg.load_golden()
    errors, warnings = eg.validate_golden(golden, enforce_distribution=True)

    assert errors == []
    assert warnings == []
    assert len(golden.items) == 36
    assert len(golden.answerable) - len(golden.borderline) == 18
    assert len(golden.borderline) == 5
    assert len(golden.unanswerable) == 13


def test_gold_terms_are_present_in_gold_answers():
    golden = eg.load_golden()
    for question in golden.answerable:
        body = eg.normalize_text(eg.answer_body(question.gold_answer))
        missing = [term for term in question.gold_terms if not eg._term_present(term, body)]
        assert not missing, f"{question.id}: нет терминов {missing} в эталоне"


def test_parse_rejects_unknown_field():
    raw = _entry("q1", sources=[{"doc_title": "A.pdf", "page_num": 1}])
    raw["sourse_file"] = "опечатка"

    with pytest.raises(eg.GoldenSetError, match="неизвестные поля"):
        eg.parse_golden_entry(raw, "test")


def test_answerable_requires_gold_sources():
    raw = _entry("q1", sources=[], terms=["ответ"])

    with pytest.raises(eg.GoldenSetError, match="gold_source"):
        eg.parse_golden_entry(raw, "test")


def test_unanswerable_rejects_gold_sources():
    raw = _entry(
        "q1",
        answerable=False,
        sources=[{"doc_title": "A.pdf", "page_num": 1}],
    )

    with pytest.raises(eg.GoldenSetError, match="unanswerable"):
        eg.parse_golden_entry(raw, "test")


def test_borderline_requires_two_sources():
    raw = _entry(
        "q1",
        sources=[{"doc_title": "A.pdf", "page_num": 1}],
        terms=["ответ"],
        borderline=True,
    )

    with pytest.raises(eg.GoldenSetError, match="borderline"):
        eg.parse_golden_entry(raw, "test")


def test_gold_terms_are_derived_when_omitted():
    raw = {
        "id": "q1",
        "question": "Что такое альфа?",
        "answerable": True,
        "gold_answer": "Альфа — это первый элемент.",
        "gold_sources": [{"doc_title": "A.pdf", "page_num": 1}],
        "notes": "тест",
    }

    question = eg.parse_golden_entry(raw, "test")

    assert "альфа" in question.gold_terms


def test_source_key_normalizes_pdf_and_case():
    left = eg.SourceRef("Книга.PDF", 7)
    right = eg.SourceRef("  книга  ", 7)

    assert eg.source_key(left) == eg.source_key(right)


def test_perfect_predictions_pass_all_metrics():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        ),
        _entry("q2", answerable=False),
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1].",
            citations=[eg.SourceRef("A.pdf", 1)],
        ),
        "q2": _prediction("q2", answer=eg.ABSTAIN_TEXT, abstained=True),
    }

    result = eg.evaluate(golden, predictions)
    metrics = result["metrics"]

    assert metrics["answer_rate_answerable"] == 1.0
    assert metrics["abstention_rate_unanswerable"] == 1.0
    assert metrics["false_answer_rate"] == 0.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert metrics["marker_validity"] == 1.0
    assert metrics["uncited_sentence_rate"] == 0.0
    assert metrics["answer_accuracy"] == 1.0
    assert all(check["passed"] for check in eg.check_thresholds(metrics).values())


def test_false_answer_on_unanswerable_is_counted():
    golden = _golden(_entry("q2", answerable=False))
    predictions = {
        "q2": _prediction(
            "q2",
            answer="Ответ есть [1].",
            citations=[eg.SourceRef("A.pdf", 1)],
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["abstention_rate_unanswerable"] == 0.0
    assert metrics["false_answer_rate"] == 1.0
    assert metrics["answer_rate_answerable"] is None


def test_citation_precision_and_recall():
    golden = _golden(
        _entry(
            "q1",
            sources=[
                {"doc_title": "A.pdf", "page_num": 1},
                {"doc_title": "B.pdf", "page_num": 2},
            ],
            terms=["альфа"],
        )
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1][2].",
            citations=[eg.SourceRef("A.pdf", 1), eg.SourceRef("C.pdf", 3)],
            fragment_count=2,
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["citation_precision"] == 0.5
    assert metrics["citation_recall"] == 0.5
    assert metrics["answer_accuracy"] == 1.0


def test_answer_without_citations_has_zero_precision():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {"q1": _prediction("q1", answer="Альфа.", citations=[])}

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["citation_precision"] == 0.0
    assert metrics["citation_recall"] == 0.0


def test_marker_validity_penalizes_out_of_range():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1][3].",
            citations=[eg.SourceRef("A.pdf", 1)],
            fragment_count=2,
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["marker_validity"] == 0.5


def test_marker_validity_is_zero_for_uncited_answer():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {"q1": _prediction("q1", answer="Альфа.", fragment_count=1)}

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["marker_validity"] == 0.0


def test_uncited_sentence_rate_detects_long_sentence_without_marker():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    answer = (
        "Это очень длинное предложение без ссылки, которое явно длиннее сорока "
        "символов. Альфа [1]."
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer=answer,
            citations=[eg.SourceRef("A.pdf", 1)],
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["uncited_sentence_rate"] == 1.0


def test_answer_accuracy_requires_all_gold_terms():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа", "бета"],
        )
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1].",
            citations=[eg.SourceRef("A.pdf", 1)],
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["answer_accuracy"] == 0.0


def test_abstained_answerable_counts_as_incorrect():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {"q1": _prediction("q1", answer=eg.ABSTAIN_TEXT, abstained=True)}

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["answer_rate_answerable"] == 0.0
    assert metrics["answer_accuracy"] == 0.0


def test_evaluate_rejects_missing_and_unknown_predictions():
    golden = _golden(_entry("q1", sources=[{"doc_title": "A.pdf", "page_num": 1}]))

    with pytest.raises(eg.GoldenSetError, match="нет предсказаний"):
        eg.evaluate(golden, {})
    with pytest.raises(eg.GoldenSetError, match="неизвестных id"):
        eg.evaluate(
            golden,
            {
                "q1": _prediction("q1", answer="Альфа [1]."),
                "other": _prediction("other", answer="Альфа [1]."),
            },
        )


def test_compare_runs_marks_improved_and_regressed_questions():
    answerable = [
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        ),
        _entry(
            "q2",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["бета"],
        ),
    ]
    golden = _golden(*answerable)
    baseline = eg.evaluate(
        golden,
        {
            "q1": _prediction("q1", answer="Неверно."),
            "q2": _prediction(
                "q2",
                answer="Бета [1].",
                citations=[eg.SourceRef("A.pdf", 1)],
            ),
        },
    )
    after = eg.evaluate(
        golden,
        {
            "q1": _prediction(
                "q1",
                answer="Альфа [1].",
                citations=[eg.SourceRef("A.pdf", 1)],
            ),
            "q2": _prediction("q2", answer="Неверно."),
        },
    )

    comparison = eg.compare_runs(baseline, after)

    assert comparison["improved"] == ["q1"]
    assert comparison["regressed"] == ["q2"]


def test_predictions_roundtrip(tmp_path):
    prediction = _prediction(
        "q1",
        answer="Альфа [1].",
        citations=[eg.SourceRef("A.pdf", 1)],
        fragment_count=1,
    )
    path = tmp_path / "predictions.jsonl"
    path.write_text(eg.predictions_to_jsonl([prediction]), encoding="utf-8")

    loaded = eg.load_predictions(path)

    assert loaded["q1"].answer == prediction.answer
    assert eg.source_key(loaded["q1"].citations[0]) == eg.source_key(prediction.citations[0])


def test_markdown_report_contains_metrics_and_question_table():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    result = eg.evaluate(
        golden,
        {
            "q1": _prediction(
                "q1",
                answer="Альфа [1].",
                citations=[eg.SourceRef("A.pdf", 1)],
            )
        },
    )
    report = eg.build_report(
        golden,
        result,
        dataset_path="golden_qa.jsonl",
        mode="predictions",
    )

    markdown = eg.format_markdown(golden, report)

    assert "## Метрики" in markdown
    assert "## Вопросы" in markdown
    assert "`q1`" in markdown
    assert "Все пороги" not in markdown


def test_cli_validate_only_accepts_repository_dataset(capsys):
    code = eg.main(["--validate-only"])

    assert code == 0
    assert "разметка корректна" in capsys.readouterr().out


def test_cli_stub_grounded_passes_thresholds(tmp_path, capsys):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    predictions_path = tmp_path / "predictions.jsonl"

    code = eg.main(
        [
            "--stub-llm",
            "--stub-mode",
            "grounded",
            "--out-json",
            str(json_path),
            "--out-md",
            str(md_path),
            "--out-predictions",
            str(predictions_path),
        ]
    )

    assert code == 0
    assert json_path.exists()
    assert md_path.exists()
    assert predictions_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["metrics"]["false_answer_rate"] == 0.0
    assert report["metrics"]["answer_accuracy"] == 1.0
    assert "Все пороги выполнены" in capsys.readouterr().out


def test_cli_stub_legacy_fails_citation_thresholds(tmp_path):
    code = eg.main(
        [
            "--stub-llm",
            "--stub-mode",
            "legacy",
            "--out-json",
            str(tmp_path / "report.json"),
            "--out-md",
            str(tmp_path / "report.md"),
            "--no-thresholds",
        ]
    )

    assert code == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["citation_precision"] == 0.0
    assert report["metrics"]["answer_accuracy"] == 1.0


def test_page_recall_at_5_counts_question_when_gold_page_retrieved():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1].",
            citations=[eg.SourceRef("A.pdf", 1)],
            retrieved=[eg.SourceRef("A.pdf", 1), eg.SourceRef("B.pdf", 9)],
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["page_recall_at_5"] == 1.0
    assert metrics["page_recall_measured"] is True


def test_page_recall_at_5_is_zero_when_gold_page_missing():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1].",
            citations=[eg.SourceRef("A.pdf", 1)],
            retrieved=[eg.SourceRef("B.pdf", 9)],
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["page_recall_at_5"] == 0.0
    assert metrics["page_coverage_at_5"] == 0.0


def test_page_recall_is_none_without_retrieved_block():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )
    predictions = {"q1": _prediction("q1", answer="Альфа [1].", citations=[])}
    result = eg.evaluate(golden, predictions)

    assert result["metrics"]["page_recall_at_5"] is None
    assert result["metrics"]["page_recall_measured"] is False
    assert eg.check_retrieval_thresholds(result["metrics"]) == {}


def test_page_recall_uses_any_gold_page_but_coverage_counts_all():
    golden = _golden(
        _entry(
            "q1",
            sources=[
                {"doc_title": "A.pdf", "page_num": 1},
                {"doc_title": "B.pdf", "page_num": 2},
            ],
            terms=["альфа"],
            borderline=True,
        )
    )
    predictions = {
        "q1": _prediction(
            "q1",
            answer="Альфа [1].",
            citations=[eg.SourceRef("A.pdf", 1)],
            retrieved=[eg.SourceRef("A.pdf", 1)],
        )
    }

    metrics = eg.evaluate(golden, predictions)["metrics"]

    assert metrics["page_recall_at_5"] == 1.0
    assert metrics["page_coverage_at_5"] == 0.5


def test_page_recall_threshold_is_eighty_percent():
    assert eg.check_retrieval_thresholds({"page_recall_at_5": 0.8})["page_recall_at_5"]["passed"]
    assert not eg.check_retrieval_thresholds({"page_recall_at_5": 0.79})["page_recall_at_5"][
        "passed"
    ]


def test_run_questions_marks_retrieved_provided():
    golden = _golden(
        _entry(
            "q1",
            sources=[{"doc_title": "A.pdf", "page_num": 1}],
            terms=["альфа"],
        )
    )

    predictions = eg.run_questions(golden, stub_mode="abstain")

    assert predictions[0].retrieved_provided is True
    assert eg.page_recall_at_k(predictions[0], golden.items[0], 5) == 1.0


def test_require_page_recall_fails_when_metric_not_measured(tmp_path, capsys):
    golden = eg.load_golden()
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": question.id,
                    "answer": eg.ABSTAIN_TEXT,
                    "abstained": True,
                    "citations": [],
                    "fragment_count": 1,
                },
                ensure_ascii=False,
            )
            for question in golden.items
        )
        + "\n",
        encoding="utf-8",
    )

    code = eg.main(
        [
            "--predictions",
            str(predictions_path),
            "--require-page-recall",
            "--out-json",
            str(tmp_path / "report.json"),
            "--out-md",
            str(tmp_path / "report.md"),
        ]
    )

    assert code == 2
    assert "page recall@5 не измерен" in capsys.readouterr().err
