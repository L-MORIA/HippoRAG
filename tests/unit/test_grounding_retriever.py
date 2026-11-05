"""Тесты real-eval адаптера HippoRAG.rag_qa (H2-6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_grounding as eg  # noqa: E402
import grounding_retriever as gr  # noqa: E402


def _citation(title: str = "A.pdf", page: int = 1):
    return SimpleNamespace(doc_title=title, source_file=title, page_num=page)


def _solution(
    *,
    docs=(),
    citations=(),
    used=(),
    answer: str = "",
    abstained: bool = False,
    errors=(),
):
    return SimpleNamespace(
        docs=list(docs),
        citations=list(citations),
        used_citations=list(used),
        answer=answer,
        abstained=abstained,
        grounding_errors=list(errors),
    )


class FakeEngine:
    """Минимальный движок: отдаёт заранее заданный ответ или исключение."""

    def __init__(self, responses, qa_top_k: int = 5):
        self.responses = responses
        self.questions = []
        self.global_config = SimpleNamespace(qa_top_k=qa_top_k)

    def rag_qa(self, queries):
        question = queries[0]
        self.questions.append(question)
        response = self.responses[question]
        if isinstance(response, Exception):
            raise response
        return [response], [""], [{}]


def _answerable(qid: str = "q1", question: str = "Что такое альфа?") -> dict:
    return {
        "id": qid,
        "question": question,
        "answerable": True,
        "gold_answer": "Альфа — это первый элемент.",
        "gold_sources": [{"doc_title": "A.pdf", "page_num": 1}],
        "gold_terms": ["альфа"],
        "notes": "тест",
    }


def _unanswerable(qid: str = "q2", question: str = "Что такое бета?") -> dict:
    return {
        "id": qid,
        "question": question,
        "answerable": False,
        "gold_answer": "",
        "gold_sources": [],
        "gold_terms": [],
        "notes": "тест",
    }


def _golden(*entries) -> eg.GoldenSet:
    return eg.GoldenSet(
        items=[eg.parse_golden_entry(raw, f"test:{index}") for index, raw in enumerate(entries, 1)]
    )


def _write_golden(path: Path, entries) -> None:
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n",
        encoding="utf-8",
    )


def _grounded_answer(title: str = "A.pdf", page: int = 1):
    return _solution(
        docs=["Альфа — это первый элемент."],
        citations=[_citation(title, page)],
        used=[_citation(title, page)],
        answer="Альфа — это первый элемент [1].",
    )


def _abstain():
    return _solution(answer=eg.ABSTAIN_TEXT, abstained=True)


def test_prediction_uses_used_citations_and_prompt_fragment_count():
    solution = _solution(
        docs=["1", "2", "3"],
        citations=[_citation("A.pdf", 1), _citation("B.pdf", 2), _citation("C.pdf", 3)],
        used=[_citation("C.pdf", 3), _citation("A.pdf", 1)],
        answer="Ответ [1][2].",
        errors=["uncited_sentences"],
    )

    prediction = gr.prediction_from_solution("q1", solution, qa_top_k=2)

    assert prediction.fragment_count == 2
    assert [eg.source_key(item) for item in prediction.citations] == [("c", 3), ("a", 1)]
    assert [eg.source_key(item) for item in prediction.retrieved] == [("a", 1), ("b", 2)]
    assert prediction.errors == ["uncited_sentences"]


def test_prediction_skips_citations_with_unknown_page_or_title():
    solution = _solution(
        docs=["1", "2", "3"],
        citations=[_citation("A.pdf", 1), _citation("B.pdf", 0), _citation("", 2)],
        used=[_citation("A.pdf", 1), _citation("C.pdf", -1), _citation("", 3)],
    )

    prediction = gr.prediction_from_solution("q1", solution, qa_top_k=5)

    assert prediction.fragment_count == 3
    assert [eg.source_key(item) for item in prediction.retrieved] == [("a", 1)]
    assert [eg.source_key(item) for item in prediction.citations] == [("a", 1)]


@pytest.mark.parametrize(
    ("docs", "qa_top_k", "expected"),
    [
        (10, 5, 5),
        (2, 5, 2),
        (3, 8, 3),
    ],
)
def test_fragment_count_is_limited_by_docs_and_qa_top_k(docs, qa_top_k, expected):
    solution = _solution(docs=[str(index) for index in range(docs)])

    prediction = gr.prediction_from_solution("q1", solution, qa_top_k=qa_top_k)

    assert prediction.fragment_count == expected


def test_runtime_error_does_not_abort_the_run():
    golden = _golden(
        _unanswerable("q1", "Вопрос q1?"),
        _unanswerable("q2", "Вопрос q2?"),
    )
    engine = FakeEngine(
        {
            "Вопрос q1?": RuntimeError("LM Studio недоступна"),
            "Вопрос q2?": _abstain(),
        }
    )

    predictions = gr.run_questions(golden, engine, qa_top_k=5)

    assert engine.questions == ["Вопрос q1?", "Вопрос q2?"]
    assert predictions[0].errors == ["runtime_error:RuntimeError"]
    assert predictions[0].answer == ""
    assert predictions[0].abstained is False
    assert predictions[0].fragment_count is None
    assert predictions[1].abstained is True
    assert all(isinstance(item.latency_ms, float) for item in predictions)


def test_fail_fast_stops_on_first_error():
    golden = _golden(_unanswerable("q1", "Вопрос q1?"), _unanswerable("q2", "Вопрос q2?"))
    engine = FakeEngine({"Вопрос q1?": RuntimeError("boom")})

    with pytest.raises(RuntimeError, match="boom"):
        gr.run_questions(golden, engine, qa_top_k=5, fail_fast=True)

    assert engine.questions == ["Вопрос q1?"]


def test_cli_writes_predictions_reports_and_run_config(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden.jsonl"
    _write_golden(
        golden_path,
        [
            _answerable("q1", "Вопрос q1?"),
            _unanswerable("q2", "Вопрос q2?"),
            _unanswerable("q3", "Вопрос q3?"),
        ],
    )
    engine = FakeEngine(
        {
            "Вопрос q1?": _grounded_answer(),
            "Вопрос q2?": _abstain(),
        },
        qa_top_k=2,
    )
    monkeypatch.setattr(gr, "load_engine", lambda args: engine)

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    predictions_path = tmp_path / "predictions.jsonl"

    code = gr.main(
        [
            "--golden",
            str(golden_path),
            "--limit",
            "2",
            "--out-json",
            str(json_path),
            "--out-md",
            str(md_path),
            "--out-predictions",
            str(predictions_path),
        ]
    )

    assert code == 0
    assert engine.questions == ["Вопрос q1?", "Вопрос q2?"]

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["mode"] == "real-rag_qa"
    assert report["quality_measurement"] is True
    assert report["run_config"]["qa_top_k"] == 2
    assert report["run_config"]["limit"] == 2
    assert report["metrics"]["answer_rate_answerable"] == 1.0
    assert report["metrics"]["abstention_rate_unanswerable"] == 1.0
    assert all(check["passed"] for check in report["thresholds"].values())

    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["q1", "q2"]

    markdown = md_path.read_text(encoding="utf-8")
    assert "`real-rag_qa`" in markdown
    assert "- Запуск:" in markdown
    assert "qa_top_k=2" in markdown


def test_load_engine_passes_eval_overrides(monkeypatch):
    import astro_app.config as config

    captured = {}

    def fake_load_hipporag(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(config, "load_hipporag", fake_load_hipporag)
    args = gr.parse_args(
        [
            "--save-dir",
            "idx",
            "--llm-name",
            "model",
            "--llm-base-url",
            "http://localhost:1234/v1",
            "--embedding-model",
            "embed",
            "--qa-top-k",
            "8",
            "--no-strict",
        ]
    )

    gr.load_engine(args)

    assert captured == {
        "llm_name": "model",
        "llm_api_name": None,
        "save_dir": "idx",
        "llm_base_url": "http://localhost:1234/v1",
        "embedding_model_name": "embed",
        "qa_top_k": 8,
        "qa_strict_grounding": False,
    }


def test_load_engine_uses_separate_api_name(monkeypatch):
    """Каталог индекса и id модели в API могут различаться (LM Studio)."""
    import astro_app.config as config

    captured = {}

    def fake_load_hipporag(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(config, "load_hipporag", fake_load_hipporag)
    monkeypatch.delenv("LLM_NAME", raising=False)
    args = gr.parse_args(
        [
            "--llm-name",
            "gemma-4-e4b",
            "--llm-api-name",
            "google/gemma-4-e4b",
        ]
    )

    gr.load_engine(args)

    assert captured["llm_name"] == "gemma-4-e4b"
    assert captured["llm_api_name"] == "google/gemma-4-e4b"

    run_config = gr.engine_run_config(
        SimpleNamespace(
            global_config=SimpleNamespace(llm_name="gemma-4-e4b", llm_api_name="google/gemma-4-e4b")
        ),
        args,
        5,
    )
    assert run_config["llm_name"] == "gemma-4-e4b"
    assert run_config["llm_api_name"] == "google/gemma-4-e4b"


def test_resolve_qa_top_k_falls_back_to_engine_config():
    engine = SimpleNamespace(global_config=SimpleNamespace(qa_top_k=7))

    assert gr.resolve_qa_top_k(engine) == 7
    assert gr.resolve_qa_top_k(engine, 3) == 3
    assert gr.resolve_qa_top_k(SimpleNamespace()) == gr.DEFAULT_QA_TOP_K
    assert gr.resolve_qa_top_k(SimpleNamespace(global_config=SimpleNamespace(qa_top_k=0))) == 5
