"""Закоммиченная синтетическая фикстура H1-5 не разъехалась и проходит gate.

Контракт формата проверяется отдельно (`test_index_fixture_contract.py`) на
чанках в tempdir. Здесь — про то, что лежит **в репозитории**: файлы целы по
манифесту, совпадают со сборкой из `scripts/make_synthetic_fixture.py` и дают
recall не ниже порога, который стоит в CI-джобе `Index Fixture Smoke`.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from astro_app import evaluation as ev
from astro_app import index_fixture as fix

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "fixtures" / "synthetic_mini"
QUESTIONS = FIXTURE_DIR / "questions.jsonl"

#: Тот же порог, что в `.github/workflows/ci-cd.yml`.
MIN_RECALL = 0.9


def _questions() -> list:
    dataset = ev.load_dataset(QUESTIONS)
    return [
        {
            "qid": question.qid,
            "question": question.question,
            "source_file": question.source_file,
        }
        for question in dataset.for_run(include_draft=False)
    ]


def test_committed_fixture_is_present_and_intact():
    assert (FIXTURE_DIR / fix.FIXTURE_FILE).exists(), "нет fixtures/synthetic_mini/fixture.json"
    assert fix.verify_fixture(FIXTURE_DIR) == []


def test_committed_fixture_covers_several_documents():
    """Смоук должен проверять выбор документа, а не только строки внутри одного."""
    fixture = fix.load_fixture(FIXTURE_DIR)
    sources = {source for chunk in fixture["chunks"] for source in fix.chunk_sources(chunk)}
    assert len(sources) >= 2, f"в фикстуре один документ: {sources}"


def test_committed_questions_reference_fixture_documents():
    fixture = fix.load_fixture(FIXTURE_DIR)
    available = {source for chunk in fixture["chunks"] for source in fix.chunk_sources(chunk)}
    questions = _questions()
    assert questions, "разметка к фикстуре пуста"
    unknown = sorted({q["source_file"] for q in questions} - available)
    assert not unknown, f"разметка ссылается на файлы вне фикстуры: {unknown}"


def test_committed_questions_are_complete_and_verified():
    dataset = ev.load_dataset(QUESTIONS)
    assert not dataset.incomplete
    assert dataset.questions
    for question in dataset.questions:
        assert question.status == "verified"
        assert question.reference_answer


def test_committed_fixture_passes_recall_gate():
    fixture = fix.load_fixture(FIXTURE_DIR)
    questions = _questions()
    smoke = fix.smoke_retrieval(fixture, questions, top_k=5)
    assert smoke["n_questions"] >= 10
    assert smoke["recall"] >= MIN_RECALL, (
        f"recall@5 = {smoke['recall']} < {MIN_RECALL}: "
        f"промахи {[row['qid'] for row in smoke['rows'] if not row['hit']]}"
    )

    # Запас: при top-1 тоже всё попадает. Иначе порог держится на грани и первая
    # же правка текста уронит CI неочевидным образом.
    strict = fix.smoke_retrieval(fixture, questions, top_k=1)
    assert (
        strict["recall"] >= MIN_RECALL
    ), f"recall@1 = {strict['recall']} — слишком малый запас у порога {MIN_RECALL}"


def test_committed_fixture_matches_generator():
    """Правка JSON руками вместо скрипта должна ломать CI, а не проходить молча."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_synthetic_fixture.py"),
            "--check",
            "--out",
            str(FIXTURE_DIR),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_loader_rejects_typo_in_field_name(tmp_path):
    """Опечатка в имени поля падает, а не превращается в пустой ground truth."""
    raw = {
        "qid": "syn-typo-check",
        "question": "Проверка опечатки в имени поля?",
        "kind": "domain",
        "subtype": "other",
        "sourse_file": "demo_minerals.pdf",
        "page_num": 1,
        "reference_answer": "Значение не важно.",
        "status": "verified",
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ev.DatasetError):
        ev.load_dataset(path)
