"""Контракт индексу-фикстуры (H1-5).

Фикстура — урезанный снимок индекса, по которому в CI гоняется лексический
retrieval (BM25) без эмбеддинг-модели, без LLM и без сети. Тест проверяет
формат, манифест с sha256, сам поиск и то, что скрипты запускаются.

Скрипты не требуют индекса: используется синтетическая фикстура.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from astro_app import index_fixture as fix

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

CHUNKS = [
    {
        "hash_id": "chunk-a1",
        "content": "Марс в Овне: квадрат, изгнание, Плутон в доме денег.",
        "metadata": {"source_file": "astro_book.pdf", "page_num": 12},
    },
    {
        "hash_id": "chunk-b2",
        "content": "Венера в Тельце: красота, чувственность, Плутон в доме любви.",
        "metadata": [{"source_file": "astro_book.pdf", "page_num": 44}],
    },
    {
        "hash_id": "chunk-c3",
        "content": "Юпитер в Стрельце: удача, экспансия, асцендент в знаке огня.",
        "metadata": {"source_file": "other_book.pdf", "page_num": 7},
    },
]


def _temp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="hipporag-fixture-"))


def test_save_fixture_writes_files_and_manifest():
    directory = _temp_dir()
    try:
        manifest = fix.save_fixture(CHUNKS, directory, created_at="2026-11-03T00:00:00")
        assert (directory / fix.FIXTURE_FILE).exists()
        assert (directory / fix.MANIFEST_FILE).exists()
        assert manifest["version"] == fix.FIXTURE_VERSION
        assert manifest["n_chunks"] == len(CHUNKS)
        assert fix.FIXTURE_FILE in manifest["files"]
        assert len(manifest["files"][fix.FIXTURE_FILE]) == 64  # sha256
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_load_fixture_roundtrip():
    directory = _temp_dir()
    try:
        fix.save_fixture(CHUNKS, directory)
        fixture = fix.load_fixture(directory)
        assert len(fixture["chunks"]) == len(CHUNKS)
        assert fixture["chunks"][0]["hash_id"] == "chunk-a1"
        assert fixture["meta"]["n_chunks"] == len(CHUNKS)
        # загрузка по пути к самому файлу тоже работает
        assert fix.load_fixture(directory / fix.FIXTURE_FILE) == fixture
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_verify_fixture_accepts_intact_fixture():
    directory = _temp_dir()
    try:
        fix.save_fixture(CHUNKS, directory)
        assert fix.verify_fixture(directory) == []
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_verify_fixture_detects_tampering():
    directory = _temp_dir()
    try:
        fix.save_fixture(CHUNKS, directory)
        path = directory / fix.FIXTURE_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        data["chunks"][0]["content"] = "подмена содержимого"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        problems = fix.verify_fixture(directory)
        assert problems
        assert any("sha256" in problem for problem in problems)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_verify_fixture_reports_missing_manifest():
    directory = _temp_dir()
    try:
        fix.save_fixture(CHUNKS, directory)
        (directory / fix.MANIFEST_FILE).unlink()
        problems = fix.verify_fixture(directory)
        assert problems and "манифест" in problems[0]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_chunk_entries_accepts_dict_and_list_metadata():
    assert fix.chunk_entries(CHUNKS[0]) == (("astro_book.pdf", 12),)
    assert fix.chunk_entries(CHUNKS[1]) == (("astro_book.pdf", 44),)
    assert fix.chunk_sources(CHUNKS[2]) == ["other_book.pdf"]


def test_bm25_ranks_matching_chunk_first():
    ranked = fix.bm25_rank(CHUNKS, "Венера в Тельце красота чувственность", top_k=3)
    assert ranked
    assert ranked[0]["hash_id"] == "chunk-b2"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["source_file"] == "astro_book.pdf"


def test_bm25_on_empty_fixture_is_empty():
    assert fix.bm25_rank([], "что угодно", top_k=5) == []


def test_smoke_retrieval_counts_hits_and_misses():
    directory = _temp_dir()
    try:
        fix.save_fixture(CHUNKS, directory)
        fixture = fix.load_fixture(directory)
        questions = [
            {"qid": "q1", "question": "Венера в Тельце", "source_file": "astro_book.pdf"},
            {"qid": "q2", "question": "Юпитер в Стрельце", "source_file": "other_book.pdf"},
            {"qid": "q3", "question": "Марс в Овне", "source_file": "missing_book.pdf"},
        ]
        result = fix.smoke_retrieval(fixture, questions, top_k=1)
        assert result["n_questions"] == 3
        assert result["n_hits"] == 2
        assert result["recall"] == round(2 / 3, 4)
        assert [row["hit"] for row in result["rows"]] == [True, True, False]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_fixture_scripts_are_runnable():
    """Скрипты собираются и показывают справку (без индекса и моделей)."""
    for script in ("build_index_fixture.py", "index_fixture_smoke.py"):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / script), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        assert result.returncode == 0, f"{script}: {result.stderr[-500:]}"
        assert "usage" in (result.stdout + result.stderr).lower()


def test_smoke_script_reports_missing_fixture():
    """Нет фикстуры — честный код возврата 2, не тихий «успех»."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "index_fixture_smoke.py"),
            "--fixture",
            "fixtures/такой-фикстуры-нет",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert result.returncode == 2
    assert "фикстура" in (result.stdout + result.stderr).lower()
