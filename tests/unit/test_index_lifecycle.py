"""Тесты H2-2: статус индекса, документы, backup/restore, delete, reindex."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from astro_app import versions
from astro_app.index_lifecycle import IndexLifecycle
from hipporag.embedding_store import EmbeddingStore


class FakeEmbeddingModel:
    def batch_encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeStore:
    """Минимальное хранилище чанков с metadata."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def get_all_ids(self):
        return list(self.rows)

    def get_all_id_to_rows(self):
        return dict(self.rows)

    def get_row(self, hash_id):
        return self.rows[hash_id]

    def remove_metadata_entries(self, hash_id, source_files):
        targets = {Path(str(source).replace("\\", "/")).name for source in source_files}
        row = self.rows.get(hash_id)
        if row is None:
            return False
        entries = row.get("metadata") or []
        kept = [
            entry
            for entry in entries
            if Path(str(entry.get("source_file") or "").replace("\\", "/")).name not in targets
        ]
        if len(kept) == len(entries):
            return False
        row["metadata"] = kept
        return True

    def close(self):
        pass


class FakeHippoRAG:
    """HippoRAG-заглушка: delete/index_pdfs пишут в журнал и меняют store."""

    def __init__(self, index_dir: Path, rows=None):
        self.working_dir = str(index_dir)
        self.chunk_embedding_store = FakeStore(rows)
        self.fact_embedding_store = FakeStore()
        self.deleted = []
        self.indexed = []
        self.reloads = 0

    def delete(self, texts):
        self.deleted.extend(texts)
        self.chunk_embedding_store.rows = {
            hash_id: row
            for hash_id, row in self.chunk_embedding_store.rows.items()
            if row.get("content") not in texts
        }
        self._write_index_manifest()

    def index_pdfs(self, paths):
        self.indexed.extend(paths)
        for number, path in enumerate(paths, start=1):
            hash_id = f"new-{number}"
            self.chunk_embedding_store.rows[hash_id] = {
                "content": f"new chunk from {Path(path).name}",
                "metadata": [
                    {
                        "source_file": Path(path).name,
                        "doc_title": Path(path).stem,
                        "page_num": 1,
                    }
                ],
            }
        self._write_index_manifest()

    def _write_index_manifest(self):
        versions.write_index_manifest(
            self.working_dir,
            llm_name="test-llm",
            embedding_model="test-embedding",
            counts={"chunks": len(self.chunk_embedding_store.rows)},
        )


@pytest.fixture
def index_env(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "graph.pickle").write_bytes(b"original-graph")
    (index_dir / "chunk_embeddings").mkdir()
    (index_dir / "chunk_embeddings" / "vdb_chunk.parquet").write_bytes(b"parquet")
    versions.write_index_manifest(
        index_dir,
        llm_name="test-llm",
        embedding_model="test-embedding",
        counts={"chunks": 3},
    )

    rows = {
        "h1": {
            "content": "shared chunk",
            "metadata": [
                {"source_file": "a.pdf", "doc_title": "A", "page_num": 1},
                {"source_file": "b.pdf", "doc_title": "B", "page_num": 2},
            ],
        },
        "h2": {
            "content": "only a",
            "metadata": [{"source_file": "a.pdf", "doc_title": "A", "page_num": 1}],
        },
        "h3": {
            "content": "only b",
            "metadata": [{"source_file": "b.pdf", "doc_title": "B", "page_num": 2}],
        },
    }
    hipporag = FakeHippoRAG(index_dir, rows)
    lifecycle = IndexLifecycle(
        index_dir,
        backup_dir=tmp_path / "backups",
        get_hipporag=lambda: hipporag,
        reload_hipporag=lambda: hipporag,
    )
    return lifecycle, hipporag, index_dir


def test_status_reports_manifest_and_counts(index_env):
    lifecycle, _, _ = index_env
    status = lifecycle.status()

    assert status["state"] == "idle"
    assert status["index"]["format_version"] == versions.INDEX_FORMAT_VERSION
    assert status["index"]["format_state"] == versions.INDEX_OK
    assert status["index"]["chunks"] == 3
    assert status["index"]["documents"] == 2
    assert status["backups"] == 0


def test_documents_groups_by_source(index_env):
    lifecycle, _, _ = index_env
    documents = lifecycle.documents()

    assert [item["source_file"] for item in documents] == ["a.pdf", "b.pdf"]
    by_name = {item["source_file"]: item for item in documents}
    assert by_name["a.pdf"]["chunks"] == 2
    assert by_name["b.pdf"]["chunks"] == 2
    assert by_name["a.pdf"]["pages"] == 1
    assert by_name["b.pdf"]["pages"] == 1
    assert by_name["a.pdf"]["doc_title"] == "A"


def test_backup_creates_zip_and_lists(index_env):
    lifecycle, _, _ = index_env
    record = lifecycle.backup("manual")

    archive = Path(record["path"])
    assert archive.exists()
    assert archive.name.endswith(".zip")
    assert "manual" in archive.name
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "graph.pickle" in names
    assert "index_manifest.json" in names

    backups = lifecycle.list_backups()
    assert len(backups) == 1
    assert backups[0]["name"] == archive.name
    assert backups[0]["label"] == "manual"


def test_restore_replaces_index_and_keeps_manifest(index_env):
    lifecycle, _, index_dir = index_env
    record = lifecycle.backup()

    (index_dir / "graph.pickle").write_bytes(b"changed-graph")
    result = lifecycle.restore(record["name"])

    assert result["restored"] == record["name"]
    assert (index_dir / "graph.pickle").read_bytes() == b"original-graph"
    assert versions.check_index(index_dir).state == versions.INDEX_OK


def test_restore_rejects_path_traversal(index_env):
    lifecycle, _, _ = index_env
    with pytest.raises(ValueError):
        lifecycle.restore("../evil.zip")


def test_restore_rejects_archive_without_graph(index_env, tmp_path):
    lifecycle, _, _ = index_env
    lifecycle.backup_dir.mkdir(parents=True, exist_ok=True)
    archive = lifecycle.backup_dir / "broken.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("index_manifest.json", "{}")

    with pytest.raises(ValueError, match="graph.pickle"):
        lifecycle.restore("broken.zip")


def test_delete_documents_skips_shared_chunks(index_env):
    lifecycle, hipporag, _ = index_env
    result = lifecycle.delete_documents(["a.pdf"])

    assert result["deleted_documents"] == ["a.pdf"]
    assert result["deleted_chunks"] == 1
    assert result["shared_chunks_updated"] == 1
    assert result["skipped_shared_chunks"] == 0
    assert hipporag.deleted == ["only a"]
    assert "a.pdf" not in {item["source_file"] for item in lifecycle.documents()}


def test_delete_documents_reports_missing(index_env):
    lifecycle, hipporag, _ = index_env
    result = lifecycle.delete_documents(["missing.pdf"])

    assert result["deleted_documents"] == []
    assert result["missing_documents"] == ["missing.pdf"]
    assert result["deleted_chunks"] == 0
    assert hipporag.deleted == []


def test_reindex_backs_up_deletes_and_indexes(index_env, tmp_path):
    lifecycle, hipporag, _ = index_env
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    result = lifecycle.reindex([str(pdf)])

    assert Path(result["backup"]["path"]).exists()
    assert hipporag.indexed == [str(pdf)]
    assert hipporag.deleted == ["only a"]
    assert result["chunks_after"] == result["chunks_before"]
    assert {item["source_file"] for item in lifecycle.documents()} == {"a.pdf", "b.pdf"}


def test_reindex_rejects_missing_pdf(index_env, tmp_path):
    lifecycle, _, _ = index_env
    with pytest.raises(FileNotFoundError):
        lifecycle.reindex([str(tmp_path / "missing.pdf")])


def test_reindex_rejects_empty_list(index_env):
    lifecycle, _, _ = index_env
    with pytest.raises(ValueError):
        lifecycle.reindex([])


def test_embedding_store_remove_metadata_entries(tmp_path):
    store = EmbeddingStore(FakeEmbeddingModel(), tmp_path / "db", 1, "chunk")
    store.insert_strings(
        ["shared chunk"],
        metadatas=[{"source_file": "a.pdf", "doc_title": "A", "page_num": 1}],
    )
    store.insert_strings(
        ["shared chunk"],
        metadatas=[{"source_file": "b.pdf", "doc_title": "B", "page_num": 2}],
    )
    hash_id = store.get_hash_id("shared chunk")
    assert len(store.get_all_metadata_entries(hash_id)) == 2

    assert store.remove_metadata_entries(hash_id, {"a.pdf"}) is True
    entries = store.get_all_metadata_entries(hash_id)
    assert [entry["source_file"] for entry in entries] == ["b.pdf"]
    assert store.remove_metadata_entries(hash_id, {"a.pdf"}) is False


# ── сброс кэша page-индекса (H1-6: page-level в UI) ─────────────────────────


def test_reindex_resets_page_index_cache(index_env, tmp_path, monkeypatch):
    """После reindex кэш page-индекса должен сбрасываться (новые страницы)."""
    from astro_app import index_lifecycle as il

    lifecycle, hipporag, _ = index_env
    resets = []
    monkeypatch.setattr(il, "reset_page_index_cache", lambda: resets.append(True))

    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    lifecycle.reindex([str(pdf)])

    assert hipporag.indexed == [str(pdf)]
    assert resets == [True]


def test_restore_resets_page_index_cache(index_env, monkeypatch):
    """После restore кэш page-индекса должен сбрасываться (старое содержимое)."""
    from astro_app import index_lifecycle as il

    lifecycle, _, _ = index_env
    backup = lifecycle.backup(label="cache-check")
    resets = []
    monkeypatch.setattr(il, "reset_page_index_cache", lambda: resets.append(True))

    lifecycle.restore(backup["name"])

    assert resets == [True]
