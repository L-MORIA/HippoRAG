"""Тесты H2-2: HTTP-маршруты жизненного цикла индекса."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hipporag.search_api import server as srv


class FakeLifecycle:
    def __init__(self):
        self.backup_label = None
        self.reindexed = []

    def status(self):
        return {
            "state": "idle",
            "operation": None,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "detail": {},
            "index": {
                "working_dir": "outputs/index",
                "format_version": 1,
                "expected_format_version": 1,
                "format_state": "ok",
                "created_at": "2026-11-05T10:00:00",
                "llm_name": "test",
                "embedding_model": "test",
                "counts": {"chunks": 2},
                "chunks": 2,
                "facts": 1,
                "documents": 1,
            },
            "backups": 1,
        }

    def documents(self):
        return [{"source_file": "a.pdf", "doc_title": "A", "pages": 1, "chunks": 2}]

    def backup(self, label=None):
        self.backup_label = label
        return {
            "name": "index-1.zip",
            "path": "backups/index-1.zip",
            "created_at": "2026-11-05T10:00:00",
            "files": 3,
            "bytes": 100,
            "label": label or "",
            "source_dir": "outputs/index",
        }

    def list_backups(self):
        return [self.backup()]

    def restore(self, name):
        return {"restored": name, "working_dir": "outputs/index", "format_state": "ok"}

    def delete_documents(self, source_files):
        return {
            "deleted_documents": list(source_files),
            "missing_documents": [],
            "deleted_chunks": 2,
            "shared_chunks_updated": 0,
            "skipped_shared_chunks": 0,
        }

    def reindex(self, pdf_paths):
        self.reindexed.append(list(pdf_paths))
        return {"pdf_paths": list(pdf_paths), "backup": {"name": "before.zip"}}


@pytest.fixture
def client(monkeypatch):
    lifecycle = FakeLifecycle()
    monkeypatch.setattr(srv, "index_lifecycle", lifecycle)
    monkeypatch.setattr(srv, "hipporag", None)
    monkeypatch.setattr(srv, "API_KEY", None)
    monkeypatch.setattr(srv, "API_KEY_REQUIRED", False)
    return TestClient(srv.app), lifecycle


def test_index_status_route(client):
    http, _ = client
    response = http.get("/index/status")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "idle"
    assert body["index"]["format_version"] == 1
    assert body["index"]["documents"] == 1


def test_index_documents_route(client):
    http, _ = client
    response = http.get("/index/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["source_file"] == "a.pdf"


def test_index_backup_and_list_routes(client):
    http, lifecycle = client
    response = http.post("/index/backup", json={"label": "manual"})
    assert response.status_code == 200
    assert response.json()["name"] == "index-1.zip"
    assert lifecycle.backup_label == "manual"

    response = http.get("/index/backups")
    assert response.status_code == 200
    assert response.json()["backups"][0]["name"] == "index-1.zip"


def test_index_restore_route(client):
    http, _ = client
    response = http.post("/index/restore", json={"name": "index-1.zip"})
    assert response.status_code == 200
    assert response.json()["restored"] == "index-1.zip"


def test_index_delete_route(client):
    http, _ = client
    response = http.post("/index/delete", json={"source_files": ["a.pdf"]})
    assert response.status_code == 200
    assert response.json()["deleted_documents"] == ["a.pdf"]


def test_index_reindex_route_accepts_existing_pdf(client, tmp_path):
    http, lifecycle = client
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")

    response = http.post("/index/reindex", json={"pdf_paths": [str(pdf)]})
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
    assert lifecycle.reindexed == [[str(pdf)]]


def test_index_reindex_route_rejects_missing_pdf(client, tmp_path):
    http, lifecycle = client
    response = http.post("/index/reindex", json={"pdf_paths": [str(tmp_path / "no.pdf")]})
    assert response.status_code == 400
    assert lifecycle.reindexed == []


def test_health_includes_index_fields(client):
    http, _ = client
    response = http.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["index_version"] == 1
    assert body["index_state"] == "ok"
    assert body["index_operation"] == "idle"
    assert body["documents"] == 1


def test_upload_and_ingest_uses_lifecycle_reindex(client, monkeypatch, tmp_path):
    http, lifecycle = client
    monkeypatch.setattr(srv, "hipporag", object())
    monkeypatch.chdir(tmp_path)

    response = http.post(
        "/upload-and-ingest",
        files={"files": ("a.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["processed_files"] == 1
    assert lifecycle.reindexed == [[str(Path("temp_uploads") / "a.pdf")]]
    assert not (tmp_path / "temp_uploads").exists()
