"""H0-8: контракт версий «индекс ↔ код ↔ кэш ↔ package-data».

Три версии объявлены в одном месте (``astro_app/versions.py``). Несовпадение
не должно приводить к тихой деградации: устаревший индекс — предупреждение при
загрузке, устаревшая запись QA-кэша — не отдаётся и о ней сообщается,
несовпадение package-data — понятная ошибка с указанием, что делать.
"""

from __future__ import annotations

import json
import logging

import pytest


def _versions():
    from astro_app import versions

    return versions


# ── Одно место объявления версий ────────────────────────────────────────────


def test_versions_are_declared_in_one_module():
    from astro_app import qa_cache, versions

    assert isinstance(versions.INDEX_FORMAT_VERSION, int)
    assert isinstance(versions.QA_CACHE_VERSION, int)
    assert isinstance(versions.PACKAGE_DATA_VERSION, int)
    assert (
        versions.QA_CACHE_VERSION == qa_cache.QA_CACHE_VERSION
    ), "QA-кэш должен использовать общую версию, а не собственное число"


# ── Манифест индекса ────────────────────────────────────────────────────────


def test_manifest_roundtrip(tmp_path):
    versions = _versions()

    path = versions.write_index_manifest(
        tmp_path, llm_name="test-llm", embedding_model="test-embed", counts={"nodes": 3}
    )

    assert path.name == versions.INDEX_MANIFEST_NAME
    manifest = versions.read_index_manifest(tmp_path)
    assert manifest["index_format_version"] == versions.INDEX_FORMAT_VERSION
    assert manifest["llm_name"] == "test-llm"
    assert manifest["embedding_model"] == "test-embed"
    assert manifest["counts"] == {"nodes": 3}


def test_missing_index_is_not_a_problem(tmp_path):
    versions = _versions()

    status = versions.check_index(tmp_path)

    assert status.state == versions.INDEX_MISSING
    assert not status.is_problem
    assert status.message == ""


def test_index_without_manifest_warns_and_suggests_adoption(tmp_path):
    versions = _versions()
    (tmp_path / versions.INDEX_GRAPH_FILE).write_bytes(b"graph")

    status = versions.check_index(tmp_path)

    assert status.state == versions.INDEX_NO_MANIFEST
    assert status.is_problem
    assert "--adopt" in status.message
    assert str(versions.INDEX_FORMAT_VERSION) in status.message


def test_stale_index_warns_with_both_versions(tmp_path):
    versions = _versions()
    (tmp_path / versions.INDEX_GRAPH_FILE).write_bytes(b"graph")
    old_version = versions.INDEX_FORMAT_VERSION - 1
    (tmp_path / versions.INDEX_MANIFEST_NAME).write_text(
        json.dumps({"index_format_version": old_version}), encoding="utf-8"
    )

    status = versions.check_index(tmp_path)

    assert status.state == versions.INDEX_STALE
    assert status.is_problem
    assert status.found_version == old_version
    assert str(old_version) in status.message
    assert str(versions.INDEX_FORMAT_VERSION) in status.message
    assert "деград" in status.message


def test_current_index_is_ok(tmp_path):
    versions = _versions()
    (tmp_path / versions.INDEX_GRAPH_FILE).write_bytes(b"graph")
    versions.write_index_manifest(tmp_path, llm_name="llm", embedding_model="embed")

    status = versions.check_index(tmp_path)

    assert status.state == versions.INDEX_OK
    assert not status.is_problem
    assert status.found_version == versions.INDEX_FORMAT_VERSION


def test_corrupt_manifest_is_reported(tmp_path):
    versions = _versions()
    (tmp_path / versions.INDEX_GRAPH_FILE).write_bytes(b"graph")
    (tmp_path / versions.INDEX_MANIFEST_NAME).write_text("{not json", encoding="utf-8")

    status = versions.check_index(tmp_path)

    assert status.state == versions.INDEX_CORRUPT
    assert status.is_problem


def test_explicit_graph_path_is_honoured(tmp_path):
    versions = _versions()
    graph = tmp_path / "custom_graph.pickle"
    graph.write_bytes(b"graph")

    status = versions.check_index(tmp_path, graph_path=graph)

    assert status.state == versions.INDEX_NO_MANIFEST


# ── Интеграция с HippoRAG: запись при сохранении и проверка при загрузке ────


def _bare_hipporag(tmp_path, graph_path=None):
    versions = _versions()
    from types import SimpleNamespace

    from hipporag import HippoRAG

    instance = object.__new__(HippoRAG)
    instance.working_dir = str(tmp_path)
    instance._graph_pickle_filename = str(graph_path or (tmp_path / "graph.pickle"))
    instance.global_config = SimpleNamespace(llm_name="test-llm", embedding_model_name="test-embed")
    # __init__ делает это до загрузки графа, поэтому повторяем порядок
    instance.index_status = versions.check_index(
        instance.working_dir, graph_path=instance._graph_pickle_filename
    )
    return instance


class _FakeGraph:
    def __init__(self, nodes, edges):
        self._nodes = nodes
        self._edges = edges
        self.written_to = None

    def vs(self):
        return list(range(self._nodes))

    def es(self):
        return list(range(self._edges))

    def write_pickle(self, path):
        self.written_to = path
        with open(path, "wb") as stream:
            stream.write(b"graph")


class _FakeStore:
    def __init__(self, ids):
        self._ids = ids

    def get_all_ids(self):
        return list(self._ids)


def test_save_igraph_writes_manifest(tmp_path):
    versions = _versions()
    hipporag = _bare_hipporag(tmp_path)
    hipporag.graph = _FakeGraph(nodes=3, edges=2)
    hipporag.chunk_embedding_store = _FakeStore(["chunk-a", "chunk-b"])
    hipporag.entity_embedding_store = _FakeStore(["entity-a"])
    hipporag.fact_embedding_store = _FakeStore([])

    hipporag.save_igraph()

    manifest = versions.read_index_manifest(tmp_path)
    assert manifest is not None, "сохранение графа должно писать манифест формата"
    assert manifest["index_format_version"] == versions.INDEX_FORMAT_VERSION
    assert manifest["counts"] == {
        "nodes": 3,
        "edges": 2,
        "chunks": 2,
        "entities": 1,
        "facts": 0,
    }
    assert hipporag.index_status.state == versions.INDEX_MISSING, "до сохранения индекса не было"


def test_load_of_stale_index_warns(tmp_path, caplog):
    versions = _versions()
    (tmp_path / "graph.pickle").write_bytes(b"graph")
    (tmp_path / versions.INDEX_MANIFEST_NAME).write_text(
        json.dumps({"index_format_version": versions.INDEX_FORMAT_VERSION + 1}),
        encoding="utf-8",
    )
    hipporag = _bare_hipporag(tmp_path)

    with caplog.at_level(logging.WARNING):
        status = hipporag._check_index_contract()

    assert status.state == versions.INDEX_STALE
    assert hipporag.index_status == status
    assert "другим форматом" in caplog.text


# ── QA-кэш ──────────────────────────────────────────────────────────────────


@pytest.fixture
def qa_cache_path(tmp_path, monkeypatch):
    from astro_app import config, qa_cache

    path = tmp_path / "qa_cache.json"
    monkeypatch.setattr(qa_cache, "QA_CACHE_PATH", path)
    monkeypatch.setattr(config, "LLM_NAME", "test-model")
    qa_cache._WARNED_STALE.clear()
    return path


def _qa_entry(version, model="test-model", query_norm="какая зона у овна?"):
    return {
        "id": "qa_1",
        "schema_version": version,
        "query": "какая зона у овна?",
        "query_norm": query_norm,
        "answer": "ОТВЕТ: прихожая",
        "short_answer": "прихожая",
        "method": "SemFilter",
        "method_id": 22,
        "model": model,
        "timestamp": "2026-11-02 12:00:00",
        "sources": [],
        "score": 999.0,
    }


def test_old_qa_cache_entry_is_not_returned_and_is_reported(qa_cache_path, capsys):
    from astro_app import qa_cache

    qa_cache_path.write_text(
        json.dumps([_qa_entry(qa_cache.QA_CACHE_VERSION - 1)]), encoding="utf-8"
    )

    assert qa_cache._qa_load() == []

    output = capsys.readouterr().out
    assert "QA-кэш" in output
    assert "1 из 1" in output
    assert str(qa_cache.QA_CACHE_VERSION) in output


def test_qa_cache_warning_is_not_repeated_for_unchanged_file(qa_cache_path, capsys):
    from astro_app import qa_cache

    qa_cache_path.write_text(
        json.dumps([_qa_entry(qa_cache.QA_CACHE_VERSION - 1)]), encoding="utf-8"
    )

    qa_cache._qa_load()
    qa_cache._qa_load()

    assert capsys.readouterr().out.count("QA-кэш") == 1


def test_current_qa_cache_entry_survives_without_warning(qa_cache_path, capsys):
    from astro_app import qa_cache

    qa_cache_path.write_text(json.dumps([_qa_entry(qa_cache.QA_CACHE_VERSION)]), encoding="utf-8")

    loaded = qa_cache._qa_load()

    assert len(loaded) == 1
    assert capsys.readouterr().out == ""


# ── Package-data ────────────────────────────────────────────────────────────


def test_shipped_package_data_matches_code():
    versions = _versions()

    manifest = versions.check_package_data()

    assert manifest["package_data_version"] == versions.PACKAGE_DATA_VERSION
    assert "synonyms.json" in manifest["files"]


def test_shipped_manifest_matches_recomputed():
    versions = _versions()

    shipped = json.loads(versions.PACKAGE_DATA_MANIFEST.read_text(encoding="utf-8"))

    assert shipped == versions.build_package_data_manifest(), (
        "манифест package-data устарел: пересоберите "
        "python scripts/update_package_data_manifest.py"
    )


def test_synonyms_are_loadable():
    from astro_app.synonyms import load_synonyms

    assert len(load_synonyms()) == 66


def test_missing_package_data_manifest_is_a_clear_error(tmp_path, monkeypatch):
    versions = _versions()
    monkeypatch.setattr(versions, "PACKAGE_DATA_MANIFEST", tmp_path / "data_manifest.json")

    with pytest.raises(versions.PackageDataError) as excinfo:
        versions.check_package_data()

    assert "Не найден манифест" in str(excinfo.value)


def test_package_data_version_mismatch_is_a_clear_error(monkeypatch):
    versions = _versions()
    expected = versions.PACKAGE_DATA_VERSION
    monkeypatch.setattr(versions, "PACKAGE_DATA_VERSION", expected + 1)

    with pytest.raises(versions.PackageDataError) as excinfo:
        versions.check_package_data()

    message = str(excinfo.value)
    assert "несовместимы" in message
    assert repr(expected) in message, "в ошибке должна быть версия из манифеста"
    assert str(expected + 1) in message, "в ошибке должна быть ожидаемая версия"
    assert "update_package_data_manifest" in message


def _tmp_data_dir(tmp_path, monkeypatch, files, version_offset=0):
    versions = _versions()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name, content in files.items():
        (data_dir / name).write_text(json.dumps(content), encoding="utf-8")

    manifest = {
        "package_data_version": versions.PACKAGE_DATA_VERSION + version_offset,
        "files": {name: {"sha256": "0" * 64} for name in files},
    }
    manifest_path = data_dir / versions.PACKAGE_DATA_MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(versions, "PACKAGE_DATA_MANIFEST", manifest_path)
    return data_dir


def test_package_data_missing_file_is_a_clear_error(tmp_path, monkeypatch):
    versions = _versions()
    _tmp_data_dir(tmp_path, monkeypatch, files={"synonyms.json": {"a": ["b"]}})
    (tmp_path / "data" / "synonyms.json").unlink()

    with pytest.raises(versions.PackageDataError) as excinfo:
        versions.check_package_data()

    assert "synonyms.json" in str(excinfo.value)
    assert "отсутствует" in str(excinfo.value)


def test_package_data_checksum_mismatch_is_a_clear_error(tmp_path, monkeypatch):
    versions = _versions()
    _tmp_data_dir(tmp_path, monkeypatch, files={"synonyms.json": {"a": ["b"]}})

    with pytest.raises(versions.PackageDataError) as excinfo:
        versions.check_package_data()

    message = str(excinfo.value)
    assert "не совпадает с манифестом" in message
    assert "0" * 64 in message
