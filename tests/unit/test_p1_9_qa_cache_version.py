"""P1-9: версионность QA-кэша.

Кэш пережил переиндексацию и правки кода: записи, сохранённые до фиксов
(пустой short_answer, старые ответы), отдавались как «кэшированный ответ» —
то есть неверный ответ под видом кэша. Запись должна содержать версию формата
и модель, а при чтении старые записи обязаны отбрасываться.
"""

import json

import pytest


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    import main

    path = tmp_path / "qa_cache.json"
    monkeypatch.setattr(main, "QA_CACHE_PATH", path)
    monkeypatch.setattr(main, "LLM_NAME", "test-model")
    return path


def _write(path, entries):
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def _entry(query="какая зона у овна?", model="test-model", version=None):
    entry = {
        "id": "qa_1",
        "query": query,
        "query_norm": query,
        "answer": "=== РЕЗУЛЬТАТ ПОИСКА ===\nОТВЕТ: прихожая и плита.",
        "short_answer": "прихожая и плита",
        "method": "SemFilter",
        "method_id": 22,
        "model": model,
        "timestamp": "2026-11-02 12:00:00",
        "sources": [],
        "score": 999.0,
    }
    if version is not None:
        entry["schema_version"] = version
    return entry


def test_legacy_entries_are_dropped(cache_path):
    import main

    _write(cache_path, [_entry(version=None)])
    assert main._qa_load() == []


def test_wrong_version_entries_are_dropped(cache_path):
    import main

    _write(cache_path, [_entry(version=main.QA_CACHE_VERSION - 1)])
    assert main._qa_load() == []


def test_entry_of_another_model_is_dropped(cache_path):
    import main

    _write(cache_path, [_entry(version=main.QA_CACHE_VERSION, model="другая-модель")])
    assert main._qa_load() == []


def test_current_entry_survives(cache_path):
    import main

    _write(cache_path, [_entry(version=main.QA_CACHE_VERSION)])
    loaded = main._qa_load()
    assert len(loaded) == 1
    assert loaded[0]["short_answer"] == "прихожая и плита"


def test_store_writes_version_and_purges_stale(cache_path):
    import main

    _write(
        cache_path,
        [
            _entry(query="старый вопрос", version=None),
            _entry(query="чужой моделью", model="другая-модель", version=main.QA_CACHE_VERSION),
        ],
    )

    main._qa_store("какая зона у тельца?", "ОТВЕТ: кладовка", "кладовка", "SemFilter", 22, [])

    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(stored) == 1, "устаревшие записи должны вычищаться при сохранении"
    assert stored[0]["schema_version"] == main.QA_CACHE_VERSION
    assert stored[0]["model"] == "test-model"


def test_search_qa_cache_does_not_return_stale_answer(cache_path, monkeypatch):
    import main

    _write(cache_path, [_entry(version=None)])

    def fake_semfilter(hipporag, query, top_k=50):
        return [("=== РЕЗУЛЬТАТ ПОИСКА ===\nОТВЕТ: свежий ответ.", 999.0, 0, 0, 0)]

    monkeypatch.setattr(main, "search_semantic_filter", fake_semfilter)

    result = main.search_qa_cache(object(), "какая зона у овна?")
    text = result[0][0]

    assert "свежий ответ" in text
    assert "ИЗ КЭША" not in text


def test_refusal_is_not_cached(cache_path, monkeypatch):
    import main

    _write(cache_path, [])

    def fake_semfilter(hipporag, query, top_k=50):
        return [("Ответ не найден: в источниках нет соответствия.", 0, 0, 0, 0)]

    monkeypatch.setattr(main, "search_semantic_filter", fake_semfilter)

    main.search_qa_cache(object(), "какие книги написала Максимова?")

    stored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert stored == [], "честный отказ не должен попадать в кэш"


def test_search_qa_cache_returns_current_answer(cache_path):
    import main

    _write(cache_path, [_entry(version=main.QA_CACHE_VERSION)])

    result = main.search_qa_cache(object(), "какая зона у овна?")
    text = result[0][0]

    assert "ИЗ КЭША" in text
    assert "прихожая и плита" in text
