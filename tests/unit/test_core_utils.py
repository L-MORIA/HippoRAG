"""Юнит-тесты ядра: нормализация, идемпотентность, парсеры, контракт кэша LLM."""

import json
import os

import numpy as np
import pytest

from hipporag.utils.config_utils import build_working_dir
from hipporag.utils.custom_types import QuerySolution
from hipporag.utils.llm_utils import fix_broken_generated_json
from hipporag.utils.misc_utils import (
    compute_mdhash_id,
    is_valid_entity,
    min_max_normalize,
    normalize_entity_name,
    text_processing,
)


def test_text_processing_keeps_yo_and_collapses_spaces():
    assert text_processing("Чёрный ёж — РАК (знак)!") == "чёрный ёж рак знак"
    assert text_processing("ёжик  в   тумане") == "ёжик в тумане"


def test_normalize_entity_name_matches_text_processing():
    """S-16: одна сущность должна давать одинаковый ключ в любых компонентах."""
    assert normalize_entity_name("Рак (знак)") == text_processing("Рак (знак)")


def test_is_valid_entity_filters_years_and_garbage():
    assert not is_valid_entity("2024")
    assert not is_valid_entity("2024 г.")
    assert not is_valid_entity("2 87 77")
    assert is_valid_entity("овен")
    assert is_valid_entity("число гуа")


def test_compute_mdhash_id_stable_with_prefix():
    h1 = compute_mdhash_id("abc", prefix="chunk-")
    h2 = compute_mdhash_id("abc", prefix="chunk-")
    assert h1 == h2
    assert h1.startswith("chunk-")
    assert len(h1) == len("chunk-") + 32


def test_min_max_normalize():
    arr = np.array([1.0, 2.0, 3.0])
    normalized = min_max_normalize(arr)
    assert normalized[0] == 0.0 and normalized[-1] == 1.0
    # Нулевой разброс — без деления на ноль
    constant = np.array([2.0, 2.0])
    assert (min_max_normalize(constant) == 1.0).all()


def test_normalize_scores_guards_degenerate_candidate_sets():
    """П-3: на 1–2 кандидатах min-max даёт 1.0/0.0 — это шум, а не уверенность."""
    from hipporag.utils.misc_utils import normalize_scores

    assert (normalize_scores([5.0]) == 0.5).all()
    assert (normalize_scores([0.9, 0.1]) == 0.5).all()
    # Три кандидата — обычная min-max нормировка
    assert np.allclose(normalize_scores([1.0, 2.0, 3.0]), [0.0, 0.5, 1.0])
    # Нулевой разброс при достаточном n — тоже fallback, а не единицы
    assert (normalize_scores([2.0, 2.0, 2.0]) == 0.5).all()
    # Явное значение для range == 0 переопределяет fallback
    assert (normalize_scores([2.0, 2.0, 2.0], degenerate=0.0) == 0.0).all()
    assert normalize_scores([]).size == 0
    # Старый контракт min_max_normalize сохранён
    assert (min_max_normalize(np.array([2.0, 2.0])) == 1.0).all()


def test_query_solution_is_single_class():
    """S-18: misc_utils и custom_types экспортируют один и тот же класс."""
    from hipporag.utils.misc_utils import QuerySolution as from_misc

    assert from_misc is QuerySolution
    q = QuerySolution(question="вопрос")
    q.gold_answers = ["ответ"]  # поля объединённого класса
    assert q.to_dict()["gold_answers"] == ["ответ"]
    assert q.citations == []


def test_fix_broken_generated_json():
    fixed = fix_broken_generated_json('{"triples": [["a","b","c"')
    assert json.loads(fixed) == {"triples": [["a", "b", "c"]]}
    # Валидный JSON возвращается без изменений
    assert fix_broken_generated_json('{"a": 1}') == '{"a": 1}'


def test_build_working_dir_matches_core_formula():
    """S-22: скрипты и ядро считают одинаковое имя каталога для любой модели."""
    assert build_working_dir("outputs", "BAAI/bge-m3", "intfloat/e5-base") == os.path.join(
        "outputs", "BAAI_bge-m3_intfloat_e5-base"
    )


def test_unknown_embedding_model_raises_value_error():
    """S-17: неизвестная модель — ValueError, а не AssertionError."""
    from hipporag.embedding_model import _get_embedding_model_class

    with pytest.raises(ValueError):
        _get_embedding_model_class("totally-unknown-model")


def test_extract_list_field_tolerates_brackets_in_values():
    """S-37: ']' внутри значения triple не ломает парсинг."""
    from hipporag.information_extraction.openie_openai import _extract_list_field

    resp = '{"triples": [["дом [7]", "есть", "зона"], ["марс", "управляет", "овен"]]}'
    assert len(_extract_list_field(resp, "triples")) == 2


def test_extract_list_field_handles_wrapped_json():
    from hipporag.information_extraction.openie_openai import _extract_list_field

    resp = 'Вот результат:\n{"named_entities": ["овен", "марс"]}\nКонец.'
    assert _extract_list_field(resp, "named_entities") == ["овен", "марс"]


def test_extract_list_field_garbage_returns_empty():
    from hipporag.information_extraction.openie_openai import _extract_list_field

    assert _extract_list_field("совсем не JSON", "triples") == []


class _CacheStub:
    """Стаб LLM: проверяет контракт cache_response (content, metadata, cache_hit)."""

    def __init__(self, cache_path):
        self.cache_file_name = str(cache_path)
        self.calls = 0

    from hipporag.llm.openai_gpt import cache_response as _cr

    @_cr
    def call(self, messages):
        self.calls += 1
        return "resp", {"calls": self.calls}


def test_cache_response_contract_and_hits(tmp_path):
    """S-04: декоратор возвращает 3-tuple и реально кэширует повторные вызовы."""
    stub = _CacheStub(tmp_path / "cache.sqlite")
    messages = [{"role": "user", "content": "вопрос"}]

    first = stub.call(messages)
    second = stub.call(messages)

    assert first == ("resp", {"calls": 1}, False)
    assert second == ("resp", {"calls": 1}, True)  # попадание в кэш, LLM не вызывался
    assert stub.calls == 1


def test_standard_rag_instantiates_offline(tmp_path, monkeypatch):
    """Smoke StandardRAG: конструктор отрабатывает без отладчика и без сети."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    from hipporag import StandardRAG
    from hipporag.utils.config_utils import BaseConfig

    cfg = BaseConfig(
        save_dir=str(tmp_path),
        llm_name="test-llm",
        llm_base_url="http://localhost:1234/v1",
    )
    cfg.openie_mode = "offline"  # без загрузки embedding-модели
    rag = StandardRAG(global_config=cfg)

    assert rag.ready_to_retrieve is False
    assert os.path.isdir(rag.working_dir)
