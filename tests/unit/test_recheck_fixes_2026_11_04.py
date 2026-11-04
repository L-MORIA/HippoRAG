"""Регрессионные тесты по находкам повторного аудита (2026-11-04).

Документ-основание: `docs/AUDIT_HIPPORAG_RECHECK_2026-11-04.md` (рецензия
исполнителя) + `docs/AUDIT_H1_3_VERIFICATION_2026-11-04.md` (находки Н-3…Н-7).

Что фиксируют тесты:

* **R-1** — `StandardRAG.qa()` падал с `AttributeError: 'StandardRAG' object has
  no attribute 'prompt_template_manager'`: атрибут не создавался в `__init__`.
* **R-2** — кэш эмбеддингов запросов не срабатывал: проверка смотрела и в
  под-словарь `'triple'`, который не заполняет никто, поэтому запросы
  перекодировались при каждом `retrieve()`.
* **R-3** — `transformers_llm.infer` декодировал промпт вместе с ответом и
  завышал `completion_tokens` на длину промпта.
* **R-4** — `retrieve()` печатал misleading-лог «No facts found after
  reranking» **до** переранка, то есть всегда.
* **S-3** — `outlines` не был объявлен в зависимостях, хотя
  `hipporag/llm/transformers_offline.py` импортирует его на уровне модуля.
* **S-4** — `transformers_offline` был откачен: декод промпта+ответа, `max_tokens`
  вместо `max_new_tokens`, безусловный `model_dump_json()` для строковых ответов.
* **Н-3** — деградация режима по умолчанию без cross-encoder была тихой:
  предупреждение уходило в stdout, причина не была доступна программно,
  строгого режима не было.
* **Н-4 / Н-5** — в машинно-читаемом снимке не было страничных метрик и
  происхождения каждого режима (прогоны смешивались).

Тесты не требуют индекса, моделей и сети.
"""

import inspect
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── R-1: prompt_template_manager ────────────────────────────────────────────


def _make_standard_rag(tmp_path, monkeypatch):
    """StandardRAG без загрузки моделей (тот же приём, что в smoke-тесте)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    from hipporag import StandardRAG
    from hipporag.utils.config_utils import BaseConfig

    cfg = BaseConfig(
        save_dir=str(tmp_path),
        llm_name="test-llm",
        llm_base_url="http://localhost:1234/v1",
    )
    cfg.openie_mode = "offline"  # без загрузки embedding-модели
    return StandardRAG(global_config=cfg)


def test_standard_rag_creates_prompt_template_manager(tmp_path, monkeypatch):
    """R-1: в `__init__` создаётся менеджер шаблонов промптов."""
    from hipporag.prompts.prompt_template_manager import PromptTemplateManager

    rag = _make_standard_rag(tmp_path, monkeypatch)

    assert isinstance(rag.prompt_template_manager, PromptTemplateManager)
    # набор шаблонов загрузился, а не остался пустым заглушкой
    assert rag.prompt_template_manager.templates


def test_standard_rag_qa_does_not_raise_attribute_error(tmp_path, monkeypatch):
    """R-1: `qa()` доходит до шаблонов и LLM, а не падает с AttributeError."""
    from hipporag.utils.custom_types import QuerySolution

    rag = _make_standard_rag(tmp_path, monkeypatch)

    class _StubLLM:
        def __init__(self):
            self.calls = 0

        def infer(self, messages, **kwargs):
            self.calls += 1
            return "Answer: 42", {"prompt_tokens": 1, "completion_tokens": 1}, False

    rag.llm_model = _StubLLM()
    rag.global_config.qa_top_k = 2
    # R-1 проверяет отсутствие AttributeError, а не заземление: старый ответ
    # без ссылок должен дойти до результата, поэтому strict выключаем.
    rag.global_config.qa_strict_grounding = False

    solutions = [QuerySolution(question="Что такое?", docs=["документ один"])]
    out_solutions, messages, metadata = rag.qa(solutions)

    assert rag.llm_model.calls == 1
    assert out_solutions[0].answer == "42"
    assert messages and metadata


# ── R-2: кэш эмбеддингов запросов ───────────────────────────────────────────


class _CountingEmbedder:
    """Считает вызовы `batch_encode` — им тестируется работа кэша."""

    def __init__(self):
        self.calls = 0
        self.batches = []

    def batch_encode(self, texts, instruction=None, norm=True):
        self.calls += 1
        self.batches.append(list(texts))
        return np.zeros((len(texts), 4), dtype=np.float32)


def _standard_rag_with_embeddings():
    """Экземпляр без `__init__` — для проверки одного метода."""
    from hipporag import StandardRAG

    rag = StandardRAG.__new__(StandardRAG)
    rag.query_to_embedding = {"triple": {}, "passage": {}}
    rag.embedding_model = _CountingEmbedder()
    return rag


def test_query_embeddings_are_not_reencoded_on_second_call():
    """R-2: повторный запрос не перекодируется — кэш работает."""
    rag = _standard_rag_with_embeddings()

    rag.get_query_embeddings(["какой знак"])
    assert rag.embedding_model.calls == 1

    rag.get_query_embeddings(["какой знак"])
    assert rag.embedding_model.calls == 1, "запрос перекодирован повторно"


def test_query_embeddings_never_touch_triple_cache():
    """R-2: под-словарь 'triple' не участвует в проверке и не заполняется."""
    from hipporag.utils.custom_types import QuerySolution

    rag = _standard_rag_with_embeddings()

    rag.get_query_embeddings([QuerySolution(question="вопрос из объекта")])

    assert rag.query_to_embedding["triple"] == {}, "'triple' кто-то заполнил"
    assert "вопрос из объекта" in rag.query_to_embedding["passage"]
    # повтор с тем же QuerySolution — кэш срабатывает
    rag.get_query_embeddings([QuerySolution(question="вопрос из объекта")])
    assert rag.embedding_model.calls == 1


# ── R-4: misleading-лог ────────────────────────────────────────────────────


def test_retrieve_has_no_misleading_log():
    """R-4: лог «No facts found after reranking» удалён из `retrieve()`."""
    from hipporag import StandardRAG

    source = inspect.getsource(StandardRAG.retrieve)
    assert "No facts found after reranking" not in source


# ── R-3: transformers_llm декодирует только завершение ─────────────────────


class _StubTokenizer:
    """Токенизатор: промпт — 4 токена, декод печатает номера токенов."""

    def apply_chat_template(self, **kwargs):
        return "ПРОМПТ"

    def encode(self, text, return_tensors=None):
        import torch

        return torch.tensor([[1, 2, 3, 4]])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


class _StubModel:
    """Модель: generate() возвращает промпт (4 токена) + ответ (2 токена)."""

    device = "cpu"

    def __init__(self):
        self.max_new_tokens = None

    def generate(self, inputs, max_new_tokens=None):
        import torch

        self.max_new_tokens = max_new_tokens
        return torch.tensor([[1, 2, 3, 4, 5, 6]])


class _StubCache:
    def read(self, params):
        return None

    def write(self, params, message, metadata):
        return None


class _StubLLMConfig:
    generate_params = {"n": 1, "temperature": 0.0}


class _StubGlobalConfig:
    llm_name = "test-llm"


def _transformers_llm_stub():
    from hipporag.llm.transformers_llm import TransformersLLM

    llm = TransformersLLM.__new__(TransformersLLM)
    llm.cache = _StubCache()
    llm.model = _StubModel()
    llm.tokenizer = _StubTokenizer()
    llm.llm_config = _StubLLMConfig()
    llm.global_config = _StubGlobalConfig()
    return llm


def test_transformers_llm_decodes_only_completion():
    """R-3: в ответ попадают только токены завершения, без промпта."""
    llm = _transformers_llm_stub()

    message, metadata, cached = llm.infer([{"role": "user", "content": "вопрос"}])

    assert message == "t5 t6", "в ответ попал промпт"
    assert cached is False


def test_transformers_llm_reports_honest_token_counts():
    """R-3: `completion_tokens` считает только завершение (2, а не 6)."""
    llm = _transformers_llm_stub()

    _, metadata, _ = llm.infer([{"role": "user", "content": "вопрос"}])

    assert metadata["prompt_tokens"] == 4
    assert metadata["completion_tokens"] == 2


# ── S-3 / S-4: объявленная зависимость и откаченный offline-бэкенд ─────────


def test_outlines_is_declared_in_offline_extra():
    """S-3: `outlines` объявлен в extra `offline` рядом с vllm."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)

    offline = pyproject["project"]["optional-dependencies"]["offline"]
    assert any(dep.startswith("outlines") for dep in offline), offline
    assert any(dep.startswith("vllm") for dep in offline), offline


def test_transformers_offline_decodes_only_completion():
    """S-4: offline-бэкенд декодирует только сгенерированную часть.

    Проверка по исходнику: модуль импортирует `outlines` на уровне модуля,
    а в extra `offline` он не входит — в CI пакет не установлен, и импортом
    модуль не проверить.
    """
    source = (PROJECT_ROOT / "src" / "hipporag" / "llm" / "transformers_offline.py").read_text(
        encoding="utf-8"
    )
    assert 'transformers_output[0]["generated_text"]' not in source
    # декодируется только сгенерированная часть — срез по длине входа
    assert "input_ids.shape[1] :" in source, "декодируется весь вывод, а не только завершение"
    # generate() получает токенизированные входы и max_new_tokens, а не строки и max_tokens
    assert "self.model.generate(**encoded, max_new_tokens=max_tokens)" in source
    assert "self.model.generate(all_prompt_texts" not in source


def test_transformers_offline_serializes_only_guided_objects():
    """S-4: `model_dump_json` вызывается только для guided-объектов."""
    source = (PROJECT_ROOT / "src" / "hipporag" / "llm" / "transformers_offline.py").read_text(
        encoding="utf-8"
    )
    assert 'hasattr(completion, "model_dump_json")' in source
    assert "completion.model_dump_json() for completion in transformers_outputs" not in source


# ── Н-3: деградация режима по умолчанию без cross-encoder ──────────────────


@pytest.fixture
def ru_rerank(monkeypatch):
    """`_rerank_with_crossencoder_ru` с недоступной моделью."""
    from astro_app import rerankers

    def _boom():
        raise OSError("модель недоступна")

    monkeypatch.setattr(rerankers, "_get_crossencoder_ru", _boom)
    rerankers.reset_crossencoder_ru_state()
    yield rerankers
    rerankers.reset_crossencoder_ru_state()


def test_rerank_degradation_is_explicit_on_stderr(ru_rerank, capsys):
    """Н-3: предупреждение уходит в stderr и прямо говорит, что переранка не было."""
    candidates = [("текст", 0.9, 0.8, 0.1, 0.0)]

    result = ru_rerank._rerank_with_crossencoder_ru("вопрос", candidates)

    captured = capsys.readouterr()
    assert result == candidates  # деградация: выдача не переранжана
    assert "ПЕРЕРАНК НЕ ВЫПОЛНЕН" in captured.err
    assert "ПЕРЕРАНК НЕ ВЫПОЛНЕН" not in captured.out, "предупреждение ушло в stdout"
    assert ru_rerank.crossencoder_ru_unavailable_reason()


def test_rerank_degradation_reason_is_available(ru_rerank):
    """Н-3: причина деградации доступна программно, а не только в тексте."""
    assert ru_rerank.crossencoder_ru_unavailable_reason() is None

    ru_rerank._rerank_with_crossencoder_ru("вопрос", [("текст", 0.9, 0.8, 0.1, 0.0)])

    reason = ru_rerank.crossencoder_ru_unavailable_reason()
    assert reason and "модель недоступна" in reason


def test_rerank_strict_mode_raises_instead_of_degrading(ru_rerank, monkeypatch):
    """Н-3: строгий режим (`HIPPORAG_REQUIRE_CROSSENCODER=1`) падает, а не деградирует."""
    monkeypatch.setenv(ru_rerank.STRICT_ENV_VAR, "1")

    with pytest.raises(RuntimeError, match="cross-encoder"):
        ru_rerank._rerank_with_crossencoder_ru("вопрос", [("текст", 0.9, 0.8, 0.1, 0.0)])


def test_rerank_strict_mode_is_off_by_default(ru_rerank, monkeypatch):
    """Н-3: без явного флага поведение прежнее — деградация с предупреждением."""
    monkeypatch.delenv(ru_rerank.STRICT_ENV_VAR, raising=False)

    assert ru_rerank.strict_crossencoder() is False
    result = ru_rerank._rerank_with_crossencoder_ru("вопрос", [("текст", 0.9, 0.8, 0.1, 0.0)])
    assert result == [("текст", 0.9, 0.8, 0.1, 0.0)]


# ── Н-4 / Н-5: страничные метрики и происхождение в снимке ─────────────────


def test_summarize_mode_reports_page_metrics():
    """Н-4: сводка режима отдаёт страничные recall/MRR/nDCG."""
    from astro_app.evaluation import summarize_mode

    rows = [
        {
            "retrieval": True,
            "page_expected": True,
            "doc_rank": 1,
            "page_rank": 2,
            "latency_ms": 10.0,
            "llm_ms": 0.0,
            "n_candidates": 5,
        }
    ]

    summary = summarize_mode(rows, mode_id="5", label="BM25+CE-ru")

    assert summary["recall@k_page"] == 1.0
    assert summary["mrr@k_page"] == 0.5
    assert summary["ndcg@k_page"] is not None


def test_repo_baseline_has_page_metrics_and_provenance():
    """Н-4/Н-5: в снимке есть и страничные метрики, и происхождение режима."""
    from astro_app import performance as perf

    baseline = perf.load_baseline()
    assert baseline, "eval/dataset/latency_baseline.json должен существоваться"

    for mode_id, entry in baseline["modes"].items():
        assert entry.get("measured_at"), f"у режима {mode_id} нет measured_at"
        assert entry.get("source"), f"у режима {mode_id} нет source"
        for key in ("recall@k_page", "mrr@k_page", "ndcg@k_page"):
            assert key in entry, f"у режима {mode_id} нет {key}"

    # страничные метрики режима 5 — рабочая метрика корпуса
    assert baseline["modes"]["5"]["recall@k_page"] == 0.65
    assert baseline["modes"]["5"]["mrr@k_page"] == 0.52
    assert baseline["modes"]["2"]["recall@k_page"] == 0.6


def test_merge_baseline_keeps_per_mode_provenance():
    """Н-5: слияние сохраняет происхождение каждого режима."""
    from astro_app import performance as perf

    existing = {
        "modes": {
            "1": {
                "label": "Multi-vector",
                "p95_ms": 12_800,
                "measured_at": "2026-11-03",
                "source": "outputs/reports/eval_after_llm.json",
            }
        }
    }
    fresh = {
        "modes": {
            "5": {
                "label": "BM25+CE-ru",
                "p95_ms": 225,
                "measured_at": "2026-11-04",
                "source": "outputs/reports/eval_mode5.json",
            }
        }
    }

    merged = perf.merge_baseline(existing, fresh)

    assert merged["modes"]["1"]["source"] == "outputs/reports/eval_after_llm.json"
    assert merged["modes"]["1"]["measured_at"] == "2026-11-03"
    assert merged["modes"]["5"]["source"] == "outputs/reports/eval_mode5.json"
    assert merged["modes"]["5"]["measured_at"] == "2026-11-04"
    # время слияния проставлено в записи режима
    assert merged["modes"]["5"]["merged_at"]
    # верхний уровень честно помечен как «последний прогон»
    assert "ПОСЛЕДНИЙ прогон" in merged["top_level_scope"]


def test_build_baseline_writes_page_metrics_and_source():
    """Н-4/Н-5: сборщик снимка пишет страничные метрики и источник прогона."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_budget", PROJECT_ROOT / "scripts" / "benchmark_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = {
        "meta": {"index": {"path": "outputs/x"}, "dataset": {"n_questions": 40}},
        "summaries": [
            {
                "mode": {"id": "5", "label": "BM25+CE-ru"},
                "latency_ms": {"p50": 197, "p95": 225},
                "retrieval_ms": {"p50": 197, "p95": 225},
                "recall@k_doc": 0.975,
                "recall@k_page": 0.65,
                "mrr@k_doc": 0.902,
                "mrr@k_page": 0.52,
                "ndcg@k_doc": 0.921,
                "ndcg@k_page": 0.61,
            }
        ],
    }

    baseline = module.build_baseline(report, source="outputs/reports/eval_mode5.json")

    entry = baseline["modes"]["5"]
    assert entry["recall@k_page"] == 0.65
    assert entry["mrr@k_page"] == 0.52
    assert entry["ndcg@k_page"] == 0.61
    assert entry["source"] == "outputs/reports/eval_mode5.json"
    assert entry["measured_at"]


def test_repo_baseline_file_is_utf8_lf_with_final_newline():
    """Н-7: снимок в git — UTF-8 без BOM, переводы строк LF, файл заканчивается LF.

    Проверяется **содержимое blob'а в git**, а не файл в рабочем дереве:
    на windows-раннере с `core.autocrlf=true` checkout переводит LF в CRLF,
    и проверка байтов рабочего дерева ловила бы не дефект репозитория, а
    особенность checkout'а (красный Test Suite (windows-latest), 2026-11-04).
    Если git недоступен (дерево не под git) — проверяется рабочее дерево с
    нормализацией переводов строк.
    """
    import subprocess

    rel = "eval/dataset/latency_baseline.json"
    raw = None
    try:
        proc = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "show", f"HEAD:{rel}"],
            capture_output=True,
            check=True,
        )
        raw = proc.stdout
    except (OSError, subprocess.CalledProcessError):
        raw = None

    if raw is None:
        raw = (PROJECT_ROOT / rel).read_bytes().replace(b"\r\n", b"\n")

    assert not raw.startswith(b"\xef\xbb\xbf"), "в файле BOM"
    assert b"\r\n" not in raw, "в содержимом репозитория CRLF"
    assert raw.endswith(b"\n"), "нет завершающего перевода строки"
    json.loads(raw.decode("utf-8"))
