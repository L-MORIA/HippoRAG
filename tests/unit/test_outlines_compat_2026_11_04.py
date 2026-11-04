"""Регрессионные тесты по декларации и совместимости `outlines` (раунд 8).

Документ-основание: `docs/AUDIT_OUTLINES_DEPENDENCY_2026-11-04.md`.

Проверяется дефект, найденный на границе правки S-3 (раунд 6): декларация
`outlines>=0.0.46` в extra `offline` была неработоспособна.

* `outlines` 0.0.46 требует `numpy<2.0.0` — установка откатывает numpy и ломает
  `pyarrow`/`transformers` в остальном окружении проекта (воспроизведено).
* `outlines` 0.1.x/0.2.x тянут Rust-расширение `outlines_core` и используют
  legacy-API `outlines.generate.json`; 0.2.x ограничен Python < 3.13.
* `outlines` 1.x использует современный API `outlines.Generator`; legacy-API
  в нём удалён, а сам пакет официально не выпускается для Python 3.14+.

Тесты не требуют пакета `outlines`: используются подмены модулей. Отдельный
тест с настоящим `outlines` и настоящей мини-моделью пропускается, если
пакет или модель недоступны.
"""

import json
import sys
import tomllib
import types
from pathlib import Path

import pydantic
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


# ── декларация зависимости ─────────────────────────────────────────────────


def _offline_requirements():
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as fh:
        pyproject = tomllib.load(fh)
    return pyproject["project"]["optional-dependencies"]


def _outlines_specifier(extras):
    """Спецификатор версии `outlines` из extra (без маркеров окружения)."""
    for dep in extras:
        if dep.split(";")[0].strip().startswith("outlines"):
            return dep
    return None


def test_outlines_declared_in_offline_extra():
    """`outlines` объявлен в extra `offline` рядом с vllm."""
    extras = _offline_requirements()
    offline = extras["offline"]

    assert any(dep.startswith("outlines") for dep in offline), offline
    assert any(dep.startswith("vllm") for dep in offline), offline


def test_outlines_requirement_excludes_numpy2_incompatible_branch():
    """Декларация не должна разрешать ветку `outlines` 0.0.x (numpy<2).

    Именно эта ветна ломает остальное окружение проекта: `numpy<2.0.0` не
    совместим с `pyarrow`/`transformers` (аудит 04.11.2026, раунд 8).
    """
    from packaging.requirements import Requirement

    spec = _outlines_specifier(_offline_requirements()["offline"])
    assert spec is not None, "outlines не объявлен в extra `offline`"

    requirement = Requirement(spec.split(";")[0].strip())
    assert not requirement.specifier.contains("0.0.46"), spec
    assert not requirement.specifier.contains("0.1.5"), spec
    assert requirement.specifier.contains("1.3.3"), spec


def test_outlines_requirement_limited_to_supported_python():
    """Декларация `outlines` ограничена Python < 3.14 (пакет там не выпускается)."""
    spec = _outlines_specifier(_offline_requirements()["offline"])
    assert spec is not None
    assert "python_version" in spec, spec
    assert "3.14" in spec, spec


def test_outlines_declared_in_dev_extra_for_ci():
    """`outlines` объявлен и в dev — иначе CI проверяет guided-путь стабом."""
    dev = _offline_requirements()["dev"]
    assert _outlines_specifier(dev) is not None, dev


# ── модуль бэкенда не обязан зависеть от `outlines` на уровне модуля ────────


def test_transformers_offline_has_no_module_level_outlines_import():
    """`outlines` подключается лениво: иначе модуль не импортируется без пакета."""
    source = (PROJECT_ROOT / "src" / "hipporag" / "llm" / "transformers_offline.py").read_text(
        encoding="utf-8"
    )
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith("import outlines") or line.startswith("from outlines")
    ]
    assert module_level == [], module_level


# ── подмены `outlines` ─────────────────────────────────────────────────────


class _Schema(pydantic.BaseModel):
    """Схема guided-декодирования для подмен."""

    code: str = pydantic.Field(min_length=2, max_length=4)


class _FakeOutlinesModel:
    """Подмена `outlines.models.Transformers`."""

    created = []

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        _FakeOutlinesModel.created.append(self)


class _FakeGenerator:
    """Подмена `outlines.Generator` (современный API)."""

    calls = []

    def __init__(self, model, output_type):
        self.model = model
        self.output_type = output_type

    def batch(self, prompts, **kwargs):
        _FakeGenerator.calls.append({"prompts": list(prompts), "kwargs": dict(kwargs)})
        return [json.dumps({"code": "ab"}) for _ in prompts]


class _FakeLegacyGenerator:
    """Подмена legacy-генератора `outlines.generate`."""

    def __init__(self, factory):
        self.factory = factory
        self.calls = []

    def __call__(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), "kwargs": dict(kwargs)})
        return [self.factory(prompt) for prompt in prompts]


def _install_modern_outlines(monkeypatch):
    """Подменяет `outlines` современным API (>=1.0)."""
    outlines = types.ModuleType("outlines")
    outlines.Generator = _FakeGenerator
    models = types.ModuleType("outlines.models")
    models.Transformers = _FakeOutlinesModel
    outlines.models = models
    monkeypatch.setitem(sys.modules, "outlines", outlines)
    monkeypatch.setitem(sys.modules, "outlines.models", models)


def _install_legacy_outlines(monkeypatch, factory):
    """Подменяет `outlines` legacy-API (<1.0) с заданным «выводом» модели."""
    outlines = types.ModuleType("outlines")
    generate = types.ModuleType("outlines.generate")
    generate.json = lambda model, schema_object, **kwargs: _FakeLegacyGenerator(factory)
    generate.regex = lambda model, pattern, **kwargs: _FakeLegacyGenerator(factory)
    models = types.ModuleType("outlines.models")
    models.Transformers = _FakeOutlinesModel
    outlines.generate = generate
    outlines.models = models
    monkeypatch.setitem(sys.modules, "outlines", outlines)
    monkeypatch.setitem(sys.modules, "outlines.generate", generate)
    monkeypatch.setitem(sys.modules, "outlines.models", models)

    # `outlines_core` тоже подменяем — иначе ветка выбора зависит от того,
    # установлен ли пакет в окружении (в CI `outlines` есть в dev-extra).
    core = types.ModuleType("outlines_core")
    fsm = types.ModuleType("outlines_core.fsm")
    json_schema = types.ModuleType("outlines_core.fsm.json_schema")
    json_schema.build_regex_from_schema = lambda schema, whitespace_pattern=None: "REGEX"
    core.fsm = fsm
    fsm.json_schema = json_schema
    monkeypatch.setitem(sys.modules, "outlines_core", core)
    monkeypatch.setitem(sys.modules, "outlines_core.fsm", fsm)
    monkeypatch.setitem(sys.modules, "outlines_core.fsm.json_schema", json_schema)


class _StubTokenizer:
    """Токенизатор: интересует только `padding_side`."""

    def __init__(self):
        self.padding_side = "right"
        self.seen_padding_side = []


@pytest.fixture
def stub_tokenizer():
    return _StubTokenizer()


# ── современный API ────────────────────────────────────────────────────────


def test_modern_api_uses_generator_and_max_new_tokens(monkeypatch, stub_tokenizer):
    """`outlines>=1.0`: используется `Generator`, лимит токенов — `max_new_tokens`."""
    _install_modern_outlines(monkeypatch)
    _FakeGenerator.calls.clear()

    from hipporag.llm.outlines_compat import build_guided_generator

    generator = build_guided_generator("model", stub_tokenizer, _Schema)
    results = generator(["первый промпт"], 128)

    assert len(_FakeGenerator.calls) == 1
    call = _FakeGenerator.calls[0]
    assert call["prompts"] == ["первый промпт"]
    assert call["kwargs"] == {"max_new_tokens": 128}, "лимит токенов передан неверно"
    assert all(isinstance(item, _Schema) for item in results)


def test_modern_api_uses_left_padding_and_restores_it(monkeypatch, stub_tokenizer):
    """Во время генерации — left-padding, после — исходное значение."""
    _install_modern_outlines(monkeypatch)

    from hipporag.llm.outlines_compat import build_guided_generator

    generator = build_guided_generator("model", stub_tokenizer, _Schema)
    generator(["промпт"], 16)

    assert stub_tokenizer.padding_side == "right", "padding_side не восстановлен"


def test_generator_construction_does_not_change_tokenizer_padding_side(monkeypatch, stub_tokenizer):
    """`outlines` сам переводит токенизатор на left-padding — бэкенд это отменяет."""
    _install_modern_outlines(monkeypatch)

    from hipporag.llm.outlines_compat import build_guided_generator

    build_guided_generator("model", stub_tokenizer, _Schema)

    assert stub_tokenizer.padding_side == "right"


# ── legacy API ─────────────────────────────────────────────────────────────


def test_legacy_api_accepts_max_tokens_kwarg(monkeypatch, stub_tokenizer):
    """`outlines<1.0`: legacy-API принимает лимит токенов как `max_tokens`."""
    _install_legacy_outlines(monkeypatch, lambda prompt: _Schema(code="ab"))

    from hipporag.llm.outlines_compat import build_guided_generator

    generator = build_guided_generator("model", stub_tokenizer, _Schema)
    results = generator(["промпт"], 64)

    assert len(results) == 1
    assert isinstance(results[0], _Schema)


def test_legacy_api_survives_truncated_json(monkeypatch, stub_tokenizer):
    """Оборванная генерация не роняет батч целиком.

    Legacy-`outlines` (`generate.json`) парсит результат сам и бросает
    `pydantic.ValidationError` на обрыве — весь батч терялся. Разбор перенесён
    в наш код, поэтому один негодный ответ не убивает остальные.
    """
    _install_legacy_outlines(monkeypatch, lambda prompt: '{"code": ')

    from hipporag.llm.outlines_compat import build_guided_generator

    generator = build_guided_generator("model", stub_tokenizer, _Schema)
    results = generator(["промпт"], 64)

    assert results == ['{"code": '], "негодный ответ должен возвращаться как есть"


# ── отсутствие пакета ──────────────────────────────────────────────────────


def test_missing_outlines_raises_clear_error(monkeypatch, stub_tokenizer):
    """Без `outlines` — внятная ошибка с подсказкой по установке."""
    monkeypatch.setitem(sys.modules, "outlines", None)

    from hipporag.llm.outlines_compat import OutlinesUnavailableError, build_guided_generator

    with pytest.raises(OutlinesUnavailableError) as excinfo:
        build_guided_generator("model", stub_tokenizer, _Schema)

    message = str(excinfo.value)
    assert "outlines" in message
    assert "pip install" in message


def test_broken_outlines_import_is_reported(monkeypatch, stub_tokenizer):
    """Падение импорта `outlines` (например, из-за конфликта numpy) не теряется."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "outlines" or name.startswith("outlines."):
            raise ImportError("Numba needs NumPy 2.4 or less")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    from hipporag.llm.outlines_compat import OutlinesUnavailableError, build_guided_generator

    with pytest.raises(OutlinesUnavailableError) as excinfo:
        build_guided_generator("model", stub_tokenizer, _Schema)

    assert "NumPy" in str(excinfo.value)


# ── сквозной путь `batch_infer` ────────────────────────────────────────────


def test_batch_infer_guided_returns_json_strings(monkeypatch):
    """Guided-путь `batch_infer` отдаёт JSON-строки (контракт потребителя OpenIE)."""
    _install_modern_outlines(monkeypatch)

    from hipporag.llm.transformers_offline import TransformersOffline

    class _Tokenizer:
        padding_side = "right"

        def apply_chat_template(self, conversation, **kwargs):
            return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)

        def encode(self, text, **kwargs):
            return list(range(1, len(text.split()) + 1))

    backend = TransformersOffline.__new__(TransformersOffline)
    backend.model = "model"
    backend.tokenizer = _Tokenizer()

    responses, metadata = backend.batch_infer(
        [[{"role": "user", "content": "вопрос"}]], json_template="ner", max_tokens=32
    )

    assert len(responses) == 1
    assert isinstance(responses[0], str)
    assert json.loads(responses[0]) == {"code": "ab"}
    assert metadata["num_request"] == 1


# ── функциональная проверка с настоящим `outlines` (skip, если нет) ───────


def test_guided_path_with_real_outlines():
    """Guided-путь на настоящем `outlines` и настоящей мини-модели.

    Проверяет то, чего не проверяют подмены: что пакет реально ограничивает
    вывод схемой, что JSON на выходе валиден и что длина строки в границах
    схемы (ограничения длины `outlines` соблюдает).

    Модель `sshleifer/tiny-gpt2` (~2,5 МБ) — тестовая, качество ответов не
    важно; нужен настоящий токенизатор с chat-шаблоном.
    """
    try:
        import outlines  # noqa: F401
    except Exception as exc:  # noqa: BLE001 — пакет может быть и не установлен, и несовместим
        pytest.skip(f"пакет `outlines` недоступен: {exc}")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"мини-модель недоступна: {exc}")

    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from hipporag.llm.outlines_compat import build_guided_generator

    class _Bounded(pydantic.BaseModel):
        code: str = pydantic.Field(min_length=2, max_length=4)

    original_padding = tokenizer.padding_side
    generator = build_guided_generator(model, tokenizer, _Bounded)
    results = generator(["Сколько будет два плюс два?"], 32)

    assert len(results) == 1
    assert isinstance(results[0], _Bounded), type(results[0]).__name__
    assert 2 <= len(results[0].code) <= 4, results[0].code
    assert tokenizer.padding_side == original_padding
