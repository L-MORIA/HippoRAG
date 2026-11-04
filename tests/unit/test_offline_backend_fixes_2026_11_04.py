"""Регрессионные тесты по находкам N-1…N-4 (раунд 7).

Документ-основание: `docs/AUDIT_RECHECK_FIX_VERIFICATION_2026-11-04.md` —
независимая верификация исправлений раунда 6. Вердикт верификатора: правки
приняты, но «на границе» замечания вскрылись ещё три находки; плюс найдена
четвёртая (N-4, контракт `infer`).

* **N-1** — `TransformersOffline.infer` строил промпт как **батч**
  (`messages_list = [messages]`): `apply_chat_template` возвращал список строк,
  а `tokenizer.encode()` такую форму отвергал — на реальной модели `infer()`
  падал, то есть offline-бэкенд не работал end-to-end даже после исправлений
  S-4.
* **N-2** — free-form `batch_infer` декодировал **промпт вместе с ответом**:
  промпт утекал в ответ, а `completion_tokens` завышался.
* **N-3** — `retrieve(queries=[QuerySolution(...)])` падал: `QuerySolution` —
  dataclass без `__hash__`, в кэше эмбеддингов он не хэшируется
  (`TypeError: unhashable type`).
* **N-4** — `infer()` у обоих offline-бэкендов возвращал 2-tuple, тогда как
  контракт `BaseLLM.infer` и четыре места вызова ждут тройку
  `(контент, метаданные, признак кэша)`.

Тесты не требуют индекса, моделей и сети: используются стабы. Отдельный тест
с настоящей мини-моделью (`sshleifer/tiny-gpt2`) помечается `skip`, если
модель недоступна.
"""

import sys
import types

import numpy as np
import pytest
import torch

# ── стабы для transformers_offline ─────────────────────────────────────────


class _Encoding(dict):
    """Минимальный BatchEncoding: умеет `.to(device)` и индексируется."""

    def to(self, device):  # noqa: ARG002
        return self


class _StubTokenizer:
    """Токенизатор с поведением настоящего transformers.

    `apply_chat_template` для одного разговора возвращает строку, для батча
    (список разговоров) — список строк: именно на этом спотыкался N-1.
    """

    def __init__(self):
        self.conversations = []
        self.padding_side = "right"

    @staticmethod
    def _render(conversation):
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)

    def apply_chat_template(self, conversation, **kwargs):
        self.conversations.append(conversation)
        if isinstance(conversation[0], (list, tuple)):
            return [self._render(item) for item in conversation]
        return self._render(conversation)

    def __call__(self, text, return_tensors=None, padding=False, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        width = max(len(item) for item in texts)
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row, item in enumerate(texts):
            ids[row, : len(item)] = torch.arange(1, len(item) + 1)
            mask[row, : len(item)] = 1
        return _Encoding({"input_ids": ids, "attention_mask": mask})

    def encode(self, text, **kwargs):
        # один токен на слово — достаточно, чтобы отличить «ответ с промптом» от «ответа»
        return list(range(1, len(text.split()) + 1))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class _StubModel:
    """Модель: generate() возвращает промпт + 2 токена ответа."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, **kwargs):
        self.calls.append({"max_new_tokens": max_new_tokens, "shape": tuple(input_ids.shape)})
        tail = torch.full((input_ids.shape[0], 2), 99, dtype=torch.long)
        return torch.cat([input_ids, tail], dim=1)


@pytest.fixture
def offline_backend(monkeypatch):
    """`TransformersOffline` со стабами: без модели, без `outlines`."""
    outlines = types.ModuleType("outlines")
    generate = types.ModuleType("outlines.generate")
    models = types.ModuleType("outlines.models")
    outlines.generate = generate
    outlines.models = models
    monkeypatch.setitem(sys.modules, "outlines", outlines)
    monkeypatch.setitem(sys.modules, "outlines.generate", generate)
    monkeypatch.setitem(sys.modules, "outlines.models", models)
    monkeypatch.delitem(sys.modules, "hipporag.llm.transformers_offline", raising=False)

    from hipporag.llm.transformers_offline import TransformersOffline

    backend = TransformersOffline.__new__(TransformersOffline)
    backend.model = _StubModel()
    backend.tokenizer = _StubTokenizer()
    backend.model_name = "stub"
    backend.cache_file_name = "stub.sqlite"
    return backend


# ── N-1: промпт строится из одного разговора, а не из батча ────────────────


def test_infer_builds_prompt_from_single_conversation(offline_backend):
    """N-1: в apply_chat_template уходит плоский список сообщений, не батч."""
    backend = offline_backend
    backend.infer([{"role": "user", "content": "два плюс два"}], max_tokens=8)

    conversation = backend.tokenizer.conversations[0]
    assert isinstance(
        conversation[0], dict
    ), "разговор обёрнут в батч — apply_chat_template вернёт список"
    assert len(conversation) == 1


def test_infer_returns_completion_only(offline_backend):
    """N-1: ответ не содержит промпта, токены считаются честно."""
    backend = offline_backend

    response, metadata, cached = backend.infer(
        [{"role": "user", "content": "два плюс два"}], max_tokens=8
    )

    assert response == "99 99", "в ответ попал промпт"
    assert metadata["completion_tokens"] == 2
    assert metadata["prompt_tokens"] > 0
    assert cached is False


def test_infer_passes_max_new_tokens(offline_backend):
    """N-1: generate() получает max_new_tokens (а не max_tokens)."""
    backend = offline_backend

    backend.infer([{"role": "user", "content": "вопрос"}], max_tokens=8)

    assert backend.model.calls[0]["max_new_tokens"] == 8


# ── N-2: free-form batch_infer не возвращает промпт ────────────────────────


def test_batch_infer_free_form_does_not_leak_prompt(offline_backend):
    """N-2: free-form батч возвращает только завершение, промпт не утекает."""
    backend = offline_backend
    messages = [
        [{"role": "user", "content": "первый вопрос"}],
        [{"role": "user", "content": "второй вопрос подлиннее"}],
    ]

    responses, metadata = backend.batch_infer(messages, max_tokens=8)

    assert responses == ["99 99", "99 99"], "ответ содержит промпт"
    # completion_tokens не должен учитывать промпт: 2 ответа по 2 токена
    assert metadata["completion_tokens"] == 4
    assert metadata["num_request"] == 2


def test_batch_infer_uses_left_padding_for_generation(offline_backend):
    """N-2: батчевая генерация требует left-padding (иначе модель идёт с pad-токена)."""
    backend = offline_backend
    original = backend.tokenizer.padding_side

    backend.batch_infer([[{"role": "user", "content": "вопрос"}]], max_tokens=8)

    assert backend.tokenizer.padding_side == original, "padding_side не восстановлен"
    assert backend.model.calls, "generate() не вызван"


# ── N-4: контракт infer — тройка ───────────────────────────────────────────


def test_transformers_offline_infer_returns_three_values(offline_backend):
    """N-4: infer() отдаёт (контент, метаданные, признак кэша)."""
    result = offline_backend.infer([{"role": "user", "content": "вопрос"}], max_tokens=8)

    assert len(result) == 3
    content, metadata, cached = result
    assert isinstance(content, str)
    assert isinstance(metadata, dict)
    assert cached is False


def test_vllm_offline_infer_returns_three_values(monkeypatch):
    """N-4: vllm-бэкенд отдаёт ту же тройку."""
    vllm_stub = types.ModuleType("vllm")
    vllm_stub.SamplingParams = lambda **kwargs: kwargs
    vllm_stub.LLM = object
    monkeypatch.setitem(sys.modules, "vllm", vllm_stub)
    monkeypatch.delitem(sys.modules, "hipporag.llm.vllm_offline", raising=False)

    from hipporag.llm.vllm_offline import VLLMOffline

    class _Tok:
        def apply_chat_template(self, **kwargs):
            return "PROMPT"

        def __call__(self, prompt, add_special_tokens=False):
            return {"input_ids": [[1, 2, 3]]}

    class _Client:
        def generate(self, prompt_token_ids, sampling_params):
            out = types.SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[types.SimpleNamespace(text="ANSWER", token_ids=[4, 5])],
            )
            return [out]

    backend = VLLMOffline.__new__(VLLMOffline)
    backend.tokenizer = _Tok()
    backend.client = _Client()

    result = backend.infer([{"role": "user", "content": "вопрос"}])

    assert len(result) == 3
    content, metadata, cached = result
    assert content == "ANSWER"
    assert metadata["prompt_tokens"] == 3
    assert metadata["completion_tokens"] == 2
    assert cached is False


def test_base_llm_contract_is_three_values():
    """N-4: контракт BaseLLM.infer — тройка, и он задекларирован в базовом классе."""
    import inspect

    from hipporag.llm.base import BaseLLM

    annotation = inspect.signature(BaseLLM.infer).return_annotation
    assert "bool" in str(annotation), annotation


# ── N-3: retrieve() принимает QuerySolution ────────────────────────────────


class _Embedder:
    def __init__(self):
        self.calls = 0

    def batch_encode(self, texts, instruction=None, norm=True):
        self.calls += 1
        if isinstance(texts, str):
            texts = [texts]
        return np.ones((len(texts), 4), dtype=np.float32)


def _standard_rag_stub():
    from hipporag import StandardRAG

    rag = StandardRAG.__new__(StandardRAG)
    rag.query_to_embedding = {"triple": {}, "passage": {}}
    rag.embedding_model = _Embedder()
    rag.passage_embeddings = np.ones((2, 4), dtype=np.float32)
    rag.passage_node_keys = ["k0", "k1"]
    rag.chunk_embedding_store = type(
        "_Store", (), {"get_row": staticmethod(lambda key: {"content": f"text-{key}"})}
    )()
    rag.global_config = type("_Cfg", (), {"retrieval_top_k": 2})()
    rag.ready_to_retrieve = True
    rag.all_retrieval_time = 0.0
    return rag


def test_standard_rag_retrieve_accepts_query_solution():
    """N-3: retrieve([QuerySolution(...)]) не падает, question — строка."""
    from hipporag.utils.custom_types import QuerySolution

    rag = _standard_rag_stub()

    solutions = rag.retrieve([QuerySolution(question="какой знак?")])

    assert len(solutions) == 1
    assert solutions[0].question == "какой знак?"
    assert sorted(solutions[0].docs) == ["text-k0", "text-k1"]


def test_standard_rag_retrieve_still_accepts_plain_strings():
    """N-3: обычные строки работают как раньше."""
    rag = _standard_rag_stub()

    solutions = rag.retrieve(["какой знак?"])

    assert len(solutions) == 1
    assert solutions[0].question == "какой знак?"


def test_standard_rag_dense_passage_retrieval_accepts_query_solution():
    """N-3: dense_passage_retrieval переживает QuerySolution (кэш — по тексту)."""
    from hipporag.utils.custom_types import QuerySolution

    rag = _standard_rag_stub()
    rag.query_to_embedding["passage"]["какой знак?"] = np.ones(4, dtype=np.float32)

    doc_ids, doc_scores = rag.dense_passage_retrieval(QuerySolution(question="какой знак?"))

    assert len(doc_ids) == 2
    assert len(doc_scores) == 2
    # попадание в кэш: эмбеддер не вызывался
    assert rag.embedding_model.calls == 0


def test_hippo_rag_dense_passage_retrieval_accepts_query_solution():
    """N-3: то же в HippoRAG — запрос ищется в кэше по тексту вопроса."""
    from hipporag import HippoRAG
    from hipporag.utils.custom_types import QuerySolution

    rag = HippoRAG.__new__(HippoRAG)
    rag.query_to_embedding = {"triple": {}, "passage": {"какой знак?": np.ones(4, np.float32)}}
    rag.embedding_model = _Embedder()
    rag.passage_embeddings = np.ones((2, 4), dtype=np.float32)
    rag.ready_to_retrieve = True

    doc_ids, doc_scores = rag.dense_passage_retrieval(QuerySolution(question="какой знак?"))

    assert len(doc_ids) == 2
    assert len(doc_scores) == 2
    assert rag.embedding_model.calls == 0


# ── функциональная проверка на настоящей мини-модели (skip, если нет) ──────


def test_offline_infer_on_real_mini_model(monkeypatch):
    """N-1 на настоящей модели: `infer()` отрабатывает и отдаёт только завершение.

    Модель `sshleifer/tiny-gpt2` (~2,5 МБ) — тестовая, качество ответов не важно;
    нужен настоящий токенизатор с chat-шаблоном. Если модель недоступна
    (нет сети) — тест пропускается.
    """
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outlines = types.ModuleType("outlines")
    generate = types.ModuleType("outlines.generate")
    models = types.ModuleType("outlines.models")
    outlines.generate = generate
    outlines.models = models
    monkeypatch.setitem(sys.modules, "outlines", outlines)
    monkeypatch.setitem(sys.modules, "outlines.generate", generate)
    monkeypatch.setitem(sys.modules, "outlines.models", models)
    monkeypatch.delitem(sys.modules, "hipporag.llm.transformers_offline", raising=False)

    try:
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"мини-модель недоступна: {exc}")

    # у tiny-gpt2 нет chat-шаблона — задаём минимальный
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from hipporag.llm.transformers_offline import TransformersOffline

    backend = TransformersOffline.__new__(TransformersOffline)
    backend.model = model
    backend.tokenizer = tokenizer
    backend.model_name = "sshleifer/tiny-gpt2"
    backend.cache_file_name = "stub.sqlite"

    prompt = "Сколько будет два плюс два?"
    response, metadata, cached = backend.infer([{"role": "user", "content": prompt}], max_tokens=8)

    assert isinstance(response, str)
    assert cached is False
    assert metadata["completion_tokens"] > 0
    # промпт не должен попасть в ответ
    assert prompt not in response
    assert "user:" not in response
