"""H1-5: смоук eval-контура в CI — 10 вопросов, без LLM и без индекса.

Задача H1-5 плана («CI-regression») требует 10-вопросного смоука в CI. Полная
её часть — прогон реального retrieval на фиксированном индексе — в CI не
воспроизводится: индекс (`outputs/astrology_test`) не лежит в репозитории, а
живой прогон требует LLM (LM Studio) и эмбеддинг-моделей.

Здесь закрывается воспроизводимая часть смоука:

* контракт датасета (`eval/dataset/questions.jsonl`) — парсинг, схема, уникальность
  `qid`, статус `verified`, валидация через `validate_dataset`;
* корректность метрик (`recall@k`, `MRR@k`, `nDCG@k`) на синтетической выдаче:
  «идеальное попадание», «полный промах», «попадание на 5-й позиции»;
* согласованность разметки: `expected_terms` каждого вопроса встречается в его
  же `reference_answer`.

Тест не требует индекса, моделей и сети, поэтому выполняется в дешёвом CI.
"""

from pathlib import Path

import pytest

from astro_app import evaluation as ev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "eval" / "dataset" / "questions.jsonl"
SMOKE_SIZE = 10


@pytest.fixture(scope="module")
def dataset():
    return ev.load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def smoke(dataset):
    """Детерминированная выборка первых SMOKE_SIZE вопросов датасета."""
    questions = dataset.for_run()
    assert len(questions) >= SMOKE_SIZE
    return questions[:SMOKE_SIZE]


def test_dataset_contract_is_valid(dataset):
    """Датасет разбирается, схема соблюдена, валидация зелёная."""
    questions = dataset.for_run()
    assert len(questions) == 40
    assert len(dataset.by_status("verified")) == 40
    assert len({question.qid for question in questions}) == len(questions)
    assert all(question.status == "verified" for question in questions)
    assert all(question.source_file for question in questions)
    assert all(question.page_num is not None for question in questions)
    assert all(question.expected_terms for question in questions)

    report = ev.validate_dataset(dataset)
    assert report.ok, report.format_text()


def test_smoke_subset_is_deterministic(dataset):
    """Выборка смоука не зависит от порядка запуска тестов."""
    first = [question.qid for question in dataset.for_run()[:SMOKE_SIZE]]
    second = [question.qid for question in dataset.for_run()[:SMOKE_SIZE]]
    assert first == second
    assert len(set(first)) == SMOKE_SIZE


def test_metrics_on_perfect_retrieval(smoke):
    """Идеальная выдача: recall@5 = MRR@5 = nDCG@5 = 1.0."""
    ranks = [ev.first_hit_rank([True]) for _ in smoke]
    assert ev.recall_at_k(ranks, 5) == 1.0
    assert ev.mrr_at_k(ranks, 5) == 1.0
    assert ev.ndcg_at_k(ranks, 5) == 1.0


def test_metrics_on_missed_retrieval(smoke):
    """Полный промах: все метрики нулевые."""
    ranks = [ev.first_hit_rank([False]) for _ in smoke]
    assert ev.recall_at_k(ranks, 5) == 0.0
    assert ev.mrr_at_k(ranks, 5) == 0.0
    assert ev.ndcg_at_k(ranks, 5) == 0.0


def test_metrics_on_boundary_hit(smoke):
    """Попадание ровно на 5-й позиции попадает в recall@5, но не в recall@3."""
    ranks = [ev.first_hit_rank([False, False, False, False, True]) for _ in smoke]
    assert ev.recall_at_k(ranks, 5) == 1.0
    assert ev.recall_at_k(ranks, 3) == 0.0
    assert ev.mrr_at_k(ranks, 5) == pytest.approx(1.0 / 5.0)
    assert ev.mrr_at_k(ranks, 3) == 0.0


def test_expected_terms_consistent_with_reference_answers(smoke):
    """Разметка согласована: эталонные термины есть в эталонном ответе."""
    for question in smoke:
        normalized = ev.normalize_text(question.reference_answer or "")
        hits = [
            term for term in (question.expected_terms or []) if ev.term_matches(term, normalized)
        ]
        assert hits, f"{question.qid}: ни один expected_term не найден в reference_answer"
