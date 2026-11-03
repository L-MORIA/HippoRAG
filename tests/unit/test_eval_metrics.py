"""Тесты eval-базы (H1-1): разметка, сопоставление выдачи с чанками, метрики.

Без индекса, без моделей: всё, что здесь проверяется, — чистые функции из
``astro_app.evaluation``. Смысл набора — зафиксировать, что метрики не врут
молча: битая разметка падает с понятной ошибкой, несопоставленный текст не
превращается в «попадание», а страничные метрики считаются только по вопросам
со размеченной страницей.
"""

import json

import pytest

from astro_app import evaluation as ev
from astro_app.modes import MODES

CHUNK_A = (
    "ОВЕН Начнём мы описание с первого знака зодиака, а это Овен. И соответствует ему "
    "прихожая и плита. Управитель у Овна — Марс, поэтому он очень любит делать в зоне "
    "прихожей арки."
)
CHUNK_B = (
    "Второй шаг. 11 баллов. Солнце и Марс. Овен ремонтирует прихожую и гостиную, либо "
    "сразу, либо последовательно, но в одном временном континууме, без задержек."
)
CHUNK_C = (
    "Число гуа 6 относится к западной группе. Для него благоприятны направления северо-"
    "запад, юго-запад и запад, а неблагоприятны восток и юго-восток."
)


def make_lookup():
    return ev.build_lookup_from_records(
        [
            {
                "hash_id": "chunk-a",
                "content": CHUNK_A,
                "metadata": [{"source_file": "Астрофэншуй квартиры.pdf", "page_num": 7}],
            },
            {
                "hash_id": "chunk-b",
                "content": CHUNK_B,
                "metadata": [{"source_file": "Астрофэншуй квартиры.pdf", "page_num": 171}],
            },
            {
                "hash_id": "chunk-c",
                "content": CHUNK_C,
                "metadata": json.dumps(
                    [{"source_file": "Психологический фэншуй.pdf", "page_num": 42}]
                ),
            },
        ]
    )


def write_dataset(tmp_path, entries, name="questions.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8"
    )
    return path


def entry(**overrides):
    base = {
        "qid": "q-1",
        "question": "Какие зоны квартиры соответствуют Овну?",
        "kind": "domain",
        "subtype": "sign-zone",
        "source_file": "Астрофэншуй квартиры.pdf",
        "page_num": 7,
        "reference_answer": "Прихожая и плита, управитель Марс.",
        "expected_terms": ["прихожая", "плита"],
        "status": "verified",
    }
    base.update(overrides)
    return base


# ── Разметка ─────────────────────────────────────────────────────────────────


def test_load_dataset_splits_complete_and_incomplete(tmp_path):
    path = write_dataset(
        tmp_path,
        [
            entry(qid="q-1"),
            entry(qid="q-2", question="", reference_answer=""),
        ],
    )
    dataset = ev.load_dataset(path)

    assert [q.qid for q in dataset.questions] == ["q-1"]
    assert dataset.incomplete == [("q-2", ("question", "reference_answer"))]
    assert dataset.for_run() == [dataset.questions[0]]


def test_load_dataset_skips_comment_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        "# комментарий разметчика\n" + json.dumps(entry(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert len(ev.load_dataset(path).questions) == 1


def test_draft_is_excluded_from_run_by_default(tmp_path):
    dataset = ev.load_dataset(write_dataset(tmp_path, [entry(status="draft")]))
    assert dataset.for_run() == []
    assert len(dataset.for_run(include_draft=True)) == 1


def test_json_object_with_questions_key(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": [entry()]}, ensure_ascii=False), encoding="utf-8")
    assert ev.load_dataset(path).questions[0].qid == "q-1"


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"sourse_file": "Астрофэншуй квартиры.pdf"}, "неизвестные поля"),
        ({"kind": "astrology"}, "kind"),
        ({"subtype": "знак-зона"}, "subtype"),
        ({"status": "ready"}, "status"),
        ({"page_num": 0}, "page_num"),
        ({"page_num": "7"}, "page_num"),
        ({"source_file": ""}, "source_file"),
        ({"qid": "плохой id"}, "qid"),
    ],
)
def test_load_dataset_rejects_broken_markup(tmp_path, overrides, fragment):
    path = write_dataset(tmp_path, [entry(**overrides)])
    with pytest.raises(ev.DatasetError, match=fragment):
        ev.load_dataset(path)


def test_load_dataset_rejects_duplicate_qid(tmp_path):
    path = write_dataset(tmp_path, [entry(), entry()])
    with pytest.raises(ev.DatasetError, match="уже был"):
        ev.load_dataset(path)


def test_load_dataset_reports_line_number_for_broken_json(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("{не json}\n", encoding="utf-8")
    with pytest.raises(ev.DatasetError, match="строка 1"):
        ev.load_dataset(path)


def test_missing_dataset_is_an_error(tmp_path):
    with pytest.raises(ev.DatasetError, match="не найден"):
        ev.load_dataset(tmp_path / "nope.jsonl")


# ── Валидация ────────────────────────────────────────────────────────────────


def test_validation_counts_and_incomplete(tmp_path):
    dataset = ev.load_dataset(
        write_dataset(
            tmp_path,
            [
                entry(qid="q-1", kind="domain", subtype="sign-zone"),
                entry(qid="q-2", kind="bibliography", subtype="title-author"),
                entry(qid="q-3", question="", status="draft"),
            ],
        )
    )
    report = ev.validate_dataset(dataset)

    assert report.ok
    assert report.total == 2
    assert report.incomplete == 1
    assert report.by_kind == {"domain": 1, "bibliography": 1}

    # Библиографический класс с подтипом title-author — без предупреждения,
    # а вот «verified меньше цели» предупредить обязано.
    assert any("verified" in w for w in report.warnings)


def test_validation_detects_source_and_page_absent_from_index(tmp_path):
    dataset = ev.load_dataset(
        write_dataset(
            tmp_path,
            [
                entry(qid="q-1", source_file="Нет такой книги.pdf"),
                entry(qid="q-2", page_num=999),
            ],
        )
    )
    report = ev.validate_dataset(dataset, make_lookup())

    assert not report.ok
    assert any("нет в индексе" in p for p in report.problems)
    assert any("страницы 999" in p for p in report.problems)


def test_validation_warns_when_page_is_null(tmp_path):
    dataset = ev.load_dataset(write_dataset(tmp_path, [entry(page_num=None)]))
    report = ev.validate_dataset(dataset, make_lookup())
    assert report.ok
    assert any("только в документных" in w for w in report.warnings)


def test_validation_handles_source_without_pages(tmp_path):
    """Документ есть, страниц в метаданных нет — не KeyError, а понятная ошибка."""
    lookup = ev.build_lookup_from_records(
        [{"hash_id": "h", "content": "текст", "metadata": {"source_file": "a.pdf"}}]
    )
    dataset = ev.load_dataset(write_dataset(tmp_path, [entry(source_file="a.pdf", page_num=5)]))
    report = ev.validate_dataset(dataset, lookup)

    assert not report.ok
    assert any("нет страниц в метаданных" in p for p in report.problems)


# ── Сопоставление выдачи с чанками ───────────────────────────────────────────


def test_lookup_matches_exact_text():
    lookup = make_lookup()
    assert [c.hash_id for c in lookup.match(CHUNK_A)] == ["chunk-a"]


def test_lookup_matches_by_hash_and_normalized_text():
    import hashlib

    chunk_id = "chunk-" + hashlib.md5(CHUNK_B.encode()).hexdigest()
    lookup = ev.build_lookup_from_records(
        [
            {
                "hash_id": chunk_id,
                "content": CHUNK_B,
                "metadata": {"source_file": "Астрофэншуй квартиры.pdf", "page_num": 171},
            }
        ]
    )
    assert lookup.match(CHUNK_B)[0].hash_id == chunk_id
    # Лишние пробелы: точного md5 нет, но нормализованный текст совпадает.
    padded = "  " + CHUNK_B.replace(" ", "  ") + " "
    assert lookup.match(padded)[0].hash_id == chunk_id


def test_lookup_strips_sentence_markers():
    lookup = make_lookup()
    marked = CHUNK_A.replace("прихожая и плита", "\u2192прихожая и плита\u2190")
    assert [c.hash_id for c in lookup.match(marked)] == ["chunk-a"]


def test_lookup_matches_merged_small_to_big_text():
    lookup = make_lookup()
    merged = CHUNK_A + "\n\n" + CHUNK_B
    assert {c.hash_id for c in lookup.match(merged)} == {"chunk-a", "chunk-b"}


def test_lookup_matches_truncated_fragment():
    lookup = make_lookup()
    assert [c.hash_id for c in lookup.match(CHUNK_C[:120])] == ["chunk-c"]


def test_lookup_falls_back_to_token_overlap():
    lookup = make_lookup()
    # Тот же набор слов, другой порядок и связки — точного вхождения нет.
    paraphrased = "Марс — управитель Овна. Прихожая и плита соответствуют первому знаку."
    assert [c.hash_id for c in lookup.match(paraphrased)] == ["chunk-a"]


def test_lookup_returns_nothing_for_unrelated_text():
    lookup = make_lookup()
    assert lookup.match("Совершенно другой текст про налоги и бухгалтерию за март.") == []


def test_lookup_parses_metadata_shapes():
    lookup = ev.build_lookup_from_records(
        [
            {"hash_id": "h1", "content": "А", "metadata": {"source_file": "a.pdf", "page_num": 3}},
            {
                "hash_id": "h2",
                "content": "Б",
                "metadata": '[{"source_file": "b.pdf", "page_num": 4}]',
            },
            {
                "hash_id": "h3",
                "content": "В",
                "metadata": [
                    {"source_file": "c.pdf", "page_num": 5},
                    {"source_file": "d.pdf", "page_num": 6},
                ],
            },
        ]
    )
    assert lookup.sources() == {"a.pdf", "b.pdf", "c.pdf", "d.pdf"}
    assert lookup.chunks[2].entries == (("c.pdf", 5), ("d.pdf", 6))
    assert lookup.chunks[1].page_num == 4


def test_lookup_handles_broken_metadata_without_crashing():
    lookup = ev.build_lookup_from_records(
        [{"hash_id": "h1", "content": "А", "metadata": "{битый json"}]
    )
    assert lookup.chunks[0].entries == ()
    assert lookup.sources() == set()


class _ChunkStoreStub:
    """Минимальный двойник chunk_embedding_store (без HippoRAG и моделей)."""

    hash_ids = ["chunk-1", "chunk-2"]
    texts = ["Первый чанк про Овна", "Второй чанк про Тельца"]
    metadatas = [
        [{"source_file": "a.pdf", "page_num": 7}],
        [{"source_file": "a.pdf", "page_num": 8}, {"source_file": "b.pdf", "page_num": 1}],
    ]


def test_build_chunk_lookup_from_store():
    lookup = ev.build_chunk_lookup(_ChunkStoreStub())
    assert len(lookup.chunks) == 2
    assert lookup.chunks[1].entries == (("a.pdf", 8), ("b.pdf", 1))
    assert lookup.sources() == {"a.pdf", "b.pdf"}
    assert lookup.pages_by_source()["b.pdf"] == {1}


def test_build_chunk_lookup_rejects_inconsistent_store():
    class Broken(_ChunkStoreStub):
        texts = ["один чанк"]

    with pytest.raises(ValueError, match="повреждено"):
        ev.build_chunk_lookup(Broken())


# ── Метрики ранжирования ─────────────────────────────────────────────────────


def test_first_hit_rank():
    assert ev.first_hit_rank([False, False, True]) == 3
    assert ev.first_hit_rank([False, False]) is None
    assert ev.first_hit_rank([]) is None


def test_recall_at_k():
    ranks = [1, 3, 6, None]
    assert ev.recall_at_k(ranks, 5) == 0.5
    assert ev.recall_at_k(ranks, 10) == 0.75
    assert ev.recall_at_k([], 5) == 0.0


def test_mrr_at_k():
    assert ev.mrr_at_k([1, 2, None], 10) == pytest.approx((1 + 0.5) / 3)
    # Попадание дальше k не даёт вклада, но и не выбрасывается из знаменателя.
    assert ev.mrr_at_k([1, 20], 10) == pytest.approx(0.5)


def test_ndcg_at_k():
    assert ev.ndcg_at_k([1], 10) == pytest.approx(1.0)
    assert ev.ndcg_at_k([2], 10) == pytest.approx(1 / 1.5849625, rel=1e-4)
    assert ev.ndcg_at_k([None], 10) == 0.0


def test_percentile_linear_interpolation():
    assert ev.percentile([4, 1, 3, 2], 50) == pytest.approx(2.5)
    assert ev.percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert ev.percentile([7], 95) == 7
    assert ev.percentile([], 50) is None


# ── Отказы, пустота, adequacy ────────────────────────────────────────────────


@pytest.mark.parametrize("text", ["", "   ", "Ничего не найдено.", "Нет результатов."])
def test_is_empty(text):
    assert ev.is_empty(text)


@pytest.mark.parametrize(
    "text",
    [
        "Материал по знаку 'Овен' не найден",
        "Не указан знак зодиака в запросе",
        "Не удалось получить ответ.",
        "Информация отсутствует в базе.",
        "Нет данных.",
    ],
)
def test_is_refusal_recognizes_known_placeholders(text):
    assert ev.is_refusal(text)


def test_is_refusal_ignores_negation_inside_a_long_answer():
    text = (
        "Овен любит прихожую и плиту, его управитель — Марс. " * 8
        + "Если зона не найдена в описании, проверьте планировку."
    )
    assert not ev.is_refusal(text)


def test_term_matches_russian_morphology():
    norm = ev.normalize_text("Овен ремонтирует прихожую и гостиную")
    assert ev.term_matches("прихожая", norm)
    assert not ev.term_matches("спальня", norm)


def test_answer_adequacy_penalizes_empty_refusal_and_short_answers():
    terms = ["прихожая", "плита"]
    full = "Овну соответствуют прихожая и плита, управитель — Марс. " * 3
    assert ev.answer_adequacy(full, terms) == 1.0
    assert ev.answer_adequacy("", terms) == 0.0
    assert ev.answer_adequacy("Материал по знаку не найден", terms) == 0.0
    # Один термин из двух — половина покрытия, длина достаточная.
    assert ev.answer_adequacy("Здесь только прихожая. " * 6, terms) == pytest.approx(0.5)
    # Короткий ответ режется по длине, а не считается полноценным.
    assert ev.answer_adequacy("прихожая и плита", terms) < 0.5


def test_answer_adequacy_ignores_sources_block():
    """Режимы 12/21 приклеивают найденные чанки: adequacy не должна их считать."""
    refusal = "=== ОТВЕТ ===\n\nНет данных.\n\n=== ИСТОЧНИКИ ===\nОвну — прихожая и плита. " * 2
    assert ev.answer_adequacy(refusal, ["прихожая", "плита"]) == 0.0
    assert ev.answer_body(refusal).startswith("=== ОТВЕТ ===")

    answer = "=== ОТВЕТ ===\n\n" + "Овну — прихожая и плита, управитель Марс. " * 3
    answer += "\n\n=== ИСТОЧНИКИ ===\nдругой текст"
    assert ev.answer_adequacy(answer, ["прихожая", "плита"]) == 1.0


# ── Сводка ───────────────────────────────────────────────────────────────────


def row(**overrides):
    base = {
        "qid": "q-1",
        "retrieval": True,
        "page_expected": True,
        "doc_rank": 1,
        "page_rank": 2,
        "n_candidates": 10,
        "latency_ms": 100.0,
        "llm_calls": 0,
        "llm_ms": 0.0,
        "answer_text": None,
        "adequacy": None,
        "error": None,
    }
    base.update(overrides)
    return base


def test_summarize_mode_retrieval_metrics():
    rows = [
        row(qid="q-1", doc_rank=1, page_rank=2),
        row(qid="q-2", doc_rank=4, page_rank=None),
        row(qid="q-3", doc_rank=None, page_rank=None, page_expected=False),
    ]
    summary = ev.summarize_mode(rows, mode_id="2", label="Граф")

    assert summary["n_retrieval"] == 3
    assert summary["n_answer"] == 0
    assert summary["recall@k_doc"] == pytest.approx(2 / 3, abs=1e-4)
    # Страница размечена у двух вопросов: попали в одном.
    assert summary["recall@k_page"] == pytest.approx(0.5, abs=1e-4)
    assert summary["mrr@k_doc"] == pytest.approx((1 + 0.25 + 0) / 3, abs=1e-4)
    assert summary["refusal_rate"] is None
    assert summary["n_candidates"]["lt3"] == 0


def test_summarize_mode_counts_degenerate_candidate_sets():
    rows = [row(n_candidates=1), row(n_candidates=2), row(n_candidates=10)]
    summary = ev.summarize_mode(rows, mode_id="2", label="Граф")
    assert summary["n_candidates"]["lt3"] == 2
    assert summary["n_candidates"]["min"] == 1
    assert summary["n_candidates"]["max"] == 10


def test_summarize_mode_answer_metrics():
    rows = [
        row(retrieval=False, answer_text="Овну — прихожая и плита, Марс. " * 3, adequacy=1.0),
        row(retrieval=False, answer_text="Материал по знаку 'Овен' не найден", adequacy=0.0),
        row(retrieval=False, answer_text="", adequacy=0.0),
    ]
    summary = ev.summarize_mode(rows, mode_id="10", label="Резюме")

    assert summary["n_retrieval"] == 0
    assert summary["n_answer"] == 3
    assert summary["refusal_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["empty_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["adequate_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["recall@k_doc"] is None  # нет retrieval-строк — метрика не применима


def test_summarize_mode_errors_do_not_break_latency():
    rows = [
        row(latency_ms=100.0),
        row(qid="q-2", latency_ms=0.0, error="RuntimeError: нет модели"),
    ]
    summary = ev.summarize_mode(rows, mode_id="5", label="BM25+CE-ru")
    assert summary["n_errors"] == 1
    assert summary["latency_ms"]["p50"] == 100.0
    assert "q-2" in summary["errors"][0]


def test_format_markdown_table_has_headers_and_rows():
    retrieval = ev.summarize_mode([row()], mode_id="2", label="Граф")
    answers = ev.summarize_mode(
        [row(retrieval=False, answer_text="x" * 120, adequacy=1.0)], mode_id="10", label="Резюме"
    )
    table = ev.format_markdown_table([retrieval, answers])

    assert "recall@5 (стр)" in table
    assert "| 2 Граф |" in table
    assert "| 10 Резюме |" in table


# ── Согласованность со реестром режимов ──────────────────────────────────────


def test_mode_sets_are_known_modes():
    assert ev.LLM_DEPENDENT_MODES <= set(MODES)
    assert ev.ANSWER_ONLY_MODES <= set(MODES)
    # Режим 9 («Весь знак») — единственный ответ без LLM: он просто склеивает
    # материал, поэтому в LLM-зависимые не входит, а в режимы-ответы входит.
    assert ev.ANSWER_ONLY_MODES - ev.LLM_DEPENDENT_MODES == {"9"}
