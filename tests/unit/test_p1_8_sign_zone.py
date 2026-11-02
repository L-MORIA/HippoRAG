"""P1-8: SemFilter (режим 22) и QACache (23) понимают ЗНАК из запроса.

Раньше метод искал пару «знак → зона» только по зоне, названной в запросе:
на «какая зона подходит для Овна?» он печатал «Целевая зона: ?» и выдавал шум,
а на вопрос про книги — случайного «знака Рыбы».
"""

import numpy as np
import pytest

TEXTS = [
    "итак: обитель овна — марс — это плита, холл, спортзал.",
    "точка силы овна — это прихожая и коридор в виде холла.",
    "овен может держать дверь в квартиру открытой.",
    "раку приписывается ванная или баня.",
    "обитель рака — луна — это уютный дом и ванная.",
    "багуа и секторы квартиры описаны в отдельной главе.",
]


class FakeStore:
    def __init__(self, texts):
        self.texts = texts
        self.embeddings = np.zeros((len(texts), 3), dtype=np.float32)
        self.hash_ids = [f"h{i}" for i in range(len(texts))]

    def get_row(self, hash_id):
        idx = int(hash_id[1:])
        return {
            "hash_id": hash_id,
            "content": self.texts[idx],
            "metadata": [
                {
                    "source_file": "astro.pdf",
                    "doc_title": "astro",
                    "page_num": idx + 1,
                    "section": "",
                }
            ],
        }


class FakeHippoRAG:
    def __init__(self, store):
        self.chunk_embedding_store = store


@pytest.fixture
def store():
    return FakeStore(TEXTS)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("какая зона подходит для Овна?", "овен"),
        ("что важно раку в квартире?", "рак"),
        ("а рыбам что нравится?", "рыбы"),
        ("какой брак лучше?", None),  # «рак» — подстрока «брак»
        ("расскажи про девальвацию", None),  # «дева» — подстрока «девальвация»
        ("какая зона у плиты?", None),  # о зоне без знака
    ],
)
def test_sign_in_query_uses_word_boundaries(query, expected):
    import main

    assert main._sign_in_query(query.lower()) == expected


def test_find_sign_zones_reads_whole_corpus(store):
    import main

    zones = main._find_sign_zones(store, "овен")
    names = [zone["zone"] for zone in zones]

    assert "плита" in names
    assert "прихожая" in names
    # Шум без связующего маркера («овен может держать дверь») не должен попадать.
    assert "дверь" not in names


def test_semfilter_answers_sign_question(store):
    import main

    result = main.search_semantic_filter(FakeHippoRAG(store), "какая зона подходит для Овна?")
    text = result[0][0]

    assert "ОТВЕТ:" in text
    assert "прихожая" in text.lower()
    assert "плита" in text.lower()
    assert "Марс" in text


def test_semfilter_answers_zone_question(store):
    import main

    result = main.search_semantic_filter(FakeHippoRAG(store), "какая планета отвечает за плиту?")
    text = result[0][0]

    assert "ОТВЕТ:" in text
    assert "Марс" in text
    assert "Овен" in text or "овен" in text.lower()


def test_semfilter_refuses_unrelated_question(store):
    import main

    result = main.search_semantic_filter(
        FakeHippoRAG(store), "какие книги написала Максимова И.В.?"
    )
    text = result[0][0]

    assert "Ответ не найден" in text
    assert "ОТВЕТ:" not in text
    assert "Рыбы" not in text


def test_semfilter_refuses_sign_without_zone_pair():
    import main

    store = FakeStore(["овен любит, когда в доме много света и воздуха."])
    result = main.search_semantic_filter(FakeHippoRAG(store), "какая зона у Овна?")
    text = result[0][0]

    assert "Ответ не найден" in text
    assert "ОТВЕТ:" not in text
