#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Собрать синтетическую индекс-фикстуру для CI (H1-5).

Зачем отдельный скрипт. ``build_index_fixture.py`` режет **реальный** индекс
владельца, и в фикстуру попадает дословный текст книг корпуса — такую фикстуру
нельзя держать в публичном репозитории. Здесь фикстура собирается из
демонстрационных текстов, зашитых в этот файл: они написаны для проверки
механики retrieval и не являются фрагментами чьих-либо книг.

Что появляется в каталоге ``--out`` (по умолчанию ``fixtures/synthetic_mini``):

* ``fixture.json`` — чанки (``hash_id``, ``content``, ``metadata``);
* ``manifest.json`` — версия формата, время создания, число чанков, sha256;
* ``questions.jsonl`` — разметка под эту фикстуру в формате
  ``eval/dataset/questions.jsonl`` (читается тем же загрузчиком).

Сборка детерминирована: ``CREATED_AT`` — константа, тексты и разметка лежат в
коде, JSON пишется с переводами строк ``\\n``. Значит повторная сборка даёт те
же байты, и ``--check`` может проверить, что закоммиченная фикстура не
разъехалась со скриптом::

    python scripts/make_synthetic_fixture.py --check

Коды возврата: ``0`` — успех (при ``--check`` — файлы совпадают),
``1`` — при ``--check`` файлы разъехались, ``2`` — сломались входные данные.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from astro_app import evaluation as ev  # noqa: E402
from astro_app import index_fixture as fix  # noqa: E402

DEFAULT_OUT = "fixtures/synthetic_mini"

#: Фиксированная метка создания: без неё повторная сборка даёт другие байты,
#: и `--check` не смог бы отличить «фикстуру пересобрали» от «её испортили».
CREATED_AT = "2026-11-06T00:00:00"

QUESTIONS_FILE = "questions.jsonl"

QUESTION_HEADER = (
    "# Синтетическая разметка к fixtures/synthetic_mini (H1-5).\n"
    "# Тексты демонстрационные, не из корпуса владельца.\n"
    "# Файл сгенерирован scripts/make_synthetic_fixture.py — не править вручную:\n"
    "# правка разъедется с manifest.json и уронит CI-джоб Index Fixture Smoke.\n"
)

#: Демонстрационный корпус: (файл, страница, текст чанка).
#: Три «документа» — чтобы смоук проверял не только поиск, но и выбор
#: документа: вопросы ниже целятся в разные файлы, и промах по файлу виден.
DOCS: Tuple[Tuple[str, int, str], ...] = (
    (
        "demo_minerals.pdf",
        1,
        "Кварц и его разновидности. В демонстрационном атласе кварц описан как минерал "
        "с твёрдостью семь по шкале Мооса. Прозрачные друзы кварца встречаются в жилах, "
        "а молочный кварц образует сплошные массы.",
    ),
    (
        "demo_minerals.pdf",
        2,
        "Аметист. Сиреневый оттенок аметиста усиливается при умеренном нагреве до трёхсот "
        "градусов; при перегреве камень теряет цвет и становится жёлтым, поэтому режим "
        "нагрева подбирают осторожно.",
    ),
    (
        "demo_minerals.pdf",
        3,
        "Малахит. Зелёные натёчные агрегаты малахита образуются в зоне окисления медных "
        "руд; рисунок камня напоминает концентрические кольца, и по нему мастера отличают "
        "образцы из разных месторождений.",
    ),
    (
        "demo_minerals.pdf",
        4,
        "Лазурит. Густой васильковый лазурит содержит силикаты серы; вкрапления пирита "
        "придают камню золотистые искры. Однородно окрашенные образцы ценятся выше "
        "пятнистых.",
    ),
    (
        "demo_minerals.pdf",
        5,
        "Обсидиан. Вулканическое стекло обсидиан раскалывается по гладким поверхностям; "
        "древние мастера делали из него наконечники и скребки, потому что край получался "
        "острым без шлифовки.",
    ),
    (
        "demo_minerals.pdf",
        6,
        "Нефрит. Плотный нефрит выдерживает удар лучше гранита; в демо-атласе упомянуты "
        "его вязкость и способность держать полировку после длительной носки.",
    ),
    (
        "demo_minerals.pdf",
        7,
        "Родонит. Розовый родонит часто пронизан чёрными прожилками оксидов марганца; "
        "крупные выделения редки, поэтому большие кабошоны встречаются нечасто.",
    ),
    (
        "demo_minerals.pdf",
        8,
        "Сердолик. Оранжевый сердолик получают прокаливанием халцедона; насыщенность "
        "цвета зависит от содержания оксидов железа в исходном сырье.",
    ),
    (
        "demo_crafts.pdf",
        1,
        "Инструмент. Для огранки применяют планшайбу с абразивом; скорость вращения "
        "подбирают под твёрдость камня, иначе кромка выкрашивается.",
    ),
    (
        "demo_crafts.pdf",
        2,
        "Полировка. Войлочный круг с окисью церия даёт зеркальный блеск на кварце и "
        "халцедоне; перед полировкой поверхность доводят мелким абразивом.",
    ),
    (
        "demo_crafts.pdf",
        3,
        "Сверление. Алмазное сверло охлаждают водой; без охлаждения скол появляется у "
        "края отверстия, а само сверло садится за несколько минут.",
    ),
    (
        "demo_crafts.pdf",
        4,
        "Оправа. Наиболее прочные оправы — цельнолитые; крапановые крепления требуют "
        "аккуратной посадки камня, иначе крапан отгибается при носке.",
    ),
    (
        "demo_crafts.pdf",
        5,
        "Гравировка. Бор машинки оставляет борозду глубиной до миллиметра; тонкую линию "
        "ведут на малых оборотах, чтобы не сколоть кромку рисунка.",
    ),
    (
        "demo_crafts.pdf",
        6,
        "Шлифовка. Последовательность абразивов идёт от крупного к мелкому; если пропустить "
        "шаг, царапины предыдущего зерна не убрать полировкой.",
    ),
    (
        "demo_crafts.pdf",
        7,
        "Пайка. Припой наносится по шву; флюс защищает металл от окисления при нагреве, "
        "а после пайки остатки флюса смывают водой.",
    ),
    (
        "demo_crafts.pdf",
        8,
        "Контроль. Готовое изделие проверяют на просвет и на ощупь; острые кромки оправы "
        "стачивают надфилем, затем деталь промывают.",
    ),
    (
        "demo_climate.pdf",
        1,
        "Осадки. За месяц выпало сорок миллиметров осадков; норма для демо-станции — "
        "тридцать два, поэтому месяц вышел заметно влажнее обычного.",
    ),
    (
        "demo_climate.pdf",
        2,
        "Ветер. Порывы достигали пятнадцати метров в секунду; флюгер фиксировал "
        "юго-западное направление почти каждый день.",
    ),
    (
        "demo_climate.pdf",
        3,
        "Температура. Среднесуточная температура держалась около семи градусов; минимум "
        "отмечен ночью, максимум — во второй половине дня.",
    ),
    (
        "demo_climate.pdf",
        4,
        "Влажность. Относительная влажность превышала восемьдесят процентов; конденсат "
        "оседал на приборах, и наблюдения приходилось проводить в помещении.",
    ),
    (
        "demo_climate.pdf",
        5,
        "Давление. Барометр показывал семьсот сорок миллиметров ртутного столба и "
        "медленно падал весь день, что обычно предвещает смену погоды.",
    ),
    (
        "demo_climate.pdf",
        6,
        "Туман. Туман возникал к утру и рассеивался к полудню; видимость падала до "
        "двухсот метров, поэтому наблюдатели работали парами.",
    ),
    (
        "demo_climate.pdf",
        7,
        "Гроза. Гроза сопровождалась линейными молниями; счётчик зафиксировал двенадцать "
        "разрядов за час, и один из них повредил антенну.",
    ),
    (
        "demo_climate.pdf",
        8,
        "Снег. Первый снег выпал в конце месяца; высота покрова не превысила трёх "
        "сантиметров, и к вечеру он растаял.",
    ),
)

#: Разметка: каждый вопрос целится в свой чанк по словам, которые встречаются
#: только там. Проверено `--check`-прогоном: recall@5 = 1.0.
QUESTIONS: Tuple[Dict[str, Any], ...] = (
    {
        "qid": "syn-mineral-ametit",
        "question": "При каком нагреве сиреневый аметист усиливает оттенок?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_minerals.pdf",
        "page_num": 2,
        "reference_answer": "При умеренном нагреве до трёхсот градусов.",
        "expected_terms": ["аметист", "сиреневый", "нагрев"],
        "status": "verified",
        "notes": "синтетика: проверяет поиск по редкому термину",
    },
    {
        "qid": "syn-mineral-malahit",
        "question": "Какой рисунок напоминает малахит?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_minerals.pdf",
        "page_num": 3,
        "reference_answer": "Концентрические кольца.",
        "expected_terms": ["малахит", "концентрические"],
        "status": "verified",
        "notes": "синтетика",
    },
    {
        "qid": "syn-mineral-lazurit",
        "question": "Какие вкрапления придают лазуриту золотистые искры?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_minerals.pdf",
        "page_num": 4,
        "reference_answer": "Вкрапления пирита.",
        "expected_terms": ["лазурит", "пирит"],
        "status": "verified",
        "notes": "синтетика",
    },
    {
        "qid": "syn-mineral-obsidian",
        "question": "Почему из обсидиана делали древние наконечники?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_minerals.pdf",
        "page_num": 5,
        "reference_answer": "Край получался острым без шлифовки.",
        "expected_terms": ["обсидиан", "наконечники"],
        "status": "verified",
        "notes": "синтетика",
    },
    {
        "qid": "syn-mineral-nefrit",
        "question": "Как нефрит переносит удар по сравнению с гранитом?",
        "kind": "domain",
        "subtype": "comparison",
        "source_file": "demo_minerals.pdf",
        "page_num": 6,
        "reference_answer": "Нефрит выдерживает удар лучше гранита.",
        "expected_terms": ["нефрит", "гранит"],
        "status": "verified",
        "notes": "синтетика",
    },
    {
        "qid": "syn-craft-polirovka",
        "question": "Окись церия на войлочном круге — что она даёт при полировке?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_crafts.pdf",
        "page_num": 2,
        "reference_answer": "Зеркальный блеск на кварце и халцедоне.",
        "expected_terms": ["полировка", "церия"],
        "status": "verified",
        "notes": "синтетика: другой документ",
    },
    {
        "qid": "syn-craft-sverlenie",
        "question": "Чем охлаждают алмазное сверло при сверлении?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_crafts.pdf",
        "page_num": 3,
        "reference_answer": "Водой.",
        "expected_terms": ["сверло", "охлаждение"],
        "status": "verified",
        "notes": "синтетика",
    },
    {
        "qid": "syn-craft-payka",
        "question": "Что защищает металл от окисления при пайке?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_crafts.pdf",
        "page_num": 7,
        "reference_answer": "Флюс.",
        "expected_terms": ["флюс", "пайка"],
        "status": "verified",
        "notes": "синтетика",
    },
    {
        "qid": "syn-climate-osadki",
        "question": "Сколько миллиметров осадков выпало за месяц на демо-станции?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_climate.pdf",
        "page_num": 1,
        "reference_answer": "Сорок миллиметров.",
        "expected_terms": ["осадки", "миллиметров"],
        "status": "verified",
        "notes": "синтетика: третий документ",
    },
    {
        "qid": "syn-climate-tuman",
        "question": "До какой видимости падал туман к утру?",
        "kind": "domain",
        "subtype": "definition",
        "source_file": "demo_climate.pdf",
        "page_num": 6,
        "reference_answer": "До двухсот метров.",
        "expected_terms": ["туман", "видимость"],
        "status": "verified",
        "notes": "синтетика",
    },
)


def build_chunks() -> List[Dict[str, Any]]:
    """Чанки фикстуры из ``DOCS`` — стабильные ``hash_id`` и метаданные."""
    chunks: List[Dict[str, Any]] = []
    for source_file, page_num, content in DOCS:
        slug = Path(source_file).stem.replace("_", "-")
        chunks.append(
            {
                "hash_id": f"syn-{slug}-p{page_num:02d}",
                "content": content,
                "metadata": {"source_file": source_file, "page_num": page_num},
            }
        )
    return chunks


def build_questions() -> List[Dict[str, Any]]:
    """Разметка из ``QUESTIONS`` (копия, чтобы вызывающий не правил константу)."""
    return [dict(question) for question in QUESTIONS]


def write_questions(path: Path, questions: Sequence[Dict[str, Any]]) -> None:
    """Записать ``questions.jsonl`` с заголовком-комментарием и LF."""
    lines = [QUESTION_HEADER]
    for question in questions:
        lines.append(json.dumps(question, ensure_ascii=False, sort_keys=False) + "\n")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(lines))


def build_into(out_dir: Path) -> Dict[str, Any]:
    """Собрать фикстуру и разметку в ``out_dir``. Возвращает манифест."""
    out_dir = Path(out_dir)
    chunks = build_chunks()
    manifest = fix.save_fixture(
        chunks,
        out_dir,
        meta={
            "source": "synthetic",
            "note": (
                "синтетическая фикстура для CI (H1-5): демонстрационные тексты из "
                "scripts/make_synthetic_fixture.py, а не корпус владельца"
            ),
            "documents": sorted({source for source, _, _ in DOCS}),
        },
        created_at=CREATED_AT,
    )
    write_questions(out_dir / QUESTIONS_FILE, build_questions())
    return manifest


def _diff_bytes(name: str, expected: bytes, actual: bytes) -> Optional[str]:
    """Описание расхождения двух файлов (``None`` — совпали)."""
    if expected == actual:
        return None
    limit = min(len(expected), len(actual))
    offset = next((i for i in range(limit) if expected[i] != actual[i]), limit)
    return (
        f"{name}: байты разъехались на позиции {offset} "
        f"({len(actual)} Б в каталоге против {len(expected)} Б после сборки)"
    )


def check(out_dir: Path) -> int:
    """Проверить, что закоммиченная фикстура совпадает со сборкой из кода."""
    out_dir = Path(out_dir)
    names = (fix.FIXTURE_FILE, fix.MANIFEST_FILE, QUESTIONS_FILE)
    missing = [name for name in names if not (out_dir / name).exists()]
    if missing:
        print(f"❌ в {out_dir} нет файлов: {', '.join(missing)}", file=sys.stderr)
        print("   соберите: python scripts/make_synthetic_fixture.py", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="hipporag-synthetic-") as tmp:
        build_into(Path(tmp))
        problems = []
        for name in names:
            expected = (Path(tmp) / name).read_bytes()
            actual = (out_dir / name).read_bytes()
            problem = _diff_bytes(name, expected, actual)
            if problem:
                problems.append(problem)

    if problems:
        print("❌ фикстура разъехалась со скриптом:", file=sys.stderr)
        for problem in problems:
            print(f"   - {problem}", file=sys.stderr)
        print(
            "   пересоберите: python scripts/make_synthetic_fixture.py --out "
            f"{out_dir.as_posix()}",
            file=sys.stderr,
        )
        return 1

    print(f"✅ {out_dir}: файлы совпадают со сборкой из scripts/make_synthetic_fixture.py")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Синтетическая индекс-фикстура для CI (H1-5)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="каталог фикстуры")
    parser.add_argument(
        "--check",
        action="store_true",
        help="не писать, а сверить закоммиченную фикстуру со сборкой из кода",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    if args.check:
        return check(out_dir)

    chunks = build_chunks()
    sources = {chunk["metadata"]["source_file"] for chunk in chunks}
    if len(sources) < 2:
        print("❌ в DOCS меньше двух документов — смоук не проверит выбор файла", file=sys.stderr)
        return 2

    manifest = build_into(out_dir)
    try:
        dataset = ev.load_dataset(out_dir / QUESTIONS_FILE)
    except ev.DatasetError as exc:
        print(f"❌ разметка не проходит загрузчик: {exc}", file=sys.stderr)
        return 2

    questions = dataset.for_run(include_draft=False)
    print(f"Фикстура: {out_dir} — чанков {manifest['n_chunks']}, документов {len(sources)}")
    print(f"Манифест: {manifest['files']}")
    print(f"Разметка: {out_dir / QUESTIONS_FILE} — вопросов {len(questions)}")
    if dataset.incomplete:
        print(f"⚠ незаполненных заготовок: {len(dataset.incomplete)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
