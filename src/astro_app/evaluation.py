"""Eval-база поиска: разметка, сопоставление выдачи с чанками, метрики (H1-1).

Модуль намеренно не знает про HippoRAG, индекс и LM Studio: здесь только
чистые функции над текстом и метаданными. Всё, что требует загрузки модели или
запуска режимов, живёт в ``scripts/eval_search_quality.py``. Так метрики можно
проверить тестами без индекса и без GPU (см. ``tests/unit/test_eval_metrics.py``).

Три части:

* **разметка** — :class:`Question` и :func:`load_dataset` (JSONL/JSON, строгая
  схема: опечатка в имени поля — ошибка, а не молча проигнорированный вопрос);
* **сопоставление** — :class:`ChunkLookup` переводит текст результата режима
  обратно в чанк (и его ``source_file``/``page_num``), иначе recall@5 по
  документам считать не из чего: режимы возвращают текст, а не id;
* **метрики** — recall@k, MRR@k, nDCG@k, p50/p95, отказы и adequacy.

Почему ground truth только ручной: LLM-судья измеряет совпадение с собственной
моделью мира, а не потребность пользователя. Поэтому ``status`` у вопроса —
``draft`` (черновик, в метрики приёмки не идёт) или ``verified`` (разметка
владельца). По умолчанию скрипт считает только ``verified``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .textutils import _tokenize

# ── Справочники разметки ─────────────────────────────────────────────────────

#: Классы вопросов из плана H1-1.
KINDS = ("bibliography", "domain", "paraphrase")

KIND_LABELS = {
    "bibliography": "библиография",
    "domain": "домен",
    "paraphrase": "перефразировка",
}

#: Подтипы внутри классов («знак → зона», определения, списки, сравнения,
#: число гуа, «где в квартире» + библиографические).
SUBTYPES = (
    "sign-zone",
    "definition",
    "list",
    "comparison",
    "gua-number",
    "where-in-flat",
    "title-author",
    "edition",
    "other",
)

SUBTYPE_LABELS = {
    "sign-zone": "знак → зона",
    "definition": "определение",
    "list": "список",
    "comparison": "сравнение",
    "gua-number": "число гуа",
    "where-in-flat": "где в квартире",
    "title-author": "название/автор",
    "edition": "издание",
    "other": "другое",
}

#: ``draft`` — черновик (в приёмку не идёт), ``verified`` — ручная разметка.
STATUSES = ("draft", "verified")

#: Поля вопроса. Лишнее поле — ошибка: опечатка ``sourse_file`` иначе
#: превратилась бы в пустой ground truth и тихий recall = 0.
QUESTION_KEYS = frozenset(
    {
        "qid",
        "question",
        "kind",
        "subtype",
        "source_file",
        "page_num",
        "reference_answer",
        "expected_terms",
        "status",
        "notes",
    }
)

#: Режимы, которые дергают LLM (проверено по вызовам ``httpx.post`` в
#: ``src/astro_app/modes``). Нужны, чтобы не запускать их вслепую, когда
#: LM Studio недоступна: у каждого таймаут 300 с.
LLM_DEPENDENT_MODES = frozenset({"1", "7", "10", "11", "12", "17", "18", "19", "21"})

#: Режимы, возвращающие готовый ответ, а не ранжированный список чанков.
#: Для них считается adequacy/отказы/latency, а retrieval-метрики — нет.
ANSWER_ONLY_MODES = frozenset({"9", "10", "11", "12", "19", "21"})

#: Минимальная длина фрагмента для сопоставления «текст ⊂ чанк» и наоборот.
MIN_MATCH_CHARS = 40

#: Порог «доля токенов результата, найденных в чанке» для последнего способа
#: сопоставления. Симметричный Jaccard здесь не годится: чанк в 40 раз длиннее
#: результата, и даже точная перефразировка даёт Jaccard ~0.2.
MIN_TOKEN_OVERLAP = 0.6

#: Сколько общих токенов нужно как минимум: иначе короткая фраза из общих слов
#: («квартира и дом») совпадёт с любым чанком.
MIN_TOKEN_HITS = 5

#: Маркеры подсветки предложения из sentence-level режимов (``→…←``).
_MARKERS = ("\u2192", "\u2190")

#: Признаки честного отказа. Русская морфология ломает точное совпадение,
#: поэтому шаблоны — по основам, и проверяется только начало ответа.
_REFUSAL_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"ничего не найдено",
        r"не удалось",
        r"не найден",
        r"не указан",
        r"не содержит",
        r"нет информаци",
        r"нет данных",
        r"информаци[яи] отсутству",
        r"не могу (?:ответить|найти|определить)",
        r"не знаю",
        r"в запросе нет",
    )
)

#: Заглушки режимов, которые формально не пусты, но ответа не содержат.
_EMPTY_PATTERNS = tuple(
    re.compile(p) for p in (r"^ничего не найдено\.?$", r"^нет результатов\.?$", r"^—+$")
)

#: Маркеры начала блока источников в ответах режимов 12/21. Всё после них —
#: найденные чанки, а не ответ: если считать adequacy по ним, метрика начнёт
#: мерить retrieval и «прощать» отказ, в котором источники содержат термины.
_SOURCES_MARKERS = ("=== ИСТОЧНИКИ ===", "=== ИСТОЧНИК ===", "\nИсточники:\n")


class DatasetError(ValueError):
    """Датасет не соответствует схеме: в файле есть ошибки разметки."""


# ── Разметка ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Question:
    """Один размеченный вопрос.

    ``page_num = None`` допустим для вопросов, у которых нет одной страницы
    (например «какие книги есть в индексе»): тогда вопрос участвует в
    документных метриках, но не в страничных.
    """

    qid: str
    question: str
    kind: str
    subtype: str = "other"
    source_file: str = ""
    page_num: Optional[int] = None
    reference_answer: str = ""
    expected_terms: Tuple[str, ...] = ()
    status: str = "draft"
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "kind": self.kind,
            "subtype": self.subtype,
            "source_file": self.source_file,
            "page_num": self.page_num,
            "reference_answer": self.reference_answer,
            "expected_terms": list(self.expected_terms),
            "status": self.status,
            "notes": self.notes,
        }


def is_complete(question: Question) -> bool:
    """Заполнен ли вопрос настолько, чтобы его можно было прогонять."""
    return bool(question.question.strip()) and bool(question.reference_answer.strip())


@dataclass
class Dataset:
    """Загруженный файл разметки.

    ``questions`` — только заполненные вопросы; ``incomplete`` — черновые
    скелеты (они не ломают прогон, но и не считаются: в отчёте видно, сколько
    ещё не размечено).
    """

    path: Path
    questions: List[Question] = field(default_factory=list)
    incomplete: List[Tuple[str, Tuple[str, ...]]] = field(default_factory=list)

    def by_status(self, status: Optional[str] = None) -> List[Question]:
        if status is None:
            return list(self.questions)
        return [q for q in self.questions if q.status == status]

    def for_run(self, include_draft: bool = False) -> List[Question]:
        """Вопросы для прогона: по умолчанию только ``verified``."""
        return self.questions if include_draft else self.by_status("verified")

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def _parse_entry(where: str, raw: Any) -> Question:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: ожидался объект JSON, получено {type(raw).__name__}")

    unknown = sorted(set(raw) - QUESTION_KEYS)
    if unknown:
        raise DatasetError(
            f"{where}: неизвестные поля {', '.join(unknown)} "
            f"(допустимы: {', '.join(sorted(QUESTION_KEYS))})"
        )

    qid = str(raw.get("qid", "")).strip()
    if not qid:
        raise DatasetError(f"{where}: нет qid")
    # Кириллица разрешена: qid читает человек в отчёте, а не файловая система.
    if not re.fullmatch(r"[\w-]{2,64}", qid) or qid.strip("-").strip("_") == "":
        raise DatasetError(
            f"{where}: qid '{qid}' — только буквы/цифры/«_»/«-», 2..64 символа, без пробелов"
        )

    kind = str(raw.get("kind", "")).strip()
    if kind not in KINDS:
        raise DatasetError(f"{where} ({qid}): kind '{kind}' не из {KINDS}")

    subtype = str(raw.get("subtype", "other")).strip() or "other"
    if subtype not in SUBTYPES:
        raise DatasetError(f"{where} ({qid}): subtype '{subtype}' не из {SUBTYPES}")

    status = str(raw.get("status", "draft")).strip() or "draft"
    if status not in STATUSES:
        raise DatasetError(f"{where} ({qid}): status '{status}' не из {STATUSES}")

    source_file = str(raw.get("source_file", "")).strip()
    if not source_file:
        raise DatasetError(f"{where} ({qid}): пустой source_file — ground truth неполный")

    page = raw.get("page_num")
    if page is not None:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise DatasetError(f"{where} ({qid}): page_num={page!r} — нужно целое ≥ 1 или null")

    terms_raw = raw.get("expected_terms") or []
    if not isinstance(terms_raw, (list, tuple)):
        raise DatasetError(f"{where} ({qid}): expected_terms должен быть списком строк")
    terms = []
    for term in terms_raw:
        text = str(term).strip().lower()
        if text and text not in terms:
            terms.append(text)

    return Question(
        qid=qid,
        question=str(raw.get("question", "")).strip(),
        kind=kind,
        subtype=subtype,
        source_file=source_file,
        page_num=page,
        reference_answer=str(raw.get("reference_answer", "")).strip(),
        expected_terms=tuple(terms),
        status=status,
        notes=str(raw.get("notes", "")).strip(),
    )


def load_dataset(path: str | Path, allow_incomplete: bool = True) -> Dataset:
    """Прочитать датасет (``.jsonl`` — по вопросу на строку, ``.json`` — массив).

    Ошибка схемы — исключение: молча пропущенный вопрос смещает метрики
    незаметно. Незаполненные заготовки (нет вопроса или эталонного ответа)
    при ``allow_incomplete=True`` не ошибка — они попадают в ``incomplete``.
    """
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"датасет не найден: {p}")

    text = p.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise DatasetError(f"датасет пуст: {p}")

    entries: List[Tuple[str, Any]] = []
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entries.append((f"строка {lineno}", json.loads(line)))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{p}: строка {lineno}: битый JSON ({exc.msg})") from exc
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{p}: битый JSON ({exc.msg})") from exc
        if isinstance(data, dict) and "questions" in data:
            data = data["questions"]
        if not isinstance(data, list):
            raise DatasetError(f"{p}: ожидался массив вопросов или {{'questions': [...]}}")
        entries = [(f"элемент {i}", item) for i, item in enumerate(data, 1)]

    dataset = Dataset(path=p)
    seen: Dict[str, str] = {}
    errors: List[str] = []

    for where, raw in entries:
        try:
            question = _parse_entry(where, raw)
        except DatasetError as exc:
            errors.append(str(exc))
            continue
        if question.qid in seen:
            errors.append(f"{where}: qid '{question.qid}' уже был в {seen[question.qid]}")
            continue
        seen[question.qid] = where

        if is_complete(question):
            dataset.questions.append(question)
        elif allow_incomplete:
            dataset.incomplete.append((question.qid, tuple(_missing(question))))
        else:
            errors.append(f"{where} ({question.qid}): не заполнено {', '.join(_missing(question))}")

    if errors:
        shown = errors[:20]
        more = f"\n… ещё {len(errors) - len(shown)}" if len(errors) > len(shown) else ""
        raise DatasetError(f"{p}: {len(errors)} ошибок разметки:\n" + "\n".join(shown) + more)

    if not dataset.questions and not dataset.incomplete:
        raise DatasetError(f"{p}: нет ни одного вопроса")
    return dataset


def _missing(question: Question) -> List[str]:
    out = []
    if not question.question.strip():
        out.append("question")
    if not question.reference_answer.strip():
        out.append("reference_answer")
    return out


@dataclass
class ValidationReport:
    """Результат проверки датасета (в т.ч. против конкретного индекса)."""

    total: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    by_subtype: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    incomplete: int = 0
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total": self.total,
            "by_kind": self.by_kind,
            "by_subtype": self.by_subtype,
            "by_status": self.by_status,
            "incomplete": self.incomplete,
            "problems": list(self.problems),
            "warnings": list(self.warnings),
        }

    def format_text(self) -> str:
        lines = [
            f"Вопросов: {self.total} (черновых заготовок: {self.incomplete})",
            "По классам: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_kind.items())),
            "По подтипам: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_subtype.items())),
            "По статусу: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_status.items())),
        ]
        if self.warnings:
            lines.append("Предупреждения:")
            lines.extend(f"  ⚠ {w}" for w in self.warnings)
        if self.problems:
            lines.append("Ошибки:")
            lines.extend(f"  ✗ {p}" for p in self.problems)
        lines.append("Итог: " + ("OK" if self.ok else "ЕСТЬ ОШИБКИ"))
        return "\n".join(lines)


#: Целевой объём разметки из плана H1-1.
TARGET_QUESTIONS = 30


def validate_dataset(dataset: Dataset, lookup: Optional["ChunkLookup"] = None) -> ValidationReport:
    """Проверить разметку; с ``lookup`` — ещё и что страницы существуют в индексе.

    Проверка против индекса обязательна для приёмки: опечатка в ``source_file``
    или несуществующая страница дают recall = 0 и выглядят как «плохой поиск».
    """
    report = ValidationReport(total=len(dataset.questions), incomplete=len(dataset.incomplete))

    for question in dataset.questions:
        report.by_kind[question.kind] = report.by_kind.get(question.kind, 0) + 1
        report.by_subtype[question.subtype] = report.by_subtype.get(question.subtype, 0) + 1
        report.by_status[question.status] = report.by_status.get(question.status, 0) + 1

        if not question.question.rstrip().endswith("?"):
            report.warnings.append(f"{question.qid}: вопрос не заканчивается на «?»")
        if len(question.reference_answer) < 10:
            report.warnings.append(
                f"{question.qid}: эталонный ответ короче 10 символов — проверьте разметку"
            )
        if question.status == "verified" and not question.expected_terms:
            report.warnings.append(
                f"{question.qid}: нет expected_terms — adequacy ответа не посчитать"
            )
        if question.kind == "bibliography" and question.subtype not in ("title-author", "edition"):
            report.warnings.append(
                f"{question.qid}: библиографический вопрос с подтипом '{question.subtype}'"
            )
        if question.kind != "bibliography" and question.subtype in ("title-author", "edition"):
            report.warnings.append(
                f"{question.qid}: подтип '{question.subtype}' у класса '{question.kind}'"
            )

    verified = report.by_status.get("verified", 0)
    if verified < TARGET_QUESTIONS:
        report.warnings.append(
            f"размечено вручную (verified): {verified} из {TARGET_QUESTIONS} — "
            "метрики приёмки неполные"
        )

    if lookup is not None:
        sources = lookup.sources()
        pages = lookup.pages_by_source()
        for question in dataset.questions:
            if question.source_file not in sources:
                report.problems.append(
                    f"{question.qid}: source_file '{question.source_file}' нет в индексе "
                    f"(есть: {', '.join(sorted(sources))})"
                )
                continue
            if question.page_num is not None and question.page_num not in pages.get(
                question.source_file, set()
            ):
                known = pages.get(question.source_file) or set()
                where = f"{min(known)}–{max(known)}" if known else "нет страниц в метаданных"
                report.problems.append(
                    f"{question.qid}: страницы {question.page_num} нет в "
                    f"'{question.source_file}' (страницы: {where})"
                )
            if question.page_num is None:
                report.warnings.append(
                    f"{question.qid}: page_num=null — вопрос только в документных метриках"
                )
    return report


# ── Сопоставление выдачи с чанками ───────────────────────────────────────────


def normalize_text(text: str) -> str:
    """Схлопнуть пробелы и регистр: выдача режимов переформатирует текст."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def strip_markers(text: str) -> str:
    """Убрать ``→…←`` из sentence-level выдачи, иначе текст не совпадёт с чанком."""
    out = str(text)
    for marker in _MARKERS:
        out = out.replace(marker, "")
    return out


def _coerce_entries(meta: Any) -> Tuple[Tuple[str, Optional[int]], ...]:
    """Метаданные чанка → список (source_file, page_num).

    Формат менялся (S-12): dict, список dict'ов или JSON-строка. Один чанк
    может принадлежать нескольким документам — тогда у него несколько записей.
    """
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return ()
    entries: List[Dict[str, Any]] = []
    if isinstance(meta, dict):
        entries = [meta]
    elif isinstance(meta, list):
        entries = [item for item in meta if isinstance(item, dict)]
    out = []
    for entry in entries:
        source = str(entry.get("source_file", "")).strip()
        if not source:
            continue
        page = entry.get("page_num")
        try:
            page = int(page) if page is not None else None
        except (TypeError, ValueError):
            page = None
        if page is not None and page < 1:
            page = None
        out.append((source, page))
    return tuple(out)


@dataclass(frozen=True)
class ChunkRef:
    """Чанк индекса: текст, hash_id и все его источники."""

    hash_id: str
    text: str
    entries: Tuple[Tuple[str, Optional[int]], ...] = ()

    @property
    def source_file(self) -> str:
        return self.entries[0][0] if self.entries else ""

    @property
    def page_num(self) -> Optional[int]:
        return self.entries[0][1] if self.entries else None


class ChunkLookup:
    """Перевод текста результата обратно в чанк(и).

    Режимы возвращают ``(text, score, ...)``, а не id чанка, причём текст
    бывает изменён: sentence-level вставляет ``→…←``, small→big склеивает
    соседей, sign-режимы добавляют заголовки. Поэтому сопоставление идёт по
    убыванию точности: md5 → точный текст → подстрока → пересечение токенов.
    """

    def __init__(self, chunks: Sequence[ChunkRef]):
        self.chunks: List[ChunkRef] = list(chunks)
        self._by_hash = {c.hash_id: c for c in self.chunks}
        self._by_text = {normalize_text(c.text): c for c in self.chunks}
        self._norm = [normalize_text(c.text) for c in self.chunks]
        self._pages: Dict[str, set] = {}
        for chunk in self.chunks:
            for source, page in chunk.entries:
                if page is not None:
                    self._pages.setdefault(source, set()).add(page)
        self._token_sets = [frozenset(_tokenize(t)) for t in self._norm]

    def sources(self) -> set:
        return {source for chunk in self.chunks for source, _ in chunk.entries}

    def pages_by_source(self) -> Dict[str, set]:
        return {source: set(pages) for source, pages in self._pages.items()}

    def match(self, text: str) -> List[ChunkRef]:
        """Чанки, из которых состоит текст результата (пусто — не сопоставился)."""
        raw = str(text or "")
        cleaned = strip_markers(raw)
        norm = normalize_text(cleaned)
        if not norm:
            return []

        # 1. md5 — точное совпадение с исходным текстом чанка.
        for candidate in (raw, cleaned):
            hit = self._by_hash.get("chunk-" + hashlib.md5(candidate.encode()).hexdigest())
            if hit is not None:
                return [hit]

        # 2. Точный текст после нормализации (регистр/пробелы).
        hit = self._by_text.get(norm)
        if hit is not None:
            return [hit]

        # 3. Результат содержит чанк(и) — склейка small→big, вставки заголовков.
        if len(norm) >= MIN_MATCH_CHARS:
            contained = [
                chunk
                for chunk, chunk_norm in zip(self.chunks, self._norm)
                if len(chunk_norm) >= MIN_MATCH_CHARS and chunk_norm in norm
            ]
            if contained:
                return contained

        # 4. Результат — фрагмент чанка (обрезанная выдача).
        if len(norm) >= MIN_MATCH_CHARS:
            for chunk, chunk_norm in zip(self.chunks, self._norm):
                if norm in chunk_norm:
                    return [chunk]

        # 5. Последний шанс: пересечение токенов (перефразированный текст).
        return self._match_by_tokens(norm)

    def _match_by_tokens(self, norm: str) -> List[ChunkRef]:
        # Служебные слова («и», «в», «не») дают ложные совпадения: считаем
        # покрытие только по содержательным токенам.
        tokens = frozenset(t for t in _tokenize(norm) if len(t) >= 3)
        if len(tokens) < MIN_TOKEN_HITS:
            return []
        best: List[Tuple[float, int]] = []
        for idx, chunk_tokens in enumerate(self._token_sets):
            if not chunk_tokens:
                continue
            common = tokens & chunk_tokens
            # Доля покрытия считается от результата: чанк длиннее в десятки раз.
            coverage = len(common) / len(tokens)
            if coverage >= MIN_TOKEN_OVERLAP and len(common) >= MIN_TOKEN_HITS:
                best.append((coverage, idx))
        best.sort(reverse=True)
        return [self.chunks[idx] for _, idx in best[:3]]


def build_chunk_lookup(chunk_store: Any) -> ChunkLookup:
    """Построить lookup по хранилищу чанков HippoRAG (``chunk_embedding_store``)."""
    hash_ids = list(getattr(chunk_store, "hash_ids", []) or [])
    texts = list(getattr(chunk_store, "texts", []) or [])
    metadatas = list(getattr(chunk_store, "metadatas", []) or [])
    if not (len(hash_ids) == len(texts) == len(metadatas)):
        raise ValueError("хранилище чанков повреждено: длины hash_ids/texts/metadatas не совпадают")
    chunks = [
        ChunkRef(hash_id=hid, text=text, entries=_coerce_entries(meta))
        for hid, text, meta in zip(hash_ids, texts, metadatas)
    ]
    return ChunkLookup(chunks)


def build_lookup_from_records(records: Iterable[Dict[str, Any]]) -> ChunkLookup:
    """То же из «сырых» записей ``{hash_id, content, metadata}`` (parquet, тесты)."""
    chunks = []
    for record in records:
        text = str(record.get("content", ""))
        hash_id = str(record.get("hash_id") or "chunk-" + hashlib.md5(text.encode()).hexdigest())
        chunks.append(
            ChunkRef(
                hash_id=hash_id, text=text, entries=_coerce_entries(record.get("metadata", ()))
            )
        )
    return ChunkLookup(chunks)


# ── Метрики ──────────────────────────────────────────────────────────────────


def first_hit_rank(flags: Sequence[bool]) -> Optional[int]:
    """Номер первой позиции с совпадением (1-based) или ``None``."""
    for position, flag in enumerate(flags, 1):
        if flag:
            return position
    return None


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """Доля вопросов, у которых попадание не дальше ``k``."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """Mean reciprocal rank: 1/rank первого попадания, 0 если дальше ``k``."""
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank is not None and 1 <= rank <= k) / len(ranks)


def ndcg_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """nDCG@k с бинарной релевантностью (эталонный чанк один, IDCG = 1)."""
    if not ranks:
        return 0.0
    return sum(1.0 / math.log2(rank + 1) for rank in ranks if rank is not None and rank <= k) / len(
        ranks
    )


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Перцентиль с линейной интерполяцией (как numpy.percentile)."""
    data = sorted(float(v) for v in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100.0
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return data[low]
    return data[low] + (data[high] - data[low]) * (position - low)


def is_empty(text: str) -> bool:
    """Ответа нет: пустая строка или заглушка режима."""
    stripped = str(text or "").strip()
    if not stripped:
        return True
    return any(pattern.match(stripped.lower()) for pattern in _EMPTY_PATTERNS)


def is_refusal(text: str) -> bool:
    """Честный отказ («материал по знаку не найден», «не удалось получить ответ»).

    Проверяется только начало: в длинном ответе «не найден» может встретиться
    в содержательной фразе.
    """
    head = normalize_text(text)[:300]
    return any(pattern.search(head) for pattern in _REFUSAL_PATTERNS)


def term_matches(term: str, text_norm: str) -> bool:
    """Есть ли термин в тексте с учётом русской морфологии (по основе).

    «прихожая» должна находиться в «прихожей»: точного вхождения нет, а
    обрезание окончания даёт основу «прихож».
    """
    term = term.strip().lower()
    if not term:
        return False
    if term in text_norm:
        return True
    if len(term) >= 5:
        return term[: max(4, len(term) - 2)] in text_norm
    return False


def answer_body(text: str) -> str:
    """Ответ без блока источников (режимы 12/21 приклеивают найденные чанки)."""
    out = str(text or "")
    cut = len(out)
    for marker in _SOURCES_MARKERS:
        index = out.find(marker)
        if 0 <= index < cut:
            cut = index
    return out[:cut].strip()


def answer_adequacy(
    text: str, expected_terms: Sequence[str], min_chars: int = 80
) -> Optional[float]:
    """Насколько ответ содержателен: покрытие терминов × длина (0..1).

    Это не LLM-судья и не оценка истинности: прокси, который ловит пустые и
    уклончивые ответы. Полная проверка фактов — ручная, по эталону.
    """
    if not expected_terms:
        return None
    body = answer_body(text)
    if is_empty(body) or is_refusal(body):
        return 0.0
    norm = normalize_text(body)
    covered = sum(1 for term in expected_terms if term_matches(term, norm))
    coverage = covered / len(expected_terms)
    length_factor = min(1.0, len(norm) / float(min_chars))
    return round(coverage * length_factor, 3)


# ── Сводка по режиму ─────────────────────────────────────────────────────────


def summarize_mode(
    rows: Sequence[Dict[str, Any]],
    *,
    mode_id: str,
    label: str,
    k: int = 5,
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> Dict[str, Any]:
    """Собрать метрики режима из построчных результатов.

    Retrieval-метрики считаются только по строкам с ``retrieval=True``:
    у режимов-ответов (9, 10, 11, 12, 19, 21) ранжированного списка нет, и
    подмешивать их нулями нельзя — метрика перестанет быть сравнимой.
    """
    retrieval_rows = [r for r in rows if r.get("retrieval")]
    # Страничные метрики — только там, где страница вообще размечена: иначе
    # вопросы с page_num=null разбавляли бы recall@5 нулями.
    page_rows = [r for r in retrieval_rows if r.get("page_expected")]
    answer_rows = [r for r in rows if r.get("answer_text") is not None]
    errors = [r for r in rows if r.get("error")]

    latencies = [float(r.get("latency_ms") or 0.0) for r in rows if not r.get("error")]
    # Время вне LLM-вызовов: для retrieval-режимов это и есть поиск, для
    # режимов-ответов — оценка retrieval-части без недетерминизма модели.
    non_llm_latencies = [
        float(r.get("latency_ms") or 0.0) - float(r.get("llm_ms") or 0.0)
        for r in rows
        if not r.get("error")
    ]
    candidate_counts = [int(r.get("n_candidates") or 0) for r in retrieval_rows]

    def _rate(values: Sequence[bool]) -> Optional[float]:
        return round(sum(1 for v in values if v) / len(values), 4) if values else None

    def _mean(values: Sequence[float]) -> Optional[float]:
        return round(sum(values) / len(values), 4) if values else None

    adequacy = [float(r["adequacy"]) for r in answer_rows if r.get("adequacy") is not None]
    answer_texts = [r.get("answer_text") or "" for r in answer_rows]

    doc_ranks = [r.get("doc_rank") for r in retrieval_rows]
    page_ranks = [r.get("page_rank") for r in page_rows]

    return {
        "mode": {"id": mode_id, "label": label},
        "n_questions": len(rows),
        "n_retrieval": len(retrieval_rows),
        "n_answer": len(answer_rows),
        "n_errors": len(errors),
        "errors": [f"{r.get('qid')}: {r.get('error')}" for r in errors][:10],
        # None, а не 0.0: у режимов-ответов ранжированного списка нет вообще,
        # и ноль в recall выглядел бы как «поиск не нашёл».
        "recall@k_doc": round(recall_at_k(doc_ranks, k), 4) if doc_ranks else None,
        "recall@k_page": round(recall_at_k(page_ranks, k), 4) if page_ranks else None,
        "mrr@k_doc": round(mrr_at_k(doc_ranks, mrr_k), 4) if doc_ranks else None,
        "mrr@k_page": round(mrr_at_k(page_ranks, mrr_k), 4) if page_ranks else None,
        "ndcg@k_doc": round(ndcg_at_k(doc_ranks, ndcg_k), 4) if doc_ranks else None,
        "refusal_rate": _rate([is_refusal(t) for t in answer_texts]),
        "empty_rate": _rate([is_empty(t) for t in answer_texts]),
        "adequacy_mean": _mean(adequacy),
        "adequate_rate": (
            round(sum(1 for v in adequacy if v >= 0.5) / len(adequacy), 4) if adequacy else None
        ),
        "latency_ms": {
            "p50": _round(percentile(latencies, 50)),
            "p95": _round(percentile(latencies, 95)),
            "mean": _round(_mean(latencies)),
        },
        "retrieval_ms": {
            "p50": _round(percentile(non_llm_latencies, 50)),
            "p95": _round(percentile(non_llm_latencies, 95)),
        },
        "llm_calls_mean": _round(
            _mean([float(r.get("llm_calls") or 0) for r in rows if not r.get("error")])
        ),
        "n_candidates": {
            "mean": _round(_mean([float(c) for c in candidate_counts])),
            "min": min(candidate_counts) if candidate_counts else None,
            "max": max(candidate_counts) if candidate_counts else None,
            # Вырожденная нормировка: при n<3 min_max_normalize даёт всем 1.0,
            # поэтому такие прогоны нельзя сравнивать с остальными.
            "lt3": sum(1 for c in candidate_counts if c < 3),
        },
        "k": k,
        "mrr_k": mrr_k,
        "ndcg_k": ndcg_k,
    }


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# ── Таблицы ──────────────────────────────────────────────────────────────────


def _fmt(value: Optional[float], digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def format_markdown_table(summaries: Sequence[Dict[str, Any]]) -> str:
    """Таблица для ``docs/INDEXING_GUIDE.md``: retrieval-режимы и режимы-ответы."""
    retrieval = [s for s in summaries if s.get("n_retrieval")]
    answers = [s for s in summaries if s.get("n_answer")]

    lines: List[str] = []
    if retrieval:
        lines += [
            "| Режим | n | recall@5 (док) | recall@5 (стр) | MRR@10 (док) | nDCG@10 | кеши <3 | p50, мс | p95, мс |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in retrieval:
            lines.append(
                "| {mode} | {n} | {rd} | {rp} | {mrr} | {ndcg} | {lt3} | {p50} | {p95} |".format(
                    mode=f"{s['mode']['id']} {s['mode']['label']}",
                    n=s["n_retrieval"],
                    rd=_fmt(s.get("recall@k_doc"), percent=True),
                    rp=_fmt(s.get("recall@k_page"), percent=True),
                    mrr=_fmt(s.get("mrr@k_doc")),
                    ndcg=_fmt(s.get("ndcg@k_doc")),
                    lt3=s["n_candidates"].get("lt3"),
                    p50=_fmt((s.get("latency_ms") or {}).get("p50"), digits=0),
                    p95=_fmt((s.get("latency_ms") or {}).get("p95"), digits=0),
                )
            )

    if answers:
        if lines:
            lines.append("")
        lines += [
            "| Режим | n | ответы | отказы | пусто | adequacy | p50, мс | p95, мс | LLM-вызовов |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for s in answers:
            lines.append(
                "| {mode} | {n} | {na} | {ref} | {empty} | {adq} | {p50} | {p95} | {llm} |".format(
                    mode=f"{s['mode']['id']} {s['mode']['label']}",
                    n=s["n_questions"],
                    na=s["n_answer"],
                    ref=_fmt(s.get("refusal_rate"), percent=True),
                    empty=_fmt(s.get("empty_rate"), percent=True),
                    adq=_fmt(s.get("adequacy_mean")),
                    p50=_fmt((s.get("latency_ms") or {}).get("p50"), digits=0),
                    p95=_fmt((s.get("latency_ms") or {}).get("p95"), digits=0),
                    llm=_fmt(s.get("llm_calls_mean"), digits=2),
                )
            )
    return "\n".join(lines)
