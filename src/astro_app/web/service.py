"""Сервис веб-UI H2-1: три режима, заземлённый ответ и цитаты.

UI не дублирует логику поиска: он вызывает те же функции реестра
``astro_app.modes``, что и CLI, а затем прогоняет найденные фрагменты через
общий контракт заземления ``answer_grounded`` (H2-6). Поэтому ответ в браузере
и ответ в CLI проверяются одинаково: либо inline-ссылки ``[n]``, либо явный
отказ.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hipporag.qa.grounding import ABSTAIN_TEXT

from ..grounded_qa import Fragment, answer_grounded, lmstudio_chat
from ..modes import DEFAULT_MODE, MODES
from ..page_retrieval import PageHit
from .models import AskResponse, CitationOut, FragmentOut, ModeInfo


@dataclass(frozen=True)
class UiMode:
    """Продуктовое описание режима: что видит пользователь."""

    id: str
    label: str
    description: str
    expert: bool
    latency: str
    needs_llm: bool


#: Ровно три режима для H2-1: default + два экспертных.
#: Все три уже измерены в H1-3, поэтому UI не обещает невыверенную скорость.
UI_MODES: Dict[str, UiMode] = {
    "5": UiMode(
        id="5",
        label="Быстрый",
        description="BM25 + русский cross-encoder. Обычный поиск по книгам.",
        expert=False,
        latency="~0,2 с",
        needs_llm=False,
    ),
    "2": UiMode(
        id="2",
        label="Граф",
        description="Entity linking + PPR по графу знаний. Для связей между понятиями.",
        expert=True,
        latency="~0,7 с",
        needs_llm=False,
    ),
    "1": UiMode(
        id="1",
        label="Максимальный охват",
        description="Multi-vector + переформулировка запроса через LLM. Медленнее, но находит больше.",
        expert=True,
        latency="~6–13 с",
        needs_llm=True,
    ),
}

#: Режим по умолчанию берётся из реестра, а не дублируется в UI.
DEFAULT_UI_MODE = DEFAULT_MODE if DEFAULT_MODE in UI_MODES else "5"

#: Сколько символов фрагмента уходит в LLM. Ограничение защищает prompt
#: от разрастания при длинных чанках.
MAX_FRAGMENT_CHARS = 1600


@dataclass
class RetrievedFragment:
    """Нормализованный фрагмент из любого режима."""

    text: str
    doc_title: str
    page_num: int
    section: str
    source_file: str
    score: float

    def to_prompt_fragment(self) -> Fragment:
        return Fragment(
            text=self.text,
            doc_title=self.doc_title,
            page_num=self.page_num,
            section=self.section,
            source_file=self.source_file,
        )

    def to_out(self, citation_id: int) -> FragmentOut:
        return FragmentOut(
            id=citation_id,
            text=self.text,
            doc_title=self.doc_title,
            page_num=self.page_num,
            section=self.section or None,
            source_file=self.source_file,
            score=self.score,
        )


@dataclass
class RetrievalResult:
    """Результат поиска без LLM."""

    fragments: List[RetrievedFragment]
    reformulated_query: Optional[str]
    retrieval_time: float


def mode_payload() -> List[ModeInfo]:
    """Три режима в порядке «default, затем экспертные»."""
    ordered = [UI_MODES[DEFAULT_UI_MODE]]
    ordered.extend(mode for mode in UI_MODES.values() if mode.id != DEFAULT_UI_MODE)
    return [
        ModeInfo(
            id=mode.id,
            label=mode.label,
            description=mode.description,
            expert=mode.expert,
            latency=mode.latency,
            needs_llm=mode.needs_llm,
        )
        for mode in ordered
    ]


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _metadata_at(store: Any, index: int) -> Dict[str, Any]:
    """Метаданные чанка по индексу; поддерживает оба формата хранилища."""
    hash_ids = getattr(store, "hash_ids", None) or []
    if not 0 <= index < len(hash_ids):
        return {}

    hash_id = hash_ids[index]
    getter = getattr(store, "get_all_metadata_entries", None)
    if callable(getter):
        entries = getter(hash_id)
        if entries and isinstance(entries[0], dict):
            return dict(entries[0])

    get_row = getattr(store, "get_row", None)
    if not callable(get_row):
        return {}
    row = get_row(hash_id)
    metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    return dict(metadata) if isinstance(metadata, dict) else {}


_TEXT_INDEX_CACHE_KEY: Optional[Tuple[int, int]] = None
_TEXT_INDEX_CACHE: Optional[Dict[str, int]] = None
_CLARIFICATIONS_CACHE: Optional[List[Dict[str, Any]]] = None
_GUA_TABLE_CACHE: Optional[Dict[str, Any]] = None
_GUA_DATE_RE = re.compile(r"(0[1-9]|[12]\d|3[01])\.(0[1-9]|1[0-2])\.((?:19|20)\d{2})")
_GUA_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _text_to_index(store: Any, texts: List[str]) -> Dict[str, int]:
    """Словарь «текст чанка → позиция» (F-12).

    Раньше здесь был ``texts.index(text)`` — линейный проход по всем чанкам
    на каждый фрагмент выдачи. Кэш ключуется store и длиной: после
    reindex/restore каталог подменяется или меняется длина, и словарь
    пересобирается. Коллизии дубликатов неважны: достаточно любой позиции —
    метаданные дубликатов совпадают.
    """
    global _TEXT_INDEX_CACHE_KEY, _TEXT_INDEX_CACHE
    key = (id(store), len(texts))
    if _TEXT_INDEX_CACHE_KEY == key and _TEXT_INDEX_CACHE is not None:
        return _TEXT_INDEX_CACHE
    mapping = {text: index for index, text in enumerate(texts)}
    _TEXT_INDEX_CACHE_KEY = key
    _TEXT_INDEX_CACHE = mapping
    return mapping


def _metadata_for_text(hipporag: Any, text: str) -> Dict[str, Any]:
    """Найти метаданные чанка по точному тексту (режимы 1/2/5)."""
    store = getattr(hipporag, "chunk_embedding_store", None)
    texts = getattr(store, "texts", None) or []
    index = _text_to_index(store, texts).get(text)
    if index is None:
        return {}
    return _metadata_at(store, index)


def _load_gua_table() -> Dict[str, Any]:
    """Таблица «даты → число гуа» (gua_year_table.json из package data).

    Транскрибация таблицы на стр. 9-12 «Эзотерического фэншуя» (в текстовом
    слое PDF её нет — она отрисована изображением). Файл необязателен.
    """
    global _GUA_TABLE_CACHE
    if _GUA_TABLE_CACHE is not None:
        return _GUA_TABLE_CACHE
    from importlib import resources

    try:
        with (
            resources.files("astro_app.data")
            .joinpath("gua_year_table.json")
            .open(encoding="utf-8") as handle
        ):
            _GUA_TABLE_CACHE = json.load(handle)
    except Exception:
        _GUA_TABLE_CACHE = {"rows": []}
    return _GUA_TABLE_CACHE


def _gua_table_fragment(question: str) -> Optional[RetrievedFragment]:
    """F-13d: строка таблицы «год → число гуа» для запросов с датой рождения.

    Таблица книги существует только картинкой, в текстовом корпусе её нет —
    без инъекции модель физически не может увидеть ответ. В фрагмент идут
    строка, покрывающая дату/год запроса, и предыдущая (граница китайского
    Нового года). Только для запросов, где есть «гуа».
    """
    q_lower = (question or "").lower()
    if "гуа" not in q_lower:
        return None

    rows = _load_gua_table().get("rows") or []
    if not rows:
        return None

    from datetime import date, timedelta

    target_date: Optional[date] = None
    date_match = _GUA_DATE_RE.search(q_lower)
    if date_match:
        target_date = date(
            int(date_match.group(3)), int(date_match.group(2)), int(date_match.group(1))
        )
    else:
        year_match = _GUA_YEAR_RE.search(q_lower)
        if year_match:
            target_date = date(int(year_match.group(1)), 7, 1)

    if target_date is None:
        return None

    def row_for(day: date) -> Optional[Dict[str, Any]]:
        for row in rows:
            if row["from"] <= day.isoformat() <= row["to"]:
                return row
        return None

    covering = row_for(target_date)
    if covering is None:
        return None

    # Пол из запроса: считаем подстановку в коде — модель получает готовый
    # факт («для даты X женщина: гуа 5») вместо таблицы, в которой ей нужно
    # самой делать lookup (gemma на таком отказывалась).
    gender_line = ""
    if re.search(r"женщин|родилась|девушк", q_lower):
        gender_line = (
            f"Для даты рождения {target_date.strftime('%d.%m.%Y')} (женщина): "
            f"число гуа — {covering['gua_f']}."
        )
    elif re.search(r"мужчин|родился|парн", q_lower):
        gender_line = (
            f"Для даты рождения {target_date.strftime('%d.%m.%Y')} (мужчина): "
            f"число гуа — {covering['gua_m']}."
        )
    lines = [
        "Таблица чисел гуа (И. В. Максимова, «Эзотерический фэншуй», стр. 9-12): "
        f"период {covering['from']} — {covering['to']} ({covering['animal']}): "
        f"число гуа мужчин — {covering['gua_m']}, число гуа женщин — {covering['gua_f']}."
    ]
    if gender_line:
        lines.insert(0, gender_line)
    # Граница китайского Нового года: рождение до февраля относится к
    # предыдущей строке — показываем её тоже.
    prev_date = target_date - timedelta(days=45)
    if prev_date.year < target_date.year or prev_date.isoformat() < covering["from"]:
        previous = row_for(prev_date)
        if previous is not None and previous is not covering:
            lines.append(
                f"Предыдущий период {previous['from']} — {previous['to']} "
                f"({previous['animal']}): гуа мужчин — {previous['gua_m']}, "
                f"женщин — {previous['gua_f']}. При рождении до даты начала "
                "текущего периода (китайский Новый год) действует предыдущая строка."
            )
    return RetrievedFragment(
        text="\n".join(lines),
        doc_title="Таблица чисел гуа, «Эзотерический фэншуй», стр. 9-12",
        page_num=0,
        section="таблица из книги",
        source_file="book_table",
        score=0.0,
    )


def _load_author_clarifications() -> List[Dict[str, Any]]:
    """Уточнения автора (author_clarifications.json из package data).

    Данные владельца/автора, а не текст издания; загрузка ленивая и
    кэшируется. Файл необязателен: нет файла — нет инъекций.
    """
    global _CLARIFICATIONS_CACHE
    if _CLARIFICATIONS_CACHE is not None:
        return _CLARIFICATIONS_CACHE
    from importlib import resources

    from astro_app import versions

    try:
        versions.check_package_data()
        with (
            resources.files("astro_app.data")
            .joinpath("author_clarifications.json")
            .open(encoding="utf-8") as handle
        ):
            data = json.load(handle)
        _CLARIFICATIONS_CACHE = list(data.get("clarifications", []))
    except Exception:
        _CLARIFICATIONS_CACHE = []
    return _CLARIFICATIONS_CACHE


def _author_clarification_fragment(question: str) -> Optional[RetrievedFragment]:
    """Фрагмент-уточнение автора, если вопрос матчится; иначе None.

    Уточнение добавляется после топ-k с явным источником: заземление
    отвечает с цитатой, пользователь видит происхождение факта.
    """
    q_lower = (question or "").lower()
    if not q_lower.strip():
        return None
    for entry in _load_author_clarifications():
        if not all(str(word) in q_lower for word in entry.get("all", [])):
            continue
        if not any(str(word) in q_lower for word in entry.get("any", [""])):
            continue
        return RetrievedFragment(
            text=entry["text"],
            doc_title=entry.get("source", "Уточнение автора"),
            page_num=0,
            section="вне текущего издания",
            source_file="author_clarification",
            score=0.0,
        )
    return None


def _fragment_from_result(hipporag: Any, result: Any) -> Optional[RetrievedFragment]:
    """Привести строку выдачи любого режима к единому фрагменту."""
    if isinstance(result, PageHit):
        page = result.page
        return RetrievedFragment(
            text=result.text[:MAX_FRAGMENT_CHARS],
            doc_title=page.doc_title or page.source_file or "Источник",
            page_num=page.page_num,
            section=page.section,
            source_file=page.source_file,
            score=float(result.score),
        )

    if isinstance(result, str):
        return RetrievedFragment(
            text=result[:MAX_FRAGMENT_CHARS],
            doc_title="Источник",
            page_num=0,
            section="",
            source_file="",
            score=0.0,
        )

    if not isinstance(result, (list, tuple)) or not result:
        return None

    text = str(result[0] or "").strip()
    if not text:
        return None
    score = _as_float(result[1] if len(result) > 1 else 0.0)

    # Режимы 6/13/21 возвращают (текст, скор, источник, стр.мин, стр.макс, буст).
    if (
        len(result) >= 6
        and isinstance(result[2], str)
        and _as_int(result[3]) is not None
        and _as_int(result[4]) is not None
    ):
        source_file = result[2]
        min_page = _as_int(result[3]) or 0
        max_page = _as_int(result[4]) or min_page
        metadata = _metadata_for_text(hipporag, text)
        doc_title = str(metadata.get("doc_title") or Path(source_file).stem or "Источник")
        section = str(metadata.get("section") or "")
        if not section and max_page > min_page:
            section = f"стр. {min_page}-{max_page}"
        return RetrievedFragment(
            text=text[:MAX_FRAGMENT_CHARS],
            doc_title=doc_title,
            page_num=min_page,
            section=section,
            source_file=source_file or str(metadata.get("source_file") or ""),
            score=score,
        )

    metadata = _metadata_for_text(hipporag, text)
    source_file = str(metadata.get("source_file") or "")
    doc_title = str(metadata.get("doc_title") or Path(source_file).stem or "Источник")
    page_num = _as_int(metadata.get("page_num")) or 0
    return RetrievedFragment(
        text=text[:MAX_FRAGMENT_CHARS],
        doc_title=doc_title,
        page_num=page_num,
        section=str(metadata.get("section") or ""),
        source_file=source_file,
        score=score,
    )


def retrieve(
    hipporag: Any,
    question: str,
    mode_id: str,
    top_k: int = 5,
) -> RetrievalResult:
    """Выполнить поиск выбранным режимом и нормализовать выдачу."""
    if mode_id not in UI_MODES:
        raise ValueError(f"Неизвестный режим: {mode_id}")

    spec = MODES.get(mode_id)
    if spec is None:
        raise ValueError(f"Режим {mode_id} отсутствует в реестре")

    started = time.perf_counter()
    raw = spec.fn(hipporag, question, top_k=top_k)

    reformulated_query: Optional[str] = None
    if spec.extra:
        if isinstance(raw, tuple) and len(raw) == 2:
            raw, reformulated_query = raw
        else:
            reformulated_query = None

    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    fragments: List[RetrievedFragment] = []
    seen: set[Tuple[str, int, str]] = set()
    for item in raw:
        fragment = _fragment_from_result(hipporag, item)
        if fragment is None:
            continue
        key = (fragment.source_file, fragment.page_num, fragment.text[:200])
        if key in seen:
            continue
        seen.add(key)
        fragments.append(fragment)

    fragments = fragments[:top_k]
    clarification = _author_clarification_fragment(question)
    if clarification is not None:
        # Уточнение автора идёт добавкой после топ-k, чтобы не вытеснять
        # фрагменты книги: модель получает его отдельным последним пунктом.
        fragments.append(clarification)
    gua_table = _gua_table_fragment(question)
    if gua_table is not None:
        fragments.append(gua_table)

    return RetrievalResult(
        fragments=fragments,
        reformulated_query=reformulated_query,
        retrieval_time=time.perf_counter() - started,
    )


def answer_question(
    hipporag: Any,
    question: str,
    mode_id: str = DEFAULT_UI_MODE,
    top_k: int = 5,
    chat: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> AskResponse:
    """Найти фрагменты, получить заземлённый ответ и собрать контракт UI."""
    mode = UI_MODES.get(mode_id)
    if mode is None:
        raise ValueError(f"Неизвестный режим: {mode_id}")

    retrieval = retrieve(hipporag, question, mode_id, top_k=top_k)
    prompt_fragments = [fragment.to_prompt_fragment() for fragment in retrieval.fragments]

    empty_llm_calls = 0

    def chat_fn(messages: List[Dict[str, str]]) -> str:
        nonlocal empty_llm_calls
        raw = (chat or lmstudio_chat)(messages)
        if not (raw or "").strip():
            empty_llm_calls += 1
        return raw or ""

    started = time.perf_counter()
    if prompt_fragments:
        grounded = answer_grounded(question, prompt_fragments, chat=chat_fn)
    else:
        # Пустой контекст — честный отказ без обращения к LLM.
        from hipporag.qa.grounding import GroundedAnswer

        grounded = GroundedAnswer(text=ABSTAIN_TEXT, abstained=True)
    answer_time = time.perf_counter() - started

    errors = list(grounded.errors)
    if not prompt_fragments:
        errors.append("no_fragments")
    elif empty_llm_calls and grounded.abstained:
        errors.append("llm_unavailable_or_empty")

    citations: List[CitationOut] = []
    for citation_id in grounded.used_citation_ids:
        if not 1 <= citation_id <= len(retrieval.fragments):
            continue
        fragment = retrieval.fragments[citation_id - 1]
        citations.append(
            CitationOut(
                id=citation_id,
                source_file=fragment.source_file or fragment.doc_title,
                doc_title=fragment.doc_title,
                page_num=fragment.page_num,
                section=fragment.section or None,
                text_snippet=fragment.text[:200],
            )
        )

    if grounded.abstained or not citations:
        confidence = 0.0
    elif errors:
        confidence = 0.5
    else:
        confidence = 1.0

    return AskResponse(
        query=question,
        mode=mode_id,
        mode_label=mode.label,
        expert=mode.expert,
        answer=grounded.text,
        confidence=confidence,
        citations=citations,
        fragments=[fragment.to_out(index) for index, fragment in enumerate(retrieval.fragments, 1)],
        retrieved_docs=[fragment.text for fragment in retrieval.fragments],
        abstained=grounded.abstained,
        grounding_errors=errors,
        reformulated_query=retrieval.reformulated_query,
        processing_time=retrieval.retrieval_time + answer_time,
        retrieval_time=retrieval.retrieval_time,
        answer_time=answer_time,
        timestamp=datetime.now(timezone.utc),
    )
