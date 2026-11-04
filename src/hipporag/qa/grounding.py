"""Контракт заземления Q&A и детерминированный разбор ответа модели.

Модуль не зависит от LLM, HippoRAG и менеджера промптов: это чистые функции,
которые можно проверить без индекса и без сети.

Формат ответа, который ожидает парсер::

    ОТВЕТ: Число гуа 8 связано с землёй [1].
    ИСТОЧНИКИ: 1
    ОТКАЗ: нет

Если ответа нет во фрагментах, модель обязана вернуть ровно
``ABSTAIN_TEXT`` и ``ОТКАЗ: да``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..utils.custom_types import Citation

ABSTAIN_TEXT = "В предоставленных фрагментах нет ответа на этот вопрос."

# [1], [1][2], [1, 2], [1,2]
_CITATION_GROUP_RE = re.compile(r"\[\s*([0-9,\s]+?)\s*\]")
_ANSWER_RE = re.compile(
    r"ОТВЕТ\s*:\s*(.*?)(?=\s*(?:ИСТОЧНИКИ|ОТКАЗ)\s*:|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_SOURCES_RE = re.compile(r"ИСТОЧНИКИ\s*:\s*([0-9,\s]*)", re.IGNORECASE)
_ABSTAIN_RE = re.compile(r"ОТКАЗ\s*:\s*(да|нет)", re.IGNORECASE)
_LEGACY_ANSWER_RE = re.compile(r"(?:Ответ|Answer)\s*:", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

#: Минимальная длина предложения, для которого отсутствие ссылки считается
#: подозрительным. Короткие реплики вроде «Да.» или «Нет.» не штрафуем.
_MIN_UNCITED_SENTENCE_LEN = 40


@dataclass
class GroundedAnswer:
    """Результат разбора ответа модели."""

    text: str
    used_citation_ids: list[int] = field(default_factory=list)
    abstained: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """Ответ либо явный отказ, либо имеет валидные ссылки без ошибок."""
        return self.abstained or (bool(self.used_citation_ids) and not self.errors)


def format_fragments(
    passages: Sequence[str],
    citations: Sequence[Optional[Citation]],
) -> str:
    """Нумерует фрагменты и подписывает источник для промпта.

    Номера начинаются с 1 и совпадают с номерами, которые модель должна
    использовать в маркерах ``[n]``. Если для фрагмента нет метаданных,
    он всё равно получает номер, чтобы ссылка оставалась проверяемой.
    """
    blocks: list[str] = []
    for index, passage in enumerate(passages, 1):
        citation = citations[index - 1] if index - 1 < len(citations) else None
        if citation is not None:
            header = f"[{index}] Источник: {citation.doc_title}, стр. {citation.page_num}"
            if getattr(citation, "section", None):
                header += f", раздел: {citation.section}"
        else:
            header = f"[{index}] Источник: не указан"
        blocks.append(f"{header}\n{passage.strip()}")
    return "\n\n".join(blocks)


def _extract_ids(text: str) -> list[int]:
    ids: list[int] = []
    for group in _CITATION_GROUP_RE.findall(text):
        ids.extend(int(part) for part in re.findall(r"\d+", group))
    return ids


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()


def _extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1)

    # Совместимость со старым форматом «Ответ: ...» / «Answer: ...».
    parts = _LEGACY_ANSWER_RE.split(text, maxsplit=1)
    if len(parts) == 2:
        return parts[1]
    return text


def parse_grounded_response(
    raw: str,
    fragment_count: int,
    strict: bool = True,
) -> GroundedAnswer:
    """Разбирает ответ модели и проверяет ссылки на фрагменты.

    Args:
        raw: сырой текст ответа LLM.
        fragment_count: сколько фрагментов было передано в промпт.
        strict: если True и модель не сослалась ни на один валидный фрагмент,
            ответ заменяется на ``ABSTAIN_TEXT``. Это защита от уверенной
            галлюцинации без доказательства.

    Returns:
        ``GroundedAnswer`` с текстом, использованными номерами и списком
        ошибок. Функция не бросает исключений на мусорном входе.
    """
    text = _strip_code_fences((raw or "").strip())

    answer = " ".join(_extract_answer(text).split()).strip()
    if not answer:
        answer = ABSTAIN_TEXT

    abstain_match = _ABSTAIN_RE.search(text)
    abstained = bool(abstain_match and abstain_match.group(1).lower() == "да")
    if answer.startswith(ABSTAIN_TEXT):
        abstained = True

    inline_ids = _extract_ids(answer)
    source_ids: list[int] = []
    sources_match = _SOURCES_RE.search(text)
    if sources_match:
        source_ids = [int(part) for part in re.findall(r"\d+", sources_match.group(1))]

    all_ids = inline_ids or source_ids
    invalid = sorted({i for i in all_ids if not 1 <= i <= fragment_count})
    used = sorted({i for i in all_ids if 1 <= i <= fragment_count})

    errors: list[str] = []
    if invalid:
        errors.append(f"invalid_citations:{invalid}")

    if strict and not abstained and not inline_ids:
        # «ИСТОЧНИКИ: 1» без маркеров в тексте — это не атрибуция:
        # непонятно, какое утверждение подтверждает фрагмент.
        return GroundedAnswer(
            text=ABSTAIN_TEXT,
            abstained=True,
            errors=errors + ["no_inline_citations"],
        )

    if strict and not abstained and not used:
        return GroundedAnswer(
            text=ABSTAIN_TEXT,
            abstained=True,
            errors=errors + ["no_valid_citations"],
        )

    if inline_ids and source_ids and set(inline_ids) != set(source_ids):
        errors.append(f"source_mismatch:{sorted(set(source_ids))}")

    if not abstained:
        uncited = [
            sentence
            for sentence in _SENTENCE_RE.split(answer)
            if len(sentence) >= _MIN_UNCITED_SENTENCE_LEN and not _extract_ids(sentence)
        ]
        if uncited:
            errors.append(f"uncited_sentences:{len(uncited)}")

    return GroundedAnswer(
        text=answer,
        used_citation_ids=used,
        abstained=abstained,
        errors=errors,
    )
