from __future__ import annotations

"""
Улучшенный семантический чанкер для PDF документов.
Решает проблемы:
1. Слипшиеся слова без пробелов
2. Обрывки слов в начале/конце чанков
3. Разделение по предложениям, а не механически
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import tiktoken

from .base import ChunkWithCitation

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunkConfig:
    """Конфигурация для семантического чанкера."""

    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    encoding_name: str = "cl100k_base"
    min_chunk_size: int = 100  # Минимальный размер чанка в символах
    sentence_endings: tuple = (".", "!", "?", "。", "！", "？")  # Знаки конца предложения


def clean_pdf_text(text: str) -> str:
    """
    Очистка и нормализация текста из PDF.

    Исправляет:
    - Слипшиеся слова (добавляет пробелы)
    - Переносы строк внутри предложений
    - Множественные пробелы
    - Дефисы переноса
    """
    if not text:
        return ""

    # 1. Убираем дефисы переноса (слово- \nслово -> словослово)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 2. Заменяем переносы строк внутри предложений на пробелы
    # Но сохраняем переносы после знаков препинания
    lines = text.split("\n")
    cleaned_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Если строка заканчивается знаком препинания или это последняя строка
        if i == len(lines) - 1 or line[-1] in ".!?:;":
            cleaned_lines.append(line)
        else:
            # Иначе добавляем пробел вместо переноса
            cleaned_lines.append(line + " ")

    text = "".join(cleaned_lines)

    # 3. УЛУЧШЕННОЕ исправление слипшихся слов
    # 3a. Строчная кириллица + заглавная кириллица
    text = re.sub(r"([а-яё])([А-ЯЁ])", r"\1 \2", text)
    # 3b. Строчная латиница + заглавная латиница
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # 3c. Буква + цифра
    text = re.sub(r"([а-яёА-ЯЁa-zA-Z])(\d)", r"\1 \2", text)
    # 3d. Цифра + буква
    text = re.sub(r"(\d)([а-яёА-ЯЁa-zA-Z])", r"\1 \2", text)
    # 3e. Пунктуация + буква (пробел после знака препинания)
    text = re.sub(r"([.!?,;:])([а-яёА-ЯЁa-zA-Z])", r"\1 \2", text)
    # 3f. Скобка + буква
    text = re.sub(r"([)\]}»])([а-яёА-ЯЁa-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([а-яёА-ЯЁa-zA-Z])([(\[{«])", r"\1 \2", text)

    # 4. Известные паттерны слипшихся слов (безопасные замены)
    known_pairs = [
        (r"Числогуа", "Число гуа"),
        (r"числогуа", "число гуа"),
        (r"Числагуа", "Числа гуа"),
        (r"числагуа", "числа гуа"),
        (r"метазодиака", "мета зодиака"),
        (r"Метазодиака", "Мета зодиака"),
        (r"фэншуй", "фэншуй"),
    ]
    for pattern, replacement in known_pairs:
        text = re.sub(pattern, replacement, text)

    # 5. Убираем множественные пробелы
    text = re.sub(r"\s+", " ", text)

    # 6. Убираем пробелы перед знаками препинания
    text = re.sub(r"\s+([.!?,;:])", r"\1", text)
    # 6b. Пробелы вокруг тире
    text = re.sub(r"\s*—\s*", " — ", text)

    return text.strip()


def split_into_sentences(text: str, sentence_endings: tuple = (".", "!", "?")) -> List[str]:
    """
    Разделяет текст на предложения.

    Args:
        text: Исходный текст
        sentence_endings: Знаки конца предложения

    Returns:
        Список предложений
    """
    # Простое разделение по знакам препинания с учетом пробела после
    pattern = r"([.!?]+)\s+"
    parts = re.split(pattern, text)

    sentences = []
    current = ""

    for i, part in enumerate(parts):
        if i % 2 == 0:  # Текст
            current = part
        else:  # Знак препинания
            current += part
            if current.strip():
                sentences.append(current.strip())
            current = ""

    # Добавляем остаток, если есть
    if current.strip():
        sentences.append(current.strip())

    return sentences


def chunk_text_semantic(
    text: str,
    source_file: str,
    doc_title: str,
    page_num: int,
    section: Optional[str] = None,
    config: Optional[SemanticChunkConfig] = None,
) -> List["ChunkWithCitation"]:
    """
    Семантическое разделение текста на чанки.

    Разделяет по предложениям, а не механически по токенам.
    Гарантирует, что чанки не обрываются посередине слова.

    Args:
        text: Исходный текст
        source_file: Имя файла-источника
        doc_title: Название документа
        page_num: Номер страницы
        section: Раздел документа
        config: Конфигурация чанкера

    Returns:
        Список чанков с метаданными
    """

    if config is None:
        config = SemanticChunkConfig()

    # Очищаем текст
    text = clean_pdf_text(text)

    if not text or len(text) < config.min_chunk_size:
        return []

    try:
        encoder = tiktoken.get_encoding(config.encoding_name)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    # Разделяем на предложения
    sentences = split_into_sentences(text, config.sentence_endings)

    if not sentences:
        # Если не удалось разделить на предложения, используем параграфы
        sentences = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoder.encode(sentence))

        # Если одно предложение больше максимального размера чанка
        if sentence_tokens > config.chunk_size_tokens:
            # Сохраняем текущий чанк, если есть
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    ChunkWithCitation(
                        text=chunk_text.strip(),
                        source_file=source_file,
                        doc_title=doc_title,
                        page_num=page_num,
                        section=section,
                    )
                )
                current_chunk = []
                current_tokens = 0

            # Разбиваем длинное предложение на части по словам
            words = sentence.split()
            temp_chunk = []
            temp_tokens = 0

            for word in words:
                word_tokens = len(encoder.encode(word + " "))
                if temp_tokens + word_tokens > config.chunk_size_tokens and temp_chunk:
                    chunk_text = " ".join(temp_chunk)
                    chunks.append(
                        ChunkWithCitation(
                            text=chunk_text.strip(),
                            source_file=source_file,
                            doc_title=doc_title,
                            page_num=page_num,
                            section=section,
                        )
                    )
                    temp_chunk = []
                    temp_tokens = 0

                temp_chunk.append(word)
                temp_tokens += word_tokens

            if temp_chunk:
                current_chunk = temp_chunk
                current_tokens = temp_tokens

            continue

        # Если добавление предложения превысит лимит
        if current_tokens + sentence_tokens > config.chunk_size_tokens and current_chunk:
            # Сохраняем текущий чанк
            chunk_text = " ".join(current_chunk)
            chunks.append(
                ChunkWithCitation(
                    text=chunk_text.strip(),
                    source_file=source_file,
                    doc_title=doc_title,
                    page_num=page_num,
                    section=section,
                )
            )

            # Начинаем новый чанк с overlap
            # Берем последние предложения для overlap
            overlap_chunk = []
            overlap_tokens = 0

            for prev_sentence in reversed(current_chunk):
                prev_tokens = len(encoder.encode(prev_sentence))
                if overlap_tokens + prev_tokens <= config.chunk_overlap_tokens:
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_tokens += prev_tokens
                else:
                    break

            current_chunk = overlap_chunk
            current_tokens = overlap_tokens

        # Добавляем предложение к текущему чанку
        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Добавляем последний чанк
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= config.min_chunk_size:
            chunks.append(
                ChunkWithCitation(
                    text=chunk_text.strip(),
                    source_file=source_file,
                    doc_title=doc_title,
                    page_num=page_num,
                    section=section,
                )
            )

    logger.info(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
    return chunks
