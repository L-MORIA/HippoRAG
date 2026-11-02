from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import tiktoken

from .base import ChunkWithCitation


@dataclass
class ChunkConfig:
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    encoding_name: str = "cl100k_base"


def chunk_text_with_metadata(
    text: str,
    source_file: str,
    doc_title: str,
    page_num: int,
    section: Optional[str] = None,
    config: Optional[ChunkConfig] = None,
) -> List["ChunkWithCitation"]:
    if config is None:
        config = ChunkConfig()

    try:
        encoder = tiktoken.get_encoding(config.encoding_name)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    tokens = encoder.encode(text)
    chunk_size = config.chunk_size_tokens
    overlap = config.chunk_overlap_tokens

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)

        # Все чанки получены с одной страницы — номер страницы не модифицируется
        chunk = ChunkWithCitation(
            text=chunk_text.strip(),
            source_file=source_file,
            doc_title=doc_title,
            page_num=page_num,
            section=section,
        )
        chunks.append(chunk)

        start += chunk_size - overlap

        if start >= len(tokens):
            break

    return chunks
