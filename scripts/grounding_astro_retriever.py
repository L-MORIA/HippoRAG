# -*- coding: utf-8 -*-
"""Ретривер-адаптер astro_app для ``eval_grounding.py --run`` (H2-6, режим 5).

Зачем отдельный адаптер: ``HippoRAG.rag_qa`` выбирает фрагменты своим графовым
ретривом, а продукт показывает результаты режима 5 (BM25 + RU-CE). На golden set
доля gold-страниц в top-5 у них различается в разы, поэтому заземление нужно
мерить на том ретривере, который реально видит пользователь.

Запуск::

    $env:ASTRO_RETRIEVER_TOP_K = "5"
    python scripts/eval_grounding.py --run \
        --retriever grounding_astro_retriever:retrieve

Настройки индекса и модели берутся из окружения (по умолчанию — прод-значения):
``ASTRO_SAVE_DIR``, ``ASTRO_LLM_NAME``, ``ASTRO_LLM_API_NAME``,
``ASTRO_LLM_BASE_URL``, ``ASTRO_RETRIEVER_TOP_K``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_TOP_K = 5
DEFAULT_SAVE_DIR = "outputs/astrology_test"
DEFAULT_LLM_NAME = "gemma-4-e4b"
DEFAULT_LLM_API_NAME = "google/gemma-4-e4b"
DEFAULT_LLM_BASE_URL = "http://localhost:1234/v1"

_ENGINE: Any = None


def _engine() -> Any:
    """Ленивая инициализация HippoRAG: индекс нужен один раз на весь прогон."""
    global _ENGINE
    if _ENGINE is None:
        from astro_app.config import load_hipporag

        _ENGINE = load_hipporag(
            llm_name=os.getenv("ASTRO_LLM_NAME", DEFAULT_LLM_NAME),
            llm_api_name=os.getenv("ASTRO_LLM_API_NAME", DEFAULT_LLM_API_NAME),
            save_dir=os.getenv("ASTRO_SAVE_DIR", DEFAULT_SAVE_DIR),
            llm_base_url=os.getenv("ASTRO_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        )
    return _ENGINE


def retriever_top_k() -> int:
    """Сколько фрагментов запрашивать у ретривера (по умолчанию 5)."""
    raw = os.getenv("ASTRO_RETRIEVER_TOP_K", "").strip()
    if not raw:
        return DEFAULT_TOP_K
    value = int(raw)
    if value < 1:
        raise ValueError(f"ASTRO_RETRIEVER_TOP_K должен быть >= 1, получено {value}")
    return value


def fragment_from_chunk(store: Any, text: str) -> Dict[str, Any]:
    """Превращает текст чанка в фрагмент с источником для ``answer_grounded``.

    Метаданные чанка — список записей (чанк может склеиваться из нескольких
    источников); для цитаты берём первую, как и ``build_citation_from_metadata``.
    """
    hash_id = getattr(store, "text_to_hash_id", {}).get(text)
    entries = store.get_all_metadata_entries(hash_id) if hash_id else []
    primary = entries[0] if entries else {}
    return {
        "text": text,
        "doc_title": str(primary.get("doc_title") or primary.get("source_file") or ""),
        "page_num": int(primary.get("page_num") or 0),
        "section": str(primary.get("section") or ""),
        "source_file": str(primary.get("source_file") or ""),
    }


def retrieve(question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Возвращает фрагменты режима 5 (BM25 + RU-CE) для вопроса."""
    from astro_app.modes.bm25_hybrid import search_bm25_hybrid_ru

    engine = _engine()
    ranked = search_bm25_hybrid_ru(engine, question, top_k=top_k or retriever_top_k())
    store = engine.chunk_embedding_store
    return [fragment_from_chunk(store, row[0]) for row in ranked]
