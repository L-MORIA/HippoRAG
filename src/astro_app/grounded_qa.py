"""Единый путь Q&A с заземлением для генеративных режимов CLI (H2-6).

Режимы `astro_app.modes` раньше собирали собственные промпты и теряли связь
между ответом и источниками. Здесь один контракт: фрагменты нумеруются,
модель обязана ссылаться на номера, парсер проверяет ссылки, а отсутствие
доказательства превращается в общий отказ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from hipporag.prompts.templates.rag_qa_astrology import rag_qa_system
from hipporag.qa.grounding import (
    ABSTAIN_TEXT,
    GroundedAnswer,
    format_fragments,
    parse_grounded_response,
)
from hipporag.utils.custom_types import Citation

from .config import LLM_BASE_URL, LLM_NAME

ChatFn = Callable[[List[Dict[str, str]]], str]


@dataclass
class Fragment:
    """Один фрагмент контекста с происхождением для цитаты."""

    text: str
    doc_title: str = "Источник"
    page_num: int = 0
    section: str = ""
    source_file: str = ""

    def to_citation(self) -> Citation:
        return Citation(
            source_file=self.source_file or self.doc_title,
            doc_title=self.doc_title,
            page_num=self.page_num,
            section=self.section or None,
            text_snippet=self.text[:200],
        )


def build_messages(question: str, fragments: Sequence[Fragment]) -> List[Dict[str, str]]:
    """Собирает system + user сообщения по единому контракту."""
    citations = [fragment.to_citation() for fragment in fragments]
    prompt_user = format_fragments([fragment.text for fragment in fragments], citations)
    prompt_user += "\n\nВопрос: " + question + "\n"
    return [
        {"role": "system", "content": rag_qa_system},
        {"role": "user", "content": prompt_user},
    ]


def answer_grounded(
    question: str,
    fragments: Sequence[Fragment],
    chat: Optional[ChatFn] = None,
    strict: bool = True,
) -> GroundedAnswer:
    """Спрашивает LLM по фрагментам и возвращает проверенный ответ."""
    if not fragments:
        return GroundedAnswer(text=ABSTAIN_TEXT, abstained=True)

    chat_fn = chat or lmstudio_chat
    raw = chat_fn(build_messages(question, fragments))
    return parse_grounded_response(raw, fragment_count=len(fragments), strict=strict)


def used_sources(fragments: Sequence[Fragment], citation_ids: Sequence[int]) -> str:
    """Строки источников только для тех фрагментов, на которые сослались."""
    lines = []
    for citation_id in citation_ids:
        if not 1 <= citation_id <= len(fragments):
            continue
        fragment = fragments[citation_id - 1]
        line = f"[{citation_id}] {fragment.doc_title}, стр. {fragment.page_num}"
        if fragment.section:
            line += f", {fragment.section}"
        lines.append(line)
    return "\n".join(lines)


def lmstudio_chat(
    messages: List[Dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0.15,
    timeout: int = 300,
) -> str:
    """Запрос к LM Studio; при недоступности возвращает пустую строку."""
    import httpx

    api_url = LLM_BASE_URL.replace("/v1", "") if "/v1" in LLM_BASE_URL else LLM_BASE_URL
    try:
        health = httpx.get(f"{api_url}/v1/models", timeout=3)
        if health.status_code != 200:
            return ""
        response = httpx.post(
            f"{api_url}/v1/chat/completions",
            json={
                "model": LLM_NAME,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return ""
        data = response.json()
    except Exception:
        return ""

    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if content:
        return content

    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if reasoning:
        lines = [line.strip() for line in reasoning.split("\n") if line.strip()]
        if lines:
            return lines[-1].strip("\"'").replace("**", "")
    return ""
