import ast
import json
import logging
import re
from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Dict, List, Literal

import numpy as np

from .custom_types import Citation, QuerySolution, Triple  # noqa: F401  (реэкспорт, S-18)
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)


def parse_fact_content(content: str) -> Any:
    """Разбирает содержимое факта из embedding-store.

    Факты сохраняются как ``str(tuple)`` (Python-repr), поэтому ``json.loads``
    такой текст не разбирает. Используется ``ast.literal_eval`` — безопасная
    альтернатива ``eval()``; ``json.loads`` остаётся fallback'ом для фактов,
    сохранённых в JSON-формате.
    """
    if not isinstance(content, str):
        return content
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        logger.warning("Не удалось разобрать факт: %r", content[:200])
        return content


FORMULAIC_DATE_YEAR = re.compile(r"^\d{4}\s*(г\.)?(года?)?$", re.IGNORECASE)


def is_valid_entity(name: str) -> bool:
    """Check if entity name is valid (not garbage like pure numbers, too short, etc)."""
    if not name or len(name) < 2:
        return False

    # Skip pure numbers
    if name.isdigit():
        return False

    # Skip entities that are mostly numbers (like "2 87 77")
    words = name.split()
    if all(word.isdigit() for word in words):
        return False

    # Skip single characters
    if len(name) == 1:
        return False

    # Skip entities with only punctuation
    if re.match(r"^[^\w\s]+$", name):
        return False

    # Skip standalone year mentions ("2024", "2024 г.")
    if FORMULAIC_DATE_YEAR.match(name.strip()):
        return False

    return True


def normalize_entity_name(name: str) -> str:
    """Единая нормализация сущностей (S-16): совпадает с text_processing, чтобы одна
    сущность всегда давала один hash-ключ во всех компонентах (граф, факты, NER)."""
    return text_processing(name)


@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal["node", "dpr"]


# Единый QuerySolution объявлен в custom_types (S-18) и здесь только реэкспортируется,
# чтобы star-import из misc_utils не создавал второй класс с другим набором полей.


def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    # Ёё указаны явно: в диапазон А-Яа-я они не входят (S-15)
    text = re.sub("[^A-Za-zА-Яа-яЁё0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def reformat_openie_results(
    corpus_openie_results,
) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):
    ner_output_dict = {
        chunk_item["idx"]: NerRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            unique_entities=list(
                np.unique(
                    [
                        normalize_entity_name(e)
                        for e in chunk_item["extracted_entities"]
                        if is_valid_entity(normalize_entity_name(e))
                    ]
                )
            ),
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item["idx"]: TripleRawOutput(
            chunk_id=chunk_item["idx"],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item["extracted_triples"]),
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict


def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []  # a list of lists of unique entities from each chunk's triples
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities


def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x

    return (x - min_val) / range_val


def build_citation_from_metadata(
    entries: List[Dict], content: str, snippet_len: int = 200
) -> Citation:
    """Строит Citation, несущую ВСЕ источники фрагмента (S-12/D-1).

    `entries` — список метаданных чанка (по записи на документ, где встречается
    этот текст); первичный источник — первый известный.
    """
    entries = entries or []
    primary = entries[0] if entries else {}
    snippet = content[:snippet_len] + "..." if len(content) > snippet_len else content
    return Citation(
        source_file=primary.get("source_file", ""),
        doc_title=primary.get("doc_title", ""),
        page_num=primary.get("page_num", 0),
        section=primary.get("section", ""),
        text_snippet=snippet,
        sources=[e.get("source_file", "") for e in entries],
    )


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
