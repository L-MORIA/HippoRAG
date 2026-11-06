"""Урезанный индекс-фикстура для CI (H1-5).

Зачем. Реальный retrieval на фиксированном индексе в CI не воспроизводится:
индекса в репозитории нет, а эмбеддинг-модель и LM Studio в job'е тоже не
поднять. Фикстура — это урезанный снимок индекса (чанки, их метаданные и
манифест с sha256), который:

* собирается владельцем из своего полного индекса
  (`scripts/build_index_fixture.py`);
* публикуется как CI-артефакт (джоб `Index Fixture Smoke` в
  `.github/workflows/ci-cd.yml`);
* позволяет прогонять **лексический** retrieval (BM25) без моделей, без LLM и
  без сети — это и есть воспроизводимая часть H1-5.

Формат каталога фикстуры:

* `fixture.json` — ``{"meta": {...}, "chunks": [{"hash_id", "content", "metadata"}]}``;
* `manifest.json` — версия, время создания, число чанков и sha256 каждого файла.

Контракт формата проверяется в `tests/unit/test_index_fixture_contract.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .evaluation import _coerce_entries
from .textutils import _tokenize

#: Версия формата фикстуры (меняется только при несовместимой смене структуры).
FIXTURE_VERSION = 1

FIXTURE_FILE = "fixture.json"
MANIFEST_FILE = "manifest.json"

#: Параметры BM25 (стандартные).
_BM25_K1 = 1.5
_BM25_B = 0.75


# ── запись и чтение ─────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    """sha256 файла — для манифеста."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Записать JSON с переводами строк ``\\n`` на любой платформе.

    Это не косметика: ``manifest.json`` хранит sha256 файла фикстуры, а
    ``Path.write_text`` в текстовом режиме на Windows пишет CRLF. Если байты
    зависят от ОС, хеш, посчитанный на машине владельца, не совпадёт с хешем
    файла, выданного git при checkout в CI (там LF) — джоб упадёт на ровном
    месте. Поэтому кодировка и переводы строк задаются явно.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def save_fixture(
    chunks: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Записать фикстуру (чанки + манифест) в каталог ``out_dir``.

    Возвращает манифест.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "version": FIXTURE_VERSION,
            "created_at": created_at,
            "n_chunks": len(chunks),
            **(meta or {}),
        },
        "chunks": [
            {
                "hash_id": str(chunk.get("hash_id") or ""),
                "content": str(chunk.get("content") or ""),
                "metadata": chunk.get("metadata") or {},
            }
            for chunk in chunks
        ],
    }
    fixture_path = out_dir / FIXTURE_FILE
    _write_json(fixture_path, payload)

    manifest = {
        "version": FIXTURE_VERSION,
        "created_at": created_at,
        "n_chunks": len(chunks),
        "files": {FIXTURE_FILE: sha256_file(fixture_path)},
    }
    _write_json(out_dir / MANIFEST_FILE, manifest)
    return manifest


def fixture_path(path: Path) -> Path:
    """Путь к ``fixture.json`` (принимается и каталог, и сам файл)."""
    path = Path(path)
    return path / FIXTURE_FILE if path.is_dir() else path


def load_fixture(path: Path) -> Dict[str, Any]:
    """Загрузить фикстуру."""
    target = fixture_path(path)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "chunks" not in data:
        raise ValueError(f"{target}: не похоже на фикстуру (нет поля 'chunks')")
    return data


def verify_fixture(path: Path) -> List[str]:
    """Проверить фикстуру по манифесту. Пустой список — всё в порядке."""
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    problems: List[str] = []
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.exists():
        return [f"нет манифеста {MANIFEST_FILE}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [f"манифест не читается: {exc}"]

    if manifest.get("version") != FIXTURE_VERSION:
        problems.append(f"версия фикстуры {manifest.get('version')} != ожидаемой {FIXTURE_VERSION}")
    for name, digest in (manifest.get("files") or {}).items():
        target = directory / name
        if not target.exists():
            problems.append(f"файл {name} из манифеста отсутствует")
            continue
        actual = sha256_file(target)
        if actual != digest:
            problems.append(f"sha256 {name}: {actual[:12]}… != ожидаемого {str(digest)[:12]}…")
    return problems


# ── лексический retrieval (BM25) ────────────────────────────────────────────


def chunk_entries(chunk: Dict[str, Any]) -> Tuple[Tuple[str, Optional[int]], ...]:
    """(source_file, page_num) чанка — из его метаданных."""
    return _coerce_entries(chunk.get("metadata"))


def chunk_sources(chunk: Dict[str, Any]) -> List[str]:
    """Список исходных файлов чанка."""
    return [source for source, _ in chunk_entries(chunk)]


def _tokenize_all(chunks: Sequence[Dict[str, Any]]) -> List[List[str]]:
    return [_tokenize(str(chunk.get("content") or "")) for chunk in chunks]


def bm25_rank(chunks: Sequence[Dict[str, Any]], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Ранжировать чанки по BM25 — без моделей и без сети.

    Возвращает список ``{"hash_id", "score", "source_file", "page_num", "rank"}``.
    """
    if not chunks:
        return []
    docs = _tokenize_all(chunks)
    lengths = [len(doc) for doc in docs]
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0.0
    n_docs = len(docs)

    df: Dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    query_terms = _tokenize(query)
    scored: List[Tuple[float, int]] = []
    for idx, doc in enumerate(docs):
        if not doc:
            continue
        counts: Dict[str, int] = {}
        for term in doc:
            counts[term] = counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            denom = freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * (lengths[idx] / (avg_len or 1.0)))
            score += idf * (freq * (_BM25_K1 + 1)) / (denom or 1.0)
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results = []
    for position, (score, idx) in enumerate(scored[: max(top_k, 0)], start=1):
        chunk = chunks[idx]
        entries = chunk_entries(chunk)
        source, page = entries[0] if entries else ("", None)
        results.append(
            {
                "rank": position,
                "index": idx,
                "score": round(score, 4),
                "hash_id": str(chunk.get("hash_id") or ""),
                "source_file": source,
                "page_num": page,
            }
        )
    return results


def smoke_retrieval(
    fixture: Dict[str, Any], questions: Iterable[Dict[str, Any]], top_k: int = 5
) -> Dict[str, Any]:
    """Лексический смоук по размеченным вопросам.

    Для каждого вопроса ищется первый чанк из ожидаемого ``source_file``;
    считается доля попаданий в top-k (recall по документам) — ровно та же
    метрика, что в H1-1, но без эмбеддингов.
    """
    chunks = fixture.get("chunks") or []
    rows: List[Dict[str, Any]] = []
    for question in questions:
        query = str(question.get("question") or "")
        expected = str(question.get("source_file") or "")
        ranked = bm25_rank(chunks, query, top_k=top_k)
        rank = None
        for item in ranked:
            chunk = chunks[item["index"]]
            if expected and expected in chunk_sources(chunk):
                rank = item["rank"]
                break
        rows.append(
            {
                "qid": question.get("qid"),
                "source_file": expected,
                "rank": rank,
                "hit": rank is not None,
                "top": [item["source_file"] for item in ranked[:top_k]],
            }
        )
    hits = sum(1 for row in rows if row["hit"])
    return {
        "n_questions": len(rows),
        "n_hits": hits,
        "recall": round(hits / len(rows), 4) if rows else 0.0,
        "top_k": top_k,
        "rows": rows,
    }
