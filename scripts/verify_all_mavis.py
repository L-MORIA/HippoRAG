# -*- coding: utf-8 -*-
"""Независимая проверка 5 критериев Mavis (docs/AUDIT_FINAL_CONCLUSION_2026-11-01.md §5)
+ контроль графового пути (регрессия json.loads в rerank_facts).

Запуск: python scripts/verify_all_mavis.py
"""
import io
import json
import logging
import os
import sys
from pathlib import Path

# Локальная приёмка без ключа: сервер поднимается в режиме fail-open.
# В проде ключ обязателен (API_KEY_REQUIRED=1 по умолчанию).
os.environ.setdefault("API_KEY_REQUIRED", "0")

INDEX_DIR = Path(
    "outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.WARNING)
results = []


def check(num, name, ok, detail):
    results.append((num, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] Критерий {num}: {name}")
    print(f"        {detail}")


# --- Критерий 1: facts > 100 ---
fp = INDEX_DIR / "fact_embeddings" / "vdb_fact.parquet"
facts = pd.read_parquet(fp) if fp.exists() else pd.DataFrame()
n_facts = len(facts)
check(1, "facts > 100 (в новом vdb_fact.parquet)", n_facts > 100, f"facts = {n_facts}")

# --- Критерий 2: synonymy-рёбра > 0 (entity↔entity) ---
import igraph as ig  # noqa: E402

g = ig.Graph.Read_Pickle(str(INDEX_DIR / "graph.pickle"))
n_syn = sum(
    1
    for e in g.es
    if not g.vs[e.source]["name"].startswith("chunk-")
    and not g.vs[e.target]["name"].startswith("chunk-")
)
check(
    2,
    "synonymy-рёбра > 0 в графе",
    n_syn > 0,
    f"entity↔entity = {n_syn} из {g.ecount()} рёбер; узлов: {g.vcount()}",
)

# --- Критерий 4: масштабы сопоставимы ---
n_chunks = len(pd.read_parquet(INDEX_DIR / "chunk_embeddings" / "vdb_chunk.parquet"))
n_entities = len(pd.read_parquet(INDEX_DIR / "entity_embeddings" / "vdb_entity.parquet"))
ratio = max(n_entities, n_facts, n_chunks) / min(n_entities, n_facts, n_chunks)
check(
    4,
    "масштабы сопоставимы (перекос < 10x)",
    ratio < 10,
    f"entities={n_entities}, facts={n_facts}, chunks={n_chunks} (перекос x{ratio:.1f})",
)

# --- Критерий 3: /health facts и без предупреждения о вырожденном факт-сторе ---
from fastapi.testclient import TestClient  # noqa: E402

from hipporag.search_api import server as srv  # noqa: E402
from hipporag.search_api.server import app  # noqa: E402

buf = io.StringIO()
h = logging.StreamHandler(buf)
srv.logger.addHandler(h)
srv.logger.setLevel(logging.WARNING)
with TestClient(app) as client:
    body = client.get("/health").json()
srv.logger.removeHandler(h)
hc, hf = body.get("chunks", -1), body.get("facts", -1)
warn = "Факт-стор почти пуст" in buf.getvalue()
check(
    3,
    "/health отдаёт facts и без предупреждения о вырожденном факт-сторе",
    hf > 100 and not warn,
    f"/health: chunks={hc}, facts={hf}; предупреждение о вырожденном факт-сторе: {warn}",
)

# --- Критерий 5: /search на русском + citations[].sources ---
buf2 = io.StringIO()
h2 = logging.StreamHandler(buf2)
logging.getLogger("hipporag.HippoRAG").addHandler(h2)
logging.getLogger("hipporag.HippoRAG").setLevel(logging.ERROR)
with TestClient(app) as client:
    r = client.post(
        "/search",
        json={"question": "Как обустроить прихожую по фэншуй для Овна?", "top_k": 5},
    )
logging.getLogger("hipporag.HippoRAG").removeHandler(h2)
data = r.json().get("data", {}) if r.status_code == 200 else {}
cites = data.get("citations", [])
docs = data.get("retrieved_docs", [])
sources_ok = bool(cites) and all(
    isinstance(c.get("sources"), list) and c["sources"] for c in cites
)
check(
    5,
    "/search на русском находит фрагмент, citations[].sources — массив имён",
    r.status_code == 200 and bool(docs) and sources_ok,
    f"HTTP {r.status_code}; цитат={len(cites)}; docs={len(docs)}; "
    f"sources[0]={cites[0]['sources'] if cites else '—'}; "
    f"answer[0:90]={str(data.get('answer', ''))[:90]!r}",
)

# --- Контроль: графовый путь (PPR по фактам) не отключён ---
err = "Error in rerank_facts" in buf2.getvalue()
check(
    "Г",
    "графовый путь поиска работает (rerank_facts без ошибок)",
    not err,
    "ошибка rerank_facts отсутствует — факты разбираются, PPR-путь включён"
    if not err
    else "rerank_facts падает — графический путь отключён, идёт dense-retrieval",
)

print()
passed = sum(1 for _, ok in results if ok)
print(f"ИТОГ: {passed}/{len(results)} проверок пройдено")
sys.exit(0 if passed == len(results) else 1)
