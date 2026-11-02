"""Приёмка переиндексации по 5 критериям Mavis (docs/AUDIT_FINAL_CONCLUSION_2026-11-01.md, §5).

Запуск: python scripts/verify_reindex.py
Выход: построчные PASS/FAIL по каждому критерию, итоговый exit-код (0 = все PASS).
"""

import io
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

INDEX_DIR = Path(
    "outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

results = []


def check(num, name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] Критерий {num}: {name} — {detail}")


def norm_sources(meta_value):
    """Метаданные могут быть JSON-строкой (старый формат) или списком."""
    if isinstance(meta_value, str):
        try:
            meta_value = json.loads(meta_value)
        except json.JSONDecodeError:
            return []
    if isinstance(meta_value, dict):
        meta_value = [meta_value]
    return [m.get("source_file", "") for m in (meta_value or []) if isinstance(m, dict)]


# --- Критерий 1: facts > 100 ---
fact_path = INDEX_DIR / "fact_embeddings" / "vdb_fact.parquet"
n_facts = len(pd.read_parquet(fact_path)) if fact_path.exists() else 0
check(1, "facts > 100", n_facts > 100, f"facts = {n_facts}")

# --- Счётчики для критериев 4 и 5 ---
n_chunks = 0
n_entities = 0
if (INDEX_DIR / "chunk_embeddings" / "vdb_chunk.parquet").exists():
    n_chunks = len(pd.read_parquet(INDEX_DIR / "chunk_embeddings" / "vdb_chunk.parquet"))
if (INDEX_DIR / "entity_embeddings" / "vdb_entity.parquet").exists():
    n_entities = len(pd.read_parquet(INDEX_DIR / "entity_embeddings" / "vdb_entity.parquet"))

# --- Критерий 2: synonymy-рёбра > 0 (entity↔entity, а не только entity→passage) ---
graph_path = INDEX_DIR / "graph.pickle"
n_syn = -1
n_total_edges = 0
if graph_path.exists():
    import igraph as ig

    g = ig.Graph.Read_Pickle(str(graph_path))
    chunk_prefix = "chunk-"
    # e.source/e.target — индексы вершин; имя берём из атрибута вершины
    n_syn = sum(
        1
        for e in g.es
        if not g.vs[e.source]["name"].startswith(chunk_prefix)
        and not g.vs[e.target]["name"].startswith(chunk_prefix)
    )
    n_total_edges = g.ecount()
check(2, "synonymy-рёбра > 0", n_syn > 0, f"entity↔entity рёбер: {n_syn} из {n_total_edges}")

# --- Критерий 3 + 4: /health (chunks, facts) и масштабы ---
health_chunks = health_facts = -1
warn_found = None
try:
    from fastapi.testclient import TestClient

    from hipporag.search_api.server import app

    with TestClient(app) as client:  # контекст запускает lifespan
        r = client.get("/health")
        body = r.json()
        health_chunks, health_facts = body.get("chunks", -1), body.get("facts", -1)
    # перехват_WARNING из лога lifespan: факт-стор почти пуст
    import io

    from hipporag.search_api import server as srv

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    srv.logger.addHandler(h)
    srv.logger.setLevel(logging.WARNING)
    with TestClient(app):
        pass
    srv.logger.removeHandler(h)
    warn_found = "Факт-стор почти пуст" in buf.getvalue()
except Exception as e:  # noqa: BLE001
    print(f"  (TestClient недоступен: {e})")

crit3_ok = health_facts > 100 and warn_found is False
check(3, "/health facts и без предупреждения", crit3_ok, f"/health: chunks={health_chunks}, facts={health_facts}; предупреждение: {warn_found}")

if all(v > 0 for v in (n_chunks, n_entities, n_facts)):
    ratio = max(n_entities, n_facts, n_chunks) / min(n_entities, n_facts, n_chunks)
    crit4_ok = ratio < 10
    check(4, "масштабы сопоставимы (перекос < 10x)", crit4_ok, f"entities={n_entities}, facts={n_facts}, chunks={n_chunks} (перекос x{ratio:.1f})")
else:
    check(4, "масштабы сопоставимы", False, f"entities={n_entities}, facts={n_facts}, chunks={n_chunks}")

# --- Критерий 5: /search на русском + citations[].sources ---
try:
    from fastapi.testclient import TestClient

    from hipporag.search_api.server import app

    with TestClient(app) as client:
        r = client.post("/search", json={"question": "Как обустроить прихожую по фэншуй для Овна?", "top_k": 5})
        ok_http = r.status_code == 200
        data = r.json().get("data", {}) if ok_http else {}
        cites = data.get("citations", [])
        has_docs = len(data.get("retrieved_docs", [])) > 0
        sources_ok = len(cites) > 0 and all(isinstance(c.get("sources"), list) and len(c["sources"]) > 0 for c in cites)
        crit5_ok = ok_http and has_docs and sources_ok
        detail = f"HTTP {r.status_code}, цитат: {len(cites)}, docs: {len(data.get('retrieved_docs', []))}, sources пример: {cites[0]['sources'][:2] if cites else '—'}"
except Exception as e:  # noqa: BLE001
    crit5_ok = False
    detail = f"ошибка: {e}"
check(5, "/search на русском + citations[].sources", crit5_ok, detail)

print()
print(f"ИТОГ: {sum(results)}/5 критериев пройдено")
sys.exit(0 if all(results) else 1)
