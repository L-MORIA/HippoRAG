"""Приёмка переиндексации по 5 критериям Mavis (docs/AUDIT_FINAL_CONCLUSION_2026-11-01.md, §5).

Запуск: python scripts/verify_reindex.py
Выход: построчные PASS/FAIL по каждому критерию, итоговый exit-код (0 = все PASS).

Приложение поднимается РОВНО ОДИН РАЗ (один lifespan): критерии 3 и 5 — из одной
TestClient-сессии, критерии 1, 2, 4 — по parquet/graph без сервера.
"""

import io
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Локальная приёмка без ключа: сервер поднимается в режиме fail-open.
# В проде ключ обязателен (API_KEY_REQUIRED=1 по умолчанию).
os.environ.setdefault("API_KEY_REQUIRED", "0")

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.WARNING)
INDEX_DIR = Path(
    "outputs/astrology_test/gemma-4-e4b_sentence-transformers_paraphrase-multilingual-mpnet-base-v2"
)
results = []


def check(num, name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] Критерий {num}: {name} — {detail}")


# --- Критерий 1: facts > 100 (parquet, без сервера) ---
fact_path = INDEX_DIR / "fact_embeddings" / "vdb_fact.parquet"
n_facts = len(pd.read_parquet(fact_path)) if fact_path.exists() else 0
check(1, "facts > 100", n_facts > 100, f"facts = {n_facts}")

# --- Критерий 2: synonymy-рёбра > 0 (graph, без сервера) ---
graph_path = INDEX_DIR / "graph.pickle"
n_syn = -1
n_total_edges = 0
if graph_path.exists():
    import igraph as ig

    g = ig.Graph.Read_Pickle(str(graph_path))
    n_syn = sum(
        1
        for e in g.es
        if not g.vs[e.source]["name"].startswith("chunk-")
        and not g.vs[e.target]["name"].startswith("chunk-")
    )
    n_total_edges = g.ecount()
check(2, "synonymy-рёбра > 0", n_syn > 0, f"entity↔entity рёбер: {n_syn} из {n_total_edges}")

# --- Счётчики для критерия 4 (parquet, без сервера) ---
n_chunks = 0
n_entities = 0
chunk_parquet = INDEX_DIR / "chunk_embeddings" / "vdb_chunk.parquet"
entity_parquet = INDEX_DIR / "entity_embeddings" / "vdb_entity.parquet"
if chunk_parquet.exists():
    n_chunks = len(pd.read_parquet(chunk_parquet))
if entity_parquet.exists():
    n_entities = len(pd.read_parquet(entity_parquet))

# --- Критерии 3 и 5: /health, предупреждение lifespan и /search — ОДИН запуск приложения ---
health_chunks = health_facts = -1
warn_found = None
crit5_ok = False
crit5_detail = "—"
try:
    from fastapi.testclient import TestClient

    from hipporag.search_api import server as srv
    from hipporag.search_api.server import app

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    srv.logger.addHandler(handler)
    old_level = srv.logger.level
    srv.logger.setLevel(logging.WARNING)
    try:
        with TestClient(app) as client:  # единственный запуск lifespan
            body = client.get("/health").json()
            health_chunks, health_facts = body.get("chunks", -1), body.get("facts", -1)
            warn_found = "Факт-стор почти пуст" in buf.getvalue()

            r5 = client.post(
                "/search",
                json={"question": "Как обустроить прихожую по фэншуй для Овна?", "top_k": 5},
            )
            ok_http = r5.status_code == 200
            data = r5.json().get("data", {}) if ok_http else {}
            cites = data.get("citations", [])
            has_docs = len(data.get("retrieved_docs", [])) > 0
            sources_ok = bool(cites) and all(
                isinstance(c.get("sources"), list) and len(c["sources"]) > 0 for c in cites
            )
            crit5_ok = ok_http and has_docs and sources_ok
            crit5_detail = (
                f"HTTP {r5.status_code}, цитат: {len(cites)}, "
                f"docs: {len(data.get('retrieved_docs', []))}, "
                f"sources пример: {cites[0]['sources'][:2] if cites else '—'}"
            )
    finally:
        srv.logger.removeHandler(handler)
        srv.logger.setLevel(old_level)
except Exception as e:  # noqa: BLE001
    crit5_detail = f"ошибка: {e}"

crit3_ok = health_facts > 100 and warn_found is False
check(
    3,
    "/health facts и без предупреждения",
    crit3_ok,
    f"/health: chunks={health_chunks}, facts={health_facts}; предупреждение: {warn_found}",
)

if all(v > 0 for v in (n_chunks, n_entities, n_facts)):
    ratio = max(n_entities, n_facts, n_chunks) / min(n_entities, n_facts, n_chunks)
    check(
        4,
        "масштабы сопоставимы (перекос < 10x)",
        ratio < 10,
        f"entities={n_entities}, facts={n_facts}, chunks={n_chunks} (перекос x{ratio:.1f})",
    )
else:
    check(4, "масштабы сопоставимы", False, f"entities={n_entities}, facts={n_facts}, chunks={n_chunks}")

check(5, "/search на русском + citations[].sources", crit5_ok, crit5_detail)

print()
print(f"ИТОГ: {sum(results)}/5 критериев пройдено")
sys.exit(0 if all(results) else 1)
