"""Снапшот выдачи всех 23 режимов main.py — базлайн для рефакторинга (шаг 0 плана
MAIN_SPLIT_PLAN.md §5.1).

Воспроизводит диспетчер меню main.py:3464-3488 ТОЧНО: каждый режим вызывается как
fn(hipporag, query); режимы 1 и 7 возвращают кортеж (results, extra). Автосохранение
в QA-кэш (main.py:3499-3516) НЕ воспроизводится — это пост-обработка, не влияющая
на выдачу, и снапшот не должен писать в пользовательские данные.

Запуск:
    python scripts/snapshot_modes.py --label pre --query "как обустроить прихожую для Овна"
Результат:
    outputs/refactor_baseline/<label>/snapshot.json
    outputs/refactor_baseline/<label>/manifest.json
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

parser = argparse.ArgumentParser(description="Снапшот выдачи всех режимов main.py")
parser.add_argument("--label", default="pre", help="Имя каталога: pre / post_step7 / ...")
parser.add_argument("--modes", default="1-23", help="Диапазон режимов, напр. 1-23 или 3,5,7")
parser.add_argument(
    "--query", default="как обустроить прихожую для Овна", help="Контрольный запрос"
)
args = parser.parse_args()


def parse_modes(spec: str) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part.strip())
    return out


MODE_IDS = parse_modes(args.modes)
QUERY = args.query
LABEL = args.label

from astro_app import config as app_config  # noqa: E402
from astro_app.modes import bm25_hybrid, dense, filtered, graph, hyde, llm_filter, qacache, sentence_dat, self_rag, sign_zones, small_to_big  # noqa: E402

from hipporag import HippoRAG  # noqa: E402
from hipporag.utils.config_utils import BaseConfig  # noqa: E402


def to_jsonable(obj, _depth=0):
    """Полная (не урезанная) сериализация выдачи режима для побитового сравнения."""
    if _depth > 6:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist(), _depth + 1)
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x, _depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in obj.items()}
    return str(obj)


# Реплика main.py:3464-3488 (порядок и состав — как в меню; функции берутся из модуля main)
MODE_MAP = [
    ("1", "Multi-vector", dense.search_multi_vector, True),
    ("2", "Граф", graph.search_hipporag_graph, False),
    ("3", "BM25+CE-en", bm25_hybrid.search_bm25_hybrid, False),
    ("4", "Small→Big-en", small_to_big.search_small_to_big, False),
    ("5", "BM25+CE-ru", bm25_hybrid.search_bm25_hybrid_ru, False),
    ("6", "Small→Big-ru", small_to_big.search_small_to_big_ru, False),
    ("7", "HyDE", hyde.search_hyde, True),
    ("8", "Фильтр", filtered.search_filtered, False),
    ("9", "Весь знак", sign_zones.extract_sign_full, False),
    ("10", "Резюме", sign_zones.extract_sign_summary, False),
    ("11", "Обраб.", sign_zones.extract_sign_polished, False),
    ("12", "Self-RAG", self_rag.search_self_rag, False),
    ("13", "RRF+Контекст", bm25_hybrid.search_rrf_context, False),
    ("14", "E5-rerank", dense.search_e5_rerank, False),
    ("15", "RRF+sign", dense.search_mxbai, False),
    ("16", "ColBERT", dense.search_colbert, False),
    ("17", "LLM-фильтр", llm_filter.search_with_llm_filter, False),
    ("18", "Qwen+FRIDA", llm_filter.search_qwen_frida, False),
    ("19", "GraphRAG", graph.search_graphrag_agent, False),
    ("20", "Sentence-level", sentence_dat.search_sentence_level, False),
    ("21", "DAT+LLM", sentence_dat.search_dat_sentence_llm, False),
    ("22", "SemFilter", sign_zones.search_semantic_filter, False),
    ("23", "QACache", qacache.search_qa_cache, False),
]
assert len(MODE_MAP) == 23

print(f"Снапшот: label={LABEL}, запрос={QUERY!r}, режимы: {MODE_IDS}")
config = BaseConfig(
    save_dir=app_config.SAVE_DIR,
    llm_name=app_config.LLM_NAME,
    llm_base_url=app_config.LLM_BASE_URL,
    embedding_model_name=app_config.EMBEDDING_MODEL,
    linking_top_k=100,
    retrieval_top_k=200,
    passage_node_weight=0.05,
    embedding_batch_size=8,
)
print("Загрузка индекса...")
hipporag = HippoRAG(global_config=config)
print("Индекс загружен.\n")

snapshot = {}
manifest_entries = []

for mode_id, mode_name, fn, has_extra in MODE_MAP:
    if mode_id not in MODE_IDS:
        continue
    print(f"--- режим {mode_id} ({mode_name}) ---", flush=True)
    t0 = time.time()
    entry = {"name": mode_name, "query": QUERY}
    try:
        result = fn(hipporag, QUERY)
        if has_extra:
            results, extra = result
            entry["extra"] = to_jsonable(extra)
        else:
            results = result
        entry["results"] = to_jsonable(results)
        entry["ok"] = True
        entry["error"] = None
    except Exception as e:  # noqa: BLE001
        import traceback

        entry["ok"] = False
        entry["results"] = None
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["traceback"] = traceback.format_exc()
    entry["seconds"] = round(time.time() - t0, 1)
    n = len(entry["results"]) if isinstance(entry.get("results"), list) else None
    print(f"    ok={entry['ok']} seconds={entry['seconds']} items={n}", flush=True)
    snapshot[mode_id] = entry
    manifest_entries.append({"mode": mode_id, "ok": entry["ok"], "seconds": entry["seconds"]})

out_dir = REPO_ROOT / "outputs" / "refactor_baseline" / LABEL
out_dir.mkdir(parents=True, exist_ok=True)
snap_path = out_dir / "snapshot.json"
snap_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")

main_src = (REPO_ROOT / "main.py").read_bytes()
manifest = {
    "label": LABEL,
    "query": QUERY,
    "modes": MODE_IDS,
    "git_head": __import__("subprocess")
    .run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    .stdout.strip(),
    "main_py_sha256": "astro_app",
    "entries": manifest_entries,
}
(out_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
)

ok_n = sum(1 for e in manifest_entries if e["ok"])
print(f"\nГотово: {snap_path}")
print(f"Режимов ok: {ok_n}/{len(manifest_entries)}")
sys.exit(0 if ok_n == len(manifest_entries) else 1)
