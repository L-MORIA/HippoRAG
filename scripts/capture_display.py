"""Снапшот вывода show_results для всех 23 режимов (для регрессии дисплея).

Использует реальные результаты из outputs/refactor_baseline/<basis>/snapshot.json.
Режим, на котором show_results падает, фиксируется как {"error": "..."} — это
часть базлайна.

Запуск: python scripts/capture_display.py --label pre --basis pre
"""

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from astro_app.display import show_results  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="Имя файла: display_<label>.json")
parser.add_argument("--basis", default="pre", help="Каталог базлайна с snapshot.json")
parser.add_argument("--query", default="как обустроить прихожую для Овна")
args = parser.parse_args()

basis = json.loads(
    (REPO_ROOT / "outputs" / "refactor_baseline" / args.basis / "snapshot.json").read_text(
        encoding="utf-8"
    )
)

out = {}
for mid in [str(i) for i in range(1, 24)]:
    entry = basis[mid]
    results = entry.get("results") or []
    extra = entry.get("extra")
    extra = extra if isinstance(extra, str) else None
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            show_results(args.query, results, entry["name"], extra)
        out[mid] = {"ok": True, "stdout": buf.getvalue()}
    except Exception as e:  # noqa: BLE001
        out[mid] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    status = "ok" if out[mid]["ok"] else f"ПАДЕНИЕ: {out[mid]['error'][:60]}"
    print(f"режим {mid} ({entry['name']}): {status}")

dest = REPO_ROOT / "outputs" / "refactor_baseline" / f"display_{args.label}.json"
dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nГотово: {dest}")
