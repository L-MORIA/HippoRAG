# HippoRAG Quick Start (EN)

Five steps to the first answer. Full product documentation is in Russian:
[USER_GUIDE.md](./USER_GUIDE.md) · [HOW_TO_RUN.md](./HOW_TO_RUN.md) ·
[TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

## 1. Install

Requirements: Python 3.10–3.12, 8+ GB RAM.

```bash
git clone https://github.com/L-MORIA/HippoRAG.git
cd HippoRAG
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## 2. LM Studio

Start LM Studio → **Developer → Start Server** (port `1234`) and check:

```bash
curl http://localhost:1234/v1/models
```

Copy `.env.example` to `.env` and set:

```dotenv
LLM_NAME=gemma-4-e4b
LLM_API_NAME=google/gemma-4-e4b
LLM_BASE_URL=http://localhost:1234/v1
SAVE_DIR=outputs/astrology_test
```

## 3. Build the index

Put your PDFs into `data/pdfs/` and run:

```bash
python index_pdfs.py --pdf_dir data/pdfs --output_dir outputs/astrology_test
python scripts/show_index_stats.py --save_dir outputs/astrology_test
```

`--output_dir` must match `SAVE_DIR`. Without an index, search returns an honest
abstention.

## 4. Run

Web UI:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:API_KEY = "your-secret-key"
uvicorn astro_app.web:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` and enter the key. For local-only use without a
key: `$env:API_KEY_REQUIRED = "0"`.

CLI:

```bash
python main.py
```

The default mode is **5 “BM25+CE-ru”**; modes **2 “Graph”** and
**22 “Pattern Search”** are also recommended.

## 5. Ask

Ask a question about the corpus, for example: “Which zones correspond to the
Aries sign?”. The answer contains `[n]` citations and a list of used sources.
If the sources do not contain the answer, the system abstains.

Troubleshooting: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).
