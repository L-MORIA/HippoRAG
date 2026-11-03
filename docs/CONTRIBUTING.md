# Contributing to HippoRAG

Thank you for your interest in contributing to HippoRAG!

> **Current project state:** 23 search methods, LLM configured via `.env`, SQLite QA/LLM cache.  
> Full documentation: **[QUICKSTART_RU.md](./QUICKSTART_RU.md)** (Russian) · **[QUICKSTART.md](./QUICKSTART.md)** (English) · **[USER_GUIDE.md](./USER_GUIDE.md)**

## How to Contribute

1. Fork the repository and clone it to your local machine.
2. **Install the pre-commit hooks** (one-time per clone — the hook is local and is not stored in git):
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```
   Without this step the checks in `.pre-commit-config.yaml` never run — `git commit` will
   silently accept unformatted code. Verify with `pre-commit run --all-files` (expect all
   hooks to pass).
3. Create a new branch: `git checkout -b my-contribution`.
4. Make your changes and ensure tests pass.
5. Commit your changes: `git commit -m "Add my contribution"`.
   The hooks may reformat files and abort the first commit — re-stage and commit again.
6. Push to your forked repository: `git push origin my-contribution`.
7. Open a pull request to the main repository.

### Checks run by the hooks

| Hook | Scope | Purpose |
|------|-------|---------|
| `black`, `isort` | `src/`, `tests/`, `main.py` | formatting (line length 100) |
| `ruff` | `src/`, `tests/`, `main.py` | real errors: syntax, pyflakes, bugbear |
| `bandit` | `src/`, `tests/` | security issues |
| `trailing-whitespace`, `end-of-file-fixer`, `mixed-line-ending` | `src/`, `tests/`, `main.py` | text hygiene |
| `check-yaml`, `check-json`, `check-merge-conflict`, `debug-statements` | — | structure and leftovers |

The hook scope matches the CI scope exactly (`src/`, `tests/`, `main.py`) — CI runs the same tools on
the whole tree after push. To skip a hook once: `SKIP=ruff git commit -m "..."`.

## Before You Start, File an Issue

Please file an issue **BEFORE** starting work to fix/implement something.  
Search existing open and closed issues first — someone else may already be working on it.

## Development Notes

- All methods run on **CPU** (no CUDA required)
- Embedding model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Russian cross-encoder: `DiTy/cross-encoder-russian-msmarco`
- The LLM is picked from `.env` (`LLM_NAME`, `LLM_BASE_URL`) — by default LM Studio with `gemma-4-e4b`
- LLM responses are cached in SQLite: `<save_dir>/llm_cache/<model>_cache.sqlite`
- Run `python main.py` to start the interactive menu with all 23 methods

---

*This contributing guide was partially adapted from the GraphRAG project.*
