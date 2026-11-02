# Contributing to HippoRAG

Thank you for your interest in contributing to HippoRAG!

> **Current project state:** 23 search methods, 4 LLM models, QA Cache.  
> See **[HOW_TO_RUN.md](./HOW_TO_RUN.md)** for the complete up-to-date documentation.

## How to Contribute

1. Fork the repository and clone it to your local machine.
2. Create a new branch: `git checkout -b my-contribution`.
3. Make your changes and ensure tests pass.
4. Commit your changes: `git commit -m "Add my contribution"`.
5. Push to your forked repository: `git push origin my-contribution`.
6. Open a pull request to the main repository.

## Before You Start, File an Issue

Please file an issue **BEFORE** starting work to fix/implement something.  
Search existing open and closed issues first — someone else may already be working on it.

## Development Notes

- All methods run on **CPU** (no CUDA required)
- Embedding model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Russian cross-encoder: `DiTy/cross-encoder-russian-msmarco`
- LLM is selected at startup from 4 available models
- QA Cache auto-saves answers to `outputs/qa_cache/qa_cache.json`
- Run `python main.py` to start the interactive menu with all 23 methods

---

*This contributing guide was partially adapted from the GraphRAG project.*
