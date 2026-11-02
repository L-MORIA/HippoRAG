"""Шаг 6: извлечение групп режимов из main.py в src/astro_app/modes/.

Тела функций вырезаются байт-в-байт по AST-границам (не по угаданным строкам),
модульные константы — по именам top-level Assign. После сборки каждый модуль
проверяется ruff (F821 — undefined name) и компиляцией.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
src = (ROOT / "main.py").read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)

tree = ast.parse(src)


def span(node):
    """[start, end) строки узла, обрезая хвостовые пустые."""
    end = node.end_lineno
    while end > node.lineno and lines[end - 1].strip() == "":
        end -= 1
    return node.lineno, end


# Топ-узлы по имени (функции и присваивания констант)
funcs, assigns = {}, {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        funcs[node.name] = node
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                assigns[t.id] = node

GROUPS = {
    "dense": {
        "funcs": ["search_multi_vector", "search_mxbai", "search_e5_rerank"],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import numpy as np\n"
            "import requests\n"
            "\n"
            "from ..config import LLM_BASE_URL, LLM_NAME, EMBEDDING_MODEL, STOP_WORDS\n"
            "from ..primitives import _build_bm25, _embedding_scores, _get_chunk_meta\n"
            "from ..query import _expand_query, reformulate_query\n"
            "from ..rerankers import _get_e5, _rerank_with_crossencoder\n"
            "from ..textutils import _tokenize, fix_chunk_start, show_text_safe\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "bm25_hybrid": {
        "funcs": ["search_bm25_hybrid", "search_bm25_hybrid_ru", "search_rrf_context"],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import numpy as np\n"
            "import requests\n"
            "\n"
            "from ..config import LLM_BASE_URL, STOP_WORDS\n"
            "from ..primitives import _build_bm25, _embedding_scores, _get_chunk_meta\n"
            "from ..query import _expand_query, _word_re\n"
            "from ..rerankers import _rerank_with_crossencoder_ru\n"
            "from ..textutils import _tokenize, fix_chunk_start, show_text_safe\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "graph": {
        "funcs": ["search_hipporag_graph", "search_graphrag_agent"],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import httpx\n"
            "\n"
            "from ..config import LLM_BASE_URL, LLM_NAME\n"
            "from ..query import _expand_query, _word_re, ASTRO_TERMS\n"
            "from ..textutils import fix_chunk_start\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "small_to_big": {
        "funcs": [
            "_build_sign_index",
            "_sign_in_query",
            "_count_sign",
            "_keyword_boost",
            "_chunk_sign_score",
            "search_small_to_big",
            "search_small_to_big_ru",
        ],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import numpy as np\n"
            "\n"
            "from ..config import STOP_WORDS\n"
            "from ..primitives import _build_bm25, _embedding_scores, _get_chunk_meta\n"
            "from ..query import ASTRO_TERMS, FORM_TO_SIGN, _ALL_SIGN_FORMS_RE, _expand_query, _word_re\n"
            "from ..rerankers import _rerank_with_crossencoder_ru\n"
            "from ..textutils import _tokenize, fix_chunk_start\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "llm_filter": {
        "funcs": [
            "_filter_by_section",
            "_llm_filter_results",
            "search_with_llm_filter",
            "_get_qwen",
            "_get_frida",
            "search_qwen_frida",
        ],
        "consts": [],
        "imports": (
            "import re\n"
            "\n"
            "import httpx\n"
            "import requests\n"
            "\n"
            "from ..config import LLM_BASE_URL\n"
            "from ..query import _expand_query\n"
            "from ..textutils import fix_chunk_start\n"
        ),
    },
    "sentence_dat": {
        "funcs": [
            "_is_bibliographic_query",
            "search_sentence_level",
            "_dat_evaluate",
            "_dat_compute_alpha",
            "_dat_combine_results",
            "search_dat_sentence_llm",
        ],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import numpy as np\n"
            "import httpx\n"
            "\n"
            "from ..config import LLM_BASE_URL\n"
            "from ..query import _word_re\n"
            "from ..textutils import _tokenize\n"
            "from ..primitives import _embedding_scores\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "self_rag": {
        "funcs": ["search_self_rag"],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import numpy as np\n"
            "import httpx\n"
            "\n"
            "from ..config import LLM_BASE_URL, LLM_NAME\n"
            "from ..primitives import _embedding_scores, _get_chunk_meta\n"
            "from ..query import _expand_query\n"
            "from ..textutils import _tokenize\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "hyde": {
        "funcs": ["search_hyde"],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import httpx\n"
            "\n"
            "from ..config import LLM_BASE_URL\n"
            "from ..query import _expand_query\n"
            "from ..rerankers import _rerank_with_crossencoder_ru\n"
            "from ..textutils import fix_chunk_start\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
    "sign_zones": {
        "funcs": [
            "extract_sign_full",
            "extract_sign_polished",
            "extract_sign_summary",
            "_sentence_index",
            "_zones_in_text",
            "_sign_zone_class",
            "_find_sign_zones",
            "_find_zone_signs",
            "_sign_zone_answer",
            "search_semantic_filter",
        ],
        "consts": [
            "SIGN_PLANET",
            "ZONE_ALIASES",
            "_ZONE_TEXT_STEMS",
            "_SIGN_ZONE_STRONG",
            "_SIGN_ZONE_WEAK",
            "_CLASS_NAMES",
            "_ACCUSATIVE",
            "_SENTENCE_INDEX",
            "_SENTENCE_INDEX_KEY",
        ],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import httpx\n"
            "\n"
            "from ..config import LLM_BASE_URL, LLM_NAME, STOP_WORDS\n"
            "from ..query import FORM_TO_SIGN, _ALL_SIGN_FORMS_RE, _expand_query, _word_re\n"
            "from ..textutils import fix_chunk_start, show_text_safe\n"
            "from ..qa_cache import _qa_load, _qa_lookup, _qa_derive_short, _qa_store, QA_CACHE_VERSION\n"
        ),
    },
    "filtered": {
        "funcs": ["search_filtered"],
        "consts": [],
        "imports": (
            "import json\n"
            "import re\n"
            "\n"
            "import numpy as np\n"
            "import requests\n"
            "\n"
            "from ..config import STOP_WORDS\n"
            "from ..primitives import _build_bm25, _embedding_scores, _get_chunk_meta\n"
            "from ..query import _expand_query\n"
            "from ..rerankers import _rerank_with_crossencoder_ru\n"
            "from ..textutils import _tokenize, fix_chunk_start, show_text_safe\n"
            "from ..qa_cache import _qa_lookup, _qa_store, _qa_derive_short\n"
            "from hipporag.utils.misc_utils import min_max_normalize\n"
        ),
    },
}

OUT = ROOT / "src" / "astro_app" / "modes"
OUT.mkdir(parents=True, exist_ok=True)

missing_all = []
for module, spec in GROUPS.items():
    parts = [
        f'"""Режимы {module} — перенос из main.py (шаг 6, байт-в-байт)."""\n\n',
        spec["imports"],
        "\n",
    ]
    for const in spec["consts"]:
        node = assigns.get(const)
        if node is None:
            missing_all.append(f"{module}: константа {const} не найдена")
            continue
        a, b = span(node)
        parts.append("".join(lines[a - 1 : b]) + "\n\n")
    for fn in spec["funcs"]:
        node = funcs.get(fn)
        if node is None:
            missing_all.append(f"{module}: функция {fn} не найдена")
            continue
        a, b = span(node)
        parts.append("".join(lines[a - 1 : b]) + "\n\n")
    body = "".join(parts)
    (OUT / f"{module}.py").write_text(body, encoding="utf-8", newline="\n")
    print(f"записан modes/{module}.py: {len(body.splitlines())} строк")

print("ПРОПУЩЕНО:", missing_all or "нет")
