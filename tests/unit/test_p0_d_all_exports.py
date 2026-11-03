"""P0-D / H0-3: единый ``__all__`` и рабочий ``from main import *``."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = PROJECT_ROOT / "main.py"


def test_all_is_assigned_once():
    """В main.py не должно быть второго присваивания __all__."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1


def test_star_import_exposes_key_api():
    """`from main import *` отдаёт функции, на которые опираются тесты и скрипты."""
    namespace = {}
    exec("from main import *", namespace)  # noqa: S102 - это и есть предмет проверки

    for name in (
        "main",
        "search_bm25_hybrid_ru",
        "search_hipporag_graph",
        "_get_chunk_meta",
    ):
        assert name in namespace
