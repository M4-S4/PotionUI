import ast
from pathlib import Path

ALLOWED = {"re", "dataclasses", "decimal", "typing"}
MODULE = Path(__file__).resolve().parents[2] / "src" / "features" / "prompt" / "conditions.py"
BANNED_CALLS = {"eval", "exec", "compile", "getattr", "setattr", "__import__", "open"}


def _tree():
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_conditions_imports_only_the_allow_listed_stdlib_modules():
    imported = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add((node.module or "").split(".")[0])
    assert imported <= ALLOWED


def test_conditions_never_calls_dynamic_execution_builtins():
    called = {
        node.func.id
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & BANNED_CALLS
