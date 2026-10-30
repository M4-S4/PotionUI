import ast
from pathlib import Path
from typing import Iterator, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
LLM_TYPE_NAMES = frozenset({
    "ollama", "openai", "openai-compatible", "native", "anthropic", "gemini", "openrouter", "deepseek",
})
EXEMPT = (
    SOURCE / "platform" / "database" / "migrations",
    SOURCE / "features" / "llm" / "harness" / "adapters",
)


def _type_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        return len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == "type"
    else:
        return False
    return name == "type" or name.endswith("_type") or name in ("type_id", "type_name")


def _names_llm_type(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in LLM_TYPE_NAMES
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_names_llm_type(item) for item in node.elts)
    return False


def _branches(tree: ast.AST) -> Iterator[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if any(_type_like(operand) for operand in operands) and any(_names_llm_type(operand) for operand in operands):
            yield node.lineno


def _violations(root: Path) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for path in sorted(root.rglob("*.py")):
        if any(path.is_relative_to(exempt) for exempt in EXEMPT):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found.extend((str(path.relative_to(root)), line) for line in _branches(tree))
    return found


def test_core_never_branches_on_an_llm_provider_type():
    assert _violations(SOURCE) == []


def test_the_check_catches_a_provider_type_branch(tmp_path):
    (tmp_path / "branchy.py").write_text(
        "def f(config):\n"
        "    if config.type == 'ollama':\n"
        "        return 1\n"
        "    if getattr(config, 'type', None) in ('openai', 'openrouter'):\n"
        "        return 2\n"
        "    if field_type == 'checkbox':\n"
        "        return 3\n",
        encoding="utf-8",
    )

    assert [line for _, line in _violations(tmp_path)] == [2, 4]
