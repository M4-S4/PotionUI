import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

ALLOWED_UNCONDITIONAL_SKIPS = {
    "tests/platform/runtime/native/lora/test_fp8_bake_fidelity.py::test_a_one_percent_delta_mostly_rounds_off_when_baked_into_fp8":
        "recorded decision: documents why the fp8 bake was rejected, not a gate",
}

UNCONDITIONAL_SKIP_MARKS = {"unittest.skip", "pytest.mark.skip"}


ASSERT_TRUE = re.compile(r"assert\s+True\b")
TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)", re.M)
SUSPECT = re.compile(r"(?:unittest|pytest\.mark)\.skip\b|pytestmark|^\s*(?:pass|\.\.\.)\s*$|assert\s+True\b", re.M)


def _test_roots():
    yield ROOT / "tests"
    for tier in ("marketplace", "local"):
        base = ROOT / "content" / "plugins" / tier
        if base.is_dir():
            for plugin in sorted(base.iterdir()):
                if (plugin / "tests").is_dir():
                    yield plugin / "tests"


def _test_files(source_index):
    for root in _test_roots():
        for path in source_index.py_files(root):
            if path.name.startswith("test_") or path.name.endswith("_test.py"):
                yield path


def _may_violate(text):
    if SUSPECT.search(text):
        return True
    names = TEST_DEF.findall(text)
    return len(names) != len(set(names))


def _mark_name(node):
    target = node.func if isinstance(node, ast.Call) else node
    return ast.unparse(target)


def _is_test(node):
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")


def _is_inert_statement(stmt):
    if isinstance(stmt, ast.Pass):
        return True
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)


def _scopes(tree):
    yield "", tree.body
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node.name + ".", node.body


def hygiene_violations(source, label):
    tree = ast.parse(source)
    found = []
    has_assert_true = ASSERT_TRUE.search(source) is not None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            marks = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            if any(_mark_name(m) in UNCONDITIONAL_SKIP_MARKS for m in marks):
                found.append(f"{label}:{node.lineno} unconditional module skip")
    for prefix, body in _scopes(tree):
        seen = {}
        for node in body:
            if isinstance(node, ast.ClassDef) and any(
                _mark_name(d) in UNCONDITIONAL_SKIP_MARKS for d in node.decorator_list
            ):
                found.append(f"{label}:{node.lineno} unconditional skip on class {node.name}")
            if not _is_test(node):
                continue
            where = f"{label}:{node.lineno} {prefix}{node.name}"
            if node.name in seen:
                found.append(f"{where} duplicate definition (first at line {seen[node.name]})")
            seen[node.name] = node.lineno
            if any(_mark_name(d) in UNCONDITIONAL_SKIP_MARKS for d in node.decorator_list):
                if f"{label}::{node.name}" not in ALLOWED_UNCONDITIONAL_SKIPS:
                    found.append(f"{where} unconditional skip")
            if all(_is_inert_statement(s) for s in node.body):
                found.append(f"{where} body does nothing")
            for inner in ast.walk(node) if has_assert_true else ():
                if (
                    isinstance(inner, ast.Assert)
                    and isinstance(inner.test, ast.Constant)
                    and inner.test.value is True
                ):
                    found.append(f"{label}:{inner.lineno} assert True in {node.name}")
    return found


def test_tests_have_no_placeholder_skips_duplicates_or_empty_bodies(source_index):
    violations = []
    for path in _test_files(source_index):
        text = source_index.text(path)
        if not _may_violate(text):
            continue
        violations.extend(hygiene_violations(text, path.relative_to(ROOT).as_posix()))
    assert not violations, "test hygiene violations:\n" + "\n".join(violations)


def test_skip_allowlist_entries_still_exist():
    for entry in ALLOWED_UNCONDITIONAL_SKIPS:
        rel, name = entry.split("::")
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert f"def {name}(" in source


@pytest.mark.parametrize(
    "source",
    [
        "import unittest\nclass T(unittest.TestCase):\n    @unittest.skip('x')\n    def test_a(self):\n        assert 1\n",
        "import pytest\n@pytest.mark.skip(reason='x')\ndef test_a():\n    assert 1\n",
        "import pytest\n@pytest.mark.skip\nclass TestA:\n    def test_a(self):\n        assert 1\n",
        "import pytest\npytestmark = pytest.mark.skip(reason='x')\n",
        "def test_a():\n    assert 1\n\ndef test_a():\n    assert 2\n",
        "class T:\n    def test_a(self):\n        assert 1\n    def test_a(self):\n        assert 2\n",
        "def test_a():\n    pass\n",
        "def test_a():\n    '''doc'''\n    ...\n",
        "async def test_a():\n    assert True\n",
        "def test_a():\n    x = 1\n    assert True\n",
    ],
)
def test_hygiene_rules_flag_each_violation(source):
    assert hygiene_violations(source, "sample.py")


@pytest.mark.parametrize(
    "source",
    [
        "import pytest\n@pytest.mark.skipif(True, reason='x')\ndef test_a():\n    assert 1\n",
        "import pytest\ndef test_a():\n    if True:\n        pytest.skip('x')\n    assert 1\n",
        "class A:\n    def test_a(self):\n        assert 1\nclass B:\n    def test_a(self):\n        assert 1\n",
        "def test_a():\n    assert 1\n\ndef helper():\n    pass\n",
    ],
)
def test_hygiene_rules_accept_legitimate_tests(source):
    assert hygiene_violations(source, "sample.py") == []
