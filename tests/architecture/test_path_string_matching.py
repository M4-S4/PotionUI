from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator, List

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("tests", "content/plugins")
SKIP_DIRS = {"__pycache__", "node_modules", "venv", ".git", ".svelte-kit", "dist"}
STRIP_METHODS = {"rstrip", "lstrip", "strip"}
MATCH_METHODS = {"endswith", "startswith", "find", "rfind", "index", "rindex"}


PATH_STR_RE = re.compile(r"str\(\s*[\w.]*path\s*\)", re.IGNORECASE)


def _may_match_path_str(text: str) -> bool:
    return PATH_STR_RE.search(text) is not None


def _may_skip_on_missing_preset(text: str) -> bool:
    return "not present" in text and "skip" in text


def _test_files(source_index) -> Iterator[Path]:
    for scope in SCOPES:
        base = ROOT / scope
        if not base.is_dir():
            continue
        for path in source_index.py_files(base):
            parts = path.relative_to(ROOT).parts
            if SKIP_DIRS.intersection(parts[:-1]):
                continue
            if "tests" in parts[:-1] or path.name.startswith("test_"):
                yield path


def _is_path_str(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
    ):
        return False
    arg = node.args[0]
    if isinstance(arg, ast.Name):
        name = arg.id
    elif isinstance(arg, ast.Attribute):
        name = arg.attr
    else:
        return False
    return name.lower().endswith("path")


def _has_separator(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and ("/" in node.value or "\\" in node.value)
    if isinstance(node, ast.Tuple):
        return any(_has_separator(item) for item in node.elts)
    return True


def _violations(tree: ast.AST) -> List[int]:
    lines: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and _is_path_str(right):
                    lines.append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if not _is_path_str(node.func.value):
                continue
            if node.func.attr in STRIP_METHODS:
                lines.append(node.lineno)
            elif node.func.attr in MATCH_METHODS and node.args and _has_separator(node.args[0]):
                lines.append(node.lineno)
    return lines


def test_tests_never_match_str_of_a_path_against_a_separator_literal(source_index):
    offenders: List[str] = []
    for path in _test_files(source_index):
        if not _may_match_path_str(source_index.text(path)):
            continue
        tree = source_index.tree(path)
        if tree is None:
            continue
        for lineno in _violations(tree):
            offenders.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}")
    assert not offenders, (
        "str(<path>) carries backslashes on Windows, so slash-based substring/endswith matching never "
        "matches there; compare Path(...).as_posix() or Path.parts instead:\n  " + "\n  ".join(sorted(offenders))
    )


def _skips_on_missing_preset(tree: ast.AST) -> List[int]:
    lines: List[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "skip"
            and node.args
        ):
            text = ast.unparse(node.args[0])
            if "preset" in text and text.rstrip("'\")").endswith("not present"):
                lines.append(node.lineno)
    return lines


def test_a_tracked_preset_missing_from_a_fixture_fails_instead_of_skipping(source_index):
    offenders: List[str] = []
    for path in _test_files(source_index):
        if not _may_skip_on_missing_preset(source_index.text(path)):
            continue
        tree = source_index.tree(path)
        if tree is None:
            continue
        for lineno in _skips_on_missing_preset(tree):
            offenders.append(f"{path.relative_to(ROOT).as_posix()}:{lineno}")
    assert not offenders, "fixtures for tracked presets must pytest.fail when the lookup misses:\n  " + "\n  ".join(
        sorted(offenders)
    )


def test_the_guard_flags_the_old_patterns_and_accepts_the_new_ones():
    flagged = [
        'next(p for p in presets if "marketplace/Flux3" in str(p.path))',
        'next(p for p in presets if str(p.path).endswith("native/LTX-2"))',
        'next(p for p in presets if str(p.path).rstrip("/").endswith("/Wan"))',
        "str(preset.path).startswith(core_root)",
    ]
    accepted = [
        'next(p for p in presets if "marketplace/Flux3" in Path(p.path).as_posix())',
        'str(p.path).replace("\\\\", "/").endswith("marketplace/SDXL")',
        'str(path).endswith(".part")',
        "str(tmp_path) in env",
    ]
    for source in flagged:
        assert _violations(ast.parse(source)), source
    for source in accepted:
        assert not _violations(ast.parse(source)), source
