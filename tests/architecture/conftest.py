from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

PRUNED_DIRS = frozenset({"__pycache__", "node_modules", "venv", ".git"})


class SourceIndex:
    def __init__(self) -> None:
        self._listings: dict[Path, list[Path]] = {}
        self._texts: dict[Path, str] = {}
        self._trees: dict[Path, ast.Module | None] = {}

    def py_files(self, base: Path) -> list[Path]:
        if base not in self._listings:
            found: list[Path] = []
            for directory, subdirs, names in os.walk(base):
                subdirs[:] = [d for d in subdirs if d not in PRUNED_DIRS]
                found.extend(Path(directory) / n for n in names if n.endswith(".py"))
            self._listings[base] = sorted(found)
        return self._listings[base]

    def text(self, path: Path) -> str:
        if path not in self._texts:
            self._texts[path] = path.read_text(encoding="utf-8")
        return self._texts[path]

    def tree(self, path: Path) -> ast.Module | None:
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse(self.text(path))
            except SyntaxError:
                self._trees[path] = None
        return self._trees[path]


@pytest.fixture(scope="session")
def source_index() -> SourceIndex:
    return SourceIndex()
