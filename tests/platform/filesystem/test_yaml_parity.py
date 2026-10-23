from pathlib import Path

import pytest
import yaml

from src.platform.filesystem import yaml_io

ROOT = Path(__file__).resolve().parents[3]


def _yaml_files():
    files = []
    for base in (ROOT / "content", ROOT / "src" / "std"):
        for pattern in ("*.yml", "*.yaml"):
            files.extend(base.rglob(pattern))
    return sorted(f for f in files if "node_modules" not in f.parts)


def _outcome(loader, text):
    try:
        return "ok", yaml.load(text, Loader=loader)
    except yaml.YAMLError as exc:
        return "error", str(exc)


def test_shipped_yaml_parses_identically_with_fast_and_pure_loaders():
    files = _yaml_files()
    assert files
    mismatches = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        pure = _outcome(yaml.SafeLoader, text)
        try:
            fast = ("ok", yaml_io.load_yaml(text))
        except yaml.YAMLError as exc:
            fast = ("error", str(exc))
        if fast != pure:
            mismatches.append(path.relative_to(ROOT).as_posix())
    assert mismatches == []


@pytest.mark.skipif(not hasattr(yaml, "CSafeLoader"), reason="libyaml not available")
def test_c_loader_matches_pure_loader_on_valid_files():
    mismatches = []
    for path in _yaml_files():
        text = path.read_text(encoding="utf-8")
        pure = _outcome(yaml.SafeLoader, text)
        if pure[0] != "ok":
            continue
        if yaml.load(text, Loader=yaml.CSafeLoader) != pure[1]:
            mismatches.append(path.relative_to(ROOT).as_posix())
    assert mismatches == []
