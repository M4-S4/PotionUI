from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_RUN = Path(__file__).resolve().parent / "run.py"
_HARNESS_DIR = _RUN.parents[1] / "harness"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

_spec = importlib.util.spec_from_file_location("e2e_ui_run_discover", _RUN)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True)


def _repo_with_specs(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    (tmp_path / ".gitignore").write_text("*shots.spec.ts\n")
    for name in ("alpha.spec.ts", "beta.spec.ts", "local-shots.spec.ts"):
        (specs / name).write_text("")
    _git(tmp_path, "init", "-q")
    return specs


def test_ignored_specs_are_skipped_and_the_rest_kept(tmp_path, monkeypatch):
    specs = _repo_with_specs(tmp_path)
    monkeypatch.setattr(run, "SPECS_DIR", specs)
    monkeypatch.setattr(run, "REPO_ROOT", tmp_path)
    assert run.discover_specs() == ["alpha", "beta"]


def test_an_unstaged_new_spec_still_runs(tmp_path, monkeypatch):
    specs = _repo_with_specs(tmp_path)
    (specs / "fresh.spec.ts").write_text("")
    monkeypatch.setattr(run, "SPECS_DIR", specs)
    monkeypatch.setattr(run, "REPO_ROOT", tmp_path)
    assert "fresh" in run.discover_specs()


def test_without_git_every_spec_is_listed(tmp_path, monkeypatch):
    specs = tmp_path / "specs"
    specs.mkdir()
    for name in ("alpha.spec.ts", "local-shots.spec.ts"):
        (specs / name).write_text("")
    monkeypatch.setattr(run, "SPECS_DIR", specs)
    monkeypatch.setattr(run, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    assert run.discover_specs() == ["alpha", "local-shots"]


def test_spec_filters_do_not_match_a_longer_spec_name():
    import re

    pattern = re.compile(run.spec_filters(["prompt-conditions"])[0])
    assert pattern.search("tests/e2e/prompt-conditions.spec.ts")
    assert not pattern.search("tests/e2e/prompt-conditions-shots.spec.ts")
