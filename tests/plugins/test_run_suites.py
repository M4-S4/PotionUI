import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

_TESTS_PLUGINS_DIR = Path(__file__).resolve().parent
if str(_TESTS_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_PLUGINS_DIR))

import run_suites  # noqa: E402


class _FakeCompletedProcess:
    def __init__(self, returncode, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _make_marketplace(tmp_path, plugin_names):
    marketplace = tmp_path / "content" / "plugins" / "marketplace"
    for name in plugin_names:
        (marketplace / name / "tests").mkdir(parents=True)
    return marketplace


class TestRunSuitesConcurrency:
    def test_all_plugins_run_and_the_right_one_fails(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_suites.py"])
        marketplace = _make_marketplace(tmp_path, ["alpha", "bravo", "charlie"])

        outcomes = {"alpha": (0, 0.03), "bravo": (1, 0.01), "charlie": (0, 0.02)}

        def fake_run(cmd, cwd, capture_output, text):
            plugin_name = Path(cmd[3]).parent.name
            returncode, delay = outcomes[plugin_name]
            time.sleep(delay)
            return _FakeCompletedProcess(returncode, stdout=f"ran {plugin_name}")

        with patch.object(run_suites, "MARKETPLACE_ROOT", marketplace), \
                patch.object(run_suites, "REPO_ROOT", tmp_path), \
                patch.object(run_suites.subprocess, "run", side_effect=fake_run):
            exit_code = run_suites.main()

        out = capsys.readouterr().out
        assert exit_code == 1
        assert "FAILED plugin suites: bravo" in out
        assert "ran alpha" in out
        assert "ran bravo" in out
        assert "ran charlie" in out

    def test_all_pass_returns_zero(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_suites.py"])
        marketplace = _make_marketplace(tmp_path, ["alpha", "bravo"])

        def fake_run(cmd, cwd, capture_output, text):
            return _FakeCompletedProcess(0, stdout="ok")

        with patch.object(run_suites, "MARKETPLACE_ROOT", marketplace), \
                patch.object(run_suites, "REPO_ROOT", tmp_path), \
                patch.object(run_suites.subprocess, "run", side_effect=fake_run):
            exit_code = run_suites.main()

        assert exit_code == 0
        assert "All 2 plugin suite(s) passed." in capsys.readouterr().out

    def test_plugins_run_concurrently_not_serially(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_suites.py"])
        monkeypatch.setenv(run_suites.WORKERS_ENV, "3")
        marketplace = _make_marketplace(tmp_path, ["alpha", "bravo", "charlie"])
        spans = {}
        gate = threading.Barrier(3, timeout=5)

        def fake_run(cmd, cwd, capture_output, text):
            name = Path(cmd[3]).parent.name
            started = time.perf_counter()
            gate.wait()
            spans[name] = (started, time.perf_counter())
            return _FakeCompletedProcess(0, stdout="ok")

        with patch.object(run_suites, "MARKETPLACE_ROOT", marketplace), \
                patch.object(run_suites, "REPO_ROOT", tmp_path), \
                patch.object(run_suites.subprocess, "run", side_effect=fake_run):
            exit_code = run_suites.main()

        assert exit_code == 0
        assert len(spans) == 3
        assert max(s for s, _ in spans.values()) < min(e for _, e in spans.values())


class TestWorkerCount:
    def test_defaults_to_cpu_count_capped_by_suite_count(self, monkeypatch):
        monkeypatch.delenv(run_suites.WORKERS_ENV, raising=False)
        monkeypatch.setattr(run_suites.os, "cpu_count", lambda: 4)
        assert run_suites._worker_count(10) == 4
        assert run_suites._worker_count(2) == 2

    def test_unknown_cpu_count_falls_back_to_one(self, monkeypatch):
        monkeypatch.delenv(run_suites.WORKERS_ENV, raising=False)
        monkeypatch.setattr(run_suites.os, "cpu_count", lambda: None)
        assert run_suites._worker_count(5) == 1

    def test_env_overrides_cpu_count_but_not_suite_count(self, monkeypatch):
        monkeypatch.setattr(run_suites.os, "cpu_count", lambda: 4)
        monkeypatch.setenv(run_suites.WORKERS_ENV, "7")
        assert run_suites._worker_count(10) == 7
        assert run_suites._worker_count(3) == 3

    def test_invalid_env_is_ignored(self, monkeypatch):
        monkeypatch.setattr(run_suites.os, "cpu_count", lambda: 4)
        for value in ("zero", "0", "-2"):
            monkeypatch.setenv(run_suites.WORKERS_ENV, value)
            assert run_suites._worker_count(10) == 4

    def test_main_uses_the_worker_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_suites.py"])
        monkeypatch.setenv(run_suites.WORKERS_ENV, "1")
        marketplace = _make_marketplace(tmp_path, ["alpha", "bravo", "charlie"])
        live = []
        peak = []

        def fake_run(cmd, cwd, capture_output, text):
            live.append(1)
            peak.append(len(live))
            time.sleep(0.01)
            live.pop()
            return _FakeCompletedProcess(0, stdout="ok")

        with patch.object(run_suites, "MARKETPLACE_ROOT", marketplace), \
                patch.object(run_suites, "REPO_ROOT", tmp_path), \
                patch.object(run_suites.subprocess, "run", side_effect=fake_run):
            assert run_suites.main() == 0

        assert max(peak) == 1


class TestOrdering:
    def test_biggest_suite_first_ties_by_name(self, tmp_path):
        marketplace = _make_marketplace(tmp_path, ["alpha", "bravo", "charlie", "delta"])
        for name, count in {"alpha": 1, "bravo": 3, "charlie": 3, "delta": 2}.items():
            nested = marketplace / name / "tests" / "unit"
            nested.mkdir()
            for index in range(count):
                (nested / f"test_{index}.py").write_text("", encoding="utf-8")
        dirs = sorted(marketplace.glob("*/tests"))

        ordered = run_suites._order_big_first(dirs)

        assert [d.parent.name for d in ordered] == ["bravo", "charlie", "delta", "alpha"]

    def test_main_submits_big_suites_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["run_suites.py"])
        monkeypatch.setenv(run_suites.WORKERS_ENV, "1")
        marketplace = _make_marketplace(tmp_path, ["alpha", "bravo"])
        for index in range(2):
            (marketplace / "bravo" / "tests" / f"test_{index}.py").write_text("", encoding="utf-8")
        (marketplace / "alpha" / "tests" / "test_0.py").write_text("", encoding="utf-8")
        order = []

        def fake_run(cmd, cwd, capture_output, text):
            order.append(Path(cmd[3]).parent.name)
            return _FakeCompletedProcess(0, stdout="ok")

        with patch.object(run_suites, "MARKETPLACE_ROOT", marketplace), \
                patch.object(run_suites, "REPO_ROOT", tmp_path), \
                patch.object(run_suites.subprocess, "run", side_effect=fake_run):
            assert run_suites.main() == 0

        assert order == ["bravo", "alpha"]
