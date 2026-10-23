import os
import threading

import psutil
import pytest

from src.platform.runtime import thread_priority
from src.platform.runtime.thread_priority import ThreadPriority, set_current_thread_priority


class FakeProcess:
    created = []
    ionice_calls = []
    fail = False

    def __init__(self, pid):
        self.pid = pid
        FakeProcess.created.append(pid)

    def ionice(self, ioclass, value=None):
        if FakeProcess.fail:
            raise psutil.AccessDenied(self.pid)
        FakeProcess.ionice_calls.append((self.pid, ioclass, value))


class FakeKernel32:
    def __init__(self, succeed=True):
        self.calls = []
        self.succeed = succeed

    def GetCurrentThread(self):
        return -2

    def SetThreadPriority(self, handle, value):
        self.calls.append((handle, value))
        return 1 if self.succeed else 0


@pytest.fixture
def linux(monkeypatch):
    calls = []
    FakeProcess.created = []
    FakeProcess.ionice_calls = []
    FakeProcess.fail = False
    monkeypatch.setattr(thread_priority, "_platform_name", lambda: "linux")
    monkeypatch.setattr(os, "PRIO_PROCESS", 0, raising=False)
    monkeypatch.setattr(os, "setpriority", lambda which, who, nice: calls.append((which, who, nice)), raising=False)
    monkeypatch.setattr(psutil, "Process", FakeProcess)
    monkeypatch.setattr(psutil, "IOPRIO_CLASS_BE", 2, raising=False)
    return calls


def _on_worker_thread(fn):
    outcome = {}

    def run():
        outcome["thread_id"] = threading.get_native_id()
        outcome["result"] = fn()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return outcome


class TestLinux:
    def test_background_lowers_cpu_and_io_priority_of_the_calling_thread_only(self, linux):
        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BACKGROUND))

        assert outcome["result"] is True
        assert linux == [(0, outcome["thread_id"], 19)]
        assert outcome["thread_id"] != os.getpid()
        assert FakeProcess.ionice_calls == [(outcome["thread_id"], 2, 7)]

    def test_below_normal_uses_the_middle_levels(self, linux):
        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BELOW_NORMAL))

        assert outcome["result"] is True
        assert linux == [(0, outcome["thread_id"], 10)]
        assert FakeProcess.ionice_calls == [(outcome["thread_id"], 2, 6)]

    def test_normal_changes_nothing(self, linux):
        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.NORMAL))

        assert outcome["result"] is False
        assert linux == []
        assert FakeProcess.created == []

    def test_the_main_thread_is_never_touched(self, linux):
        assert set_current_thread_priority(ThreadPriority.BACKGROUND) is False

        assert linux == []
        assert FakeProcess.created == []

    def test_a_refused_io_priority_does_not_undo_the_cpu_priority(self, linux):
        FakeProcess.fail = True

        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BACKGROUND))

        assert outcome["result"] is True
        assert len(linux) == 1

    def test_a_refused_cpu_priority_reports_false_without_raising(self, linux, monkeypatch):
        def refuse(which, who, nice):
            raise PermissionError("not allowed")

        monkeypatch.setattr(os, "setpriority", refuse, raising=False)

        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BACKGROUND))

        assert outcome["result"] is False
        assert FakeProcess.ionice_calls == []


class TestWindows:
    @pytest.fixture
    def kernel32(self, monkeypatch):
        fake = FakeKernel32()
        monkeypatch.setattr(thread_priority, "_platform_name", lambda: "win32")
        monkeypatch.setattr(thread_priority, "_windows_kernel32", lambda: fake)
        return fake

    def test_background_enters_background_mode_on_the_current_thread_handle(self, kernel32):
        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BACKGROUND))

        assert outcome["result"] is True
        assert kernel32.calls == [(-2, 0x00010000)]

    def test_below_normal_sets_the_below_normal_thread_priority(self, kernel32):
        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BELOW_NORMAL))

        assert outcome["result"] is True
        assert kernel32.calls == [(-2, -1)]

    def test_a_failed_call_reports_false(self, kernel32):
        kernel32.succeed = False

        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BACKGROUND))

        assert outcome["result"] is False

    def test_normal_and_the_main_thread_are_left_alone(self, kernel32):
        assert set_current_thread_priority(ThreadPriority.NORMAL) is False
        assert set_current_thread_priority(ThreadPriority.BACKGROUND) is False

        assert kernel32.calls == []


class TestOtherPlatforms:
    def test_does_nothing_and_does_not_raise(self, monkeypatch):
        touched = []
        monkeypatch.setattr(thread_priority, "_platform_name", lambda: "darwin")
        monkeypatch.setattr(os, "setpriority", lambda *args: touched.append(args), raising=False)
        monkeypatch.setattr(thread_priority, "_windows_kernel32", lambda: touched.append("kernel32"))

        outcome = _on_worker_thread(lambda: set_current_thread_priority(ThreadPriority.BACKGROUND))

        assert outcome["result"] is False
        assert touched == []
