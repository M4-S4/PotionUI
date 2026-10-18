"""Cgroup-aware system RAM reading (src.platform.runtime.system_memory)."""

from collections import namedtuple

import pytest

from src.platform.runtime import system_memory

_FakeVirtualMemory = namedtuple("_FakeVirtualMemory", ["total", "available"])

_GB = 1024 ** 3

_V1_UNLIMITED = "9223372036854771712"


@pytest.fixture(autouse=True)
def _reset_warning():
    system_memory.reset_warning_for_tests()
    yield
    system_memory.reset_warning_for_tests()


def _fake_psutil(monkeypatch, total_gb: float, available_gb: float):
    monkeypatch.setattr(
        system_memory.psutil,
        "virtual_memory",
        lambda: _FakeVirtualMemory(total=int(total_gb * _GB), available=int(available_gb * _GB)),
    )


def _host(tmp_path, proc="0::/\n", files=None):
    """A fake cgroup mount under tmp_path/cgroup and a fake /proc/self/cgroup; files maps a path under the mount to its text."""
    root = tmp_path / "cgroup"
    root.mkdir()
    for name, text in (files or {}).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    proc_path = tmp_path / "proc-self-cgroup"
    proc_path.write_text(proc)
    return {"cgroup_root": root, "proc_cgroup_path": proc_path}


def _read(tmp_path, proc="0::/\n", files=None):
    return system_memory.get_system_memory(**_host(tmp_path, proc, files))


class TestNoCgroupLimit:
    def test_missing_cgroup_files_pass_through_psutil(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(tmp_path)
        assert mem.total == int(64.0 * _GB)
        assert mem.available == int(40.0 * _GB)

    def test_missing_proc_file_and_mount_pass_through_psutil(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = system_memory.get_system_memory(cgroup_root=tmp_path / "none", proc_cgroup_path=tmp_path / "none-proc")
        assert mem.total == int(64.0 * _GB)
        assert mem.available == int(40.0 * _GB)

    def test_unlimited_cgroup_max_passes_through_psutil(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(tmp_path, files={"memory.max": "max\n", "memory.current": str(int(5 * _GB))})

        assert mem.total == int(64.0 * _GB)
        assert mem.available == int(40.0 * _GB)


class TestCgroupLimitPresent:
    def test_limit_below_host_ram_clamps_total_and_available(self, monkeypatch, tmp_path):
        # Host has 64GB; container is capped to 16GB via cgroup v2, 10GB already used.
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(tmp_path, files={"memory.max": str(16 * _GB), "memory.current": str(10 * _GB)})

        assert mem.total == 16 * _GB
        assert mem.available == 6 * _GB  # 16GB limit - 10GB current usage

    def test_available_never_exceeds_psutils_own_reading(self, monkeypatch, tmp_path):
        # cgroup says lots of headroom, but psutil (host-wide) reports less
        # available than the cgroup math alone would suggest — never claim
        # more available RAM than psutil itself sees.
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=2.0)
        mem = _read(tmp_path, files={"memory.max": str(32 * _GB), "memory.current": str(1 * _GB)})

        assert mem.total == 32 * _GB
        assert mem.available == 2 * _GB

    def test_missing_current_file_falls_back_to_psutil_available(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=12.0)
        mem = _read(tmp_path, files={"memory.max": str(16 * _GB)})

        assert mem.total == 16 * _GB
        assert mem.available == 12 * _GB

    def test_current_exceeding_limit_clamps_available_to_zero(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(tmp_path, files={"memory.max": str(4 * _GB), "memory.current": str(6 * _GB)})

        assert mem.total == 4 * _GB
        assert mem.available == 0

    def test_unreadable_limit_value_passes_through_psutil(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(tmp_path, files={"memory.max": "not-a-number\n"})

        assert mem.total == int(64.0 * _GB)
        assert mem.available == int(40.0 * _GB)

    def test_logs_loudly_exactly_once(self, monkeypatch, tmp_path, caplog):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        host = _host(tmp_path, files={"memory.max": str(16 * _GB), "memory.current": str(10 * _GB)})

        with caplog.at_level("WARNING", logger=system_memory.__name__):
            system_memory.get_system_memory(**host)
            system_memory.get_system_memory(**host)

        limit_warnings = [r for r in caplog.records if "cgroup v2 memory limit detected" in r.message]
        assert len(limit_warnings) == 1


class TestPageCache:
    def test_the_page_cache_of_a_file_read_in_a_two_gb_container_counts_as_free(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        stat = "anon 26034176\nfile 1259085824\nactive_file 28672\ninactive_file 1259057152\n"
        mem = _read(tmp_path, files={"memory.max": str(2 * _GB), "memory.current": "1285120000", "memory.stat": stat})

        assert mem.total == 2 * _GB
        assert mem.available == 2 * _GB - (1285120000 - 1259057152)

    def test_active_file_pages_stay_used(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        stat = "active_file %d\ninactive_file 0\n" % (3 * _GB)
        mem = _read(tmp_path, files={"memory.max": str(8 * _GB), "memory.current": str(4 * _GB), "memory.stat": stat})

        assert mem.available == 4 * _GB

    def test_inactive_file_larger_than_current_never_exceeds_the_limit(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        stat = "inactive_file %d\n" % (5 * _GB)
        mem = _read(tmp_path, files={"memory.max": str(8 * _GB), "memory.current": str(4 * _GB), "memory.stat": stat})

        assert mem.available == 8 * _GB


class TestOwnCgroup:
    def test_a_container_in_the_hosts_cgroup_namespace_reads_the_limit_of_its_own_folder_not_the_mount_root(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=62.7, available_gb=59.8)
        own = "docker/ae445fa884de"
        mem = _read(
            tmp_path,
            proc=f"0::/{own}\n",
            files={f"{own}/memory.max": str(2 * _GB), f"{own}/memory.current": str(_GB // 2)},
        )

        assert mem.total == 2 * _GB
        assert mem.available == 2 * _GB - _GB // 2

    def test_the_tightest_ancestor_limit_wins(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(
            tmp_path,
            proc="0::/user.slice/app.scope\n",
            files={
                "user.slice/memory.max": str(6 * _GB),
                "user.slice/memory.current": str(5 * _GB),
                "user.slice/app.scope/memory.max": str(8 * _GB),
                "user.slice/app.scope/memory.current": str(2 * _GB),
            },
        )

        assert mem.total == 6 * _GB
        assert mem.available == 1 * _GB

    def test_an_own_folder_missing_from_the_mount_reads_the_mount_root(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(
            tmp_path,
            proc="0::/kubepods/pod1/abc\n",
            files={"memory.max": str(4 * _GB), "memory.current": str(1 * _GB)},
        )

        assert mem.total == 4 * _GB
        assert mem.available == 3 * _GB

    def test_a_path_outside_the_namespace_reads_the_mount_root(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        mem = _read(tmp_path, proc="0::/../sibling\n", files={"memory.max": str(4 * _GB), "memory.current": str(1 * _GB)})

        assert mem.total == 4 * _GB
        assert mem.available == 3 * _GB


class TestCgroupV1:
    def test_v1_limit_usage_and_inactive_file(self, monkeypatch, tmp_path, caplog):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        files = {
            "memory/memory.limit_in_bytes": str(4 * _GB),
            "memory/memory.usage_in_bytes": str(3 * _GB),
            "memory/memory.stat": "cache %d\ntotal_inactive_file %d\n" % (2 * _GB, 1 * _GB),
        }
        with caplog.at_level("WARNING", logger=system_memory.__name__):
            mem = _read(tmp_path, proc="12:memory:/docker/abc\n11:cpu,cpuacct:/docker/abc\n", files=files)

        assert mem.total == 4 * _GB
        assert mem.available == 2 * _GB
        assert any("cgroup v1 memory limit detected" in r.message for r in caplog.records)

    def test_v1_unlimited_passes_through_psutil(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        files = {"memory/memory.limit_in_bytes": _V1_UNLIMITED, "memory/memory.usage_in_bytes": str(3 * _GB)}
        mem = _read(tmp_path, proc="12:memory:/\n", files=files)

        assert mem.total == int(64.0 * _GB)
        assert mem.available == int(40.0 * _GB)

    def test_v1_parent_limit_of_a_host_process(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        files = {
            "memory/memory.limit_in_bytes": _V1_UNLIMITED,
            "memory/batch/memory.limit_in_bytes": str(8 * _GB),
            "memory/batch/memory.usage_in_bytes": str(6 * _GB),
            "memory/batch/job/memory.limit_in_bytes": _V1_UNLIMITED,
            "memory/batch/job/memory.usage_in_bytes": str(1 * _GB),
        }
        mem = _read(tmp_path, proc="4:memory:/batch/job\n", files=files)

        assert mem.total == 8 * _GB
        assert mem.available == 2 * _GB

    def test_a_v2_limit_comes_before_v1(self, monkeypatch, tmp_path):
        _fake_psutil(monkeypatch, total_gb=64.0, available_gb=40.0)
        files = {
            "memory.max": str(16 * _GB),
            "memory.current": str(4 * _GB),
            "memory/memory.limit_in_bytes": str(2 * _GB),
            "memory/memory.usage_in_bytes": str(1 * _GB),
        }
        mem = _read(tmp_path, proc="0::/\n12:memory:/\n", files=files)

        assert mem.total == 16 * _GB
        assert mem.available == 12 * _GB


class TestSystemMemoryProperties:
    def test_total_gb_and_available_gb_convert_bytes(self):
        mem = system_memory.SystemMemory(total=8 * _GB, available=4 * _GB)
        assert mem.total_gb == pytest.approx(8.0)
        assert mem.available_gb == pytest.approx(4.0)
