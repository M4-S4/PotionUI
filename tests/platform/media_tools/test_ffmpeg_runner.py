from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import psutil
import pytest

from src.platform.media_tools import (
    FfmpegRunner,
    FfmpegUnavailable,
    ToolCancelled,
    ToolContext,
    ToolFailed,
)
from src.platform.media_tools.ffmpeg import parse_progress_line
from tests.platform.media_tools.media_factory import make_video

pytestmark = pytest.mark.usefixtures("need_ffmpeg")


@pytest.fixture(autouse=True)
def no_leaked_ffmpeg(tmp_path):
    yield
    for pid in processes_writing(tmp_path):
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            pass


def long_encode(out: Path) -> list[str]:
    return [
        "-re", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=20",
        "-c:v", "libx264", "-preset", "ultrafast", str(out),
    ]


def processes_writing(path: Path) -> list[int]:
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        cmdline = " ".join(process.info["cmdline"] or [])
        if str(path) in cmdline and process.info["pid"] != os.getpid():
            found.append(process.info["pid"])
    return found


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.mark.parametrize(
    "line,expected",
    [
        ("out_time_us=1500000", 1.5),
        ("out_time_ms=2000000", 2.0),
        ("out_time=00:01:02.500000", 62.5),
        ("out_time_us=N/A", None),
        ("out_time=N/A", None),
        ("progress=continue", None),
        ("frame=12", None),
        ("", None),
        ("out_time_us=-5", 0.0),
    ],
)
def test_parse_progress_line(line, expected):
    assert parse_progress_line(line) == expected


def test_runner_builds_one_recorded_command_with_progress_pipe(tmp_path):
    runner = FfmpegRunner()
    out = tmp_path / "out.mp4"

    runner.run(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.5", "-pix_fmt", "yuv420p", str(out)])

    assert out.stat().st_size > 0
    assert len(runner.commands) == 1
    command = runner.commands[0]
    assert command[0] == runner.binary()
    assert command[command.index("-progress") + 1] == "pipe:1"
    assert "-nostdin" in command and "-y" in command
    assert command[-1] == str(out)


def test_runner_stringifies_path_arguments(tmp_path):
    runner = FfmpegRunner()
    out = tmp_path / "out.mp4"

    runner.run(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.3", out])

    assert out.exists()
    assert all(isinstance(part, str) for part in runner.commands[0])


def test_progress_reports_increasing_fractions_below_one(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=3.0, size=(160, 120), fps=25)
    seen = []

    FfmpegRunner().run(
        ["-i", str(source), "-c:v", "libx264", "-preset", "slow", str(tmp_path / "out.mp4")],
        duration=3.0,
        on_progress=lambda fraction, label: seen.append(fraction),
    )

    assert seen
    assert seen == sorted(seen)
    assert all(0.0 <= fraction < 1.0 for fraction in seen)


def test_progress_is_capped_below_one_when_the_estimate_is_short(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=2.0, size=(160, 120), fps=25)
    seen = []

    FfmpegRunner().run(
        ["-i", str(source), "-c:v", "libx264", "-preset", "slow", str(tmp_path / "out.mp4")],
        duration=0.5,
        on_progress=lambda fraction, label: seen.append(fraction),
    )

    assert seen
    assert max(seen) == pytest.approx(0.99)


def test_no_progress_without_a_known_duration(tmp_path):
    seen = []

    FfmpegRunner().run(
        ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=0.5", str(tmp_path / "out.mp4")],
        on_progress=lambda fraction, label: seen.append(fraction),
    )

    assert seen == []


def test_failure_raises_a_generic_error_without_stderr_or_paths(tmp_path):
    secret = tmp_path / "secret-dir" / "missing.mp4"

    with pytest.raises(ToolFailed) as raised:
        FfmpegRunner().run(["-i", str(secret), str(tmp_path / "out.mp4")])

    assert str(tmp_path) not in raised.value.message
    assert "secret-dir" not in raised.value.message
    assert raised.value.message == "Failed to encode the media"


def test_missing_binary_is_reported_as_unavailable(tmp_path):
    runner = FfmpegRunner(ffmpeg=str(tmp_path / "no-such-ffmpeg"))

    with pytest.raises(FfmpegUnavailable):
        runner.run(["-version"])


def test_unresolvable_ffmpeg_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr("src.platform.media_tools.ffmpeg.find_ffmpeg", lambda: None)

    with pytest.raises(FfmpegUnavailable):
        FfmpegRunner().binary()


def test_already_cancelled_run_never_starts_a_process(tmp_path):
    out = tmp_path / "never.mp4"

    with pytest.raises(ToolCancelled):
        FfmpegRunner().run(long_encode(out), is_cancelled=lambda: True)

    assert not out.exists()
    assert processes_writing(out) == []


def test_cancel_kills_the_ffmpeg_process_quickly(tmp_path):
    out = tmp_path / "long.mp4"
    flag = threading.Event()
    result = {}

    def work():
        try:
            FfmpegRunner().run(long_encode(out), is_cancelled=flag.is_set)
        except BaseException as exc:
            result["error"] = exc

    worker = threading.Thread(target=work)
    worker.start()
    assert wait_until(lambda: bool(processes_writing(out)))
    started = time.monotonic()
    flag.set()
    worker.join(timeout=15)

    assert not worker.is_alive()
    assert isinstance(result["error"], ToolCancelled)
    assert time.monotonic() - started < 10
    assert wait_until(lambda: processes_writing(out) == [])


def test_timeout_kills_the_process_and_reports_it(tmp_path):
    out = tmp_path / "slow.mp4"

    with pytest.raises(ToolFailed) as raised:
        FfmpegRunner(timeout=0.5).run(long_encode(out))

    assert "timed out" in raised.value.message
    assert wait_until(lambda: processes_writing(out) == [])


def test_context_ffmpeg_forwards_progress_and_cancel(tmp_path):
    source = make_video(tmp_path / "src.mp4", seconds=2.0, size=(160, 120), fps=25)
    seen = []
    context = ToolContext(tmp_path / "scratch", progress=lambda fraction, label: seen.append(fraction))

    context.ffmpeg(["-i", str(source), "-c:v", "libx264", str(context.new_path(".mp4"))], duration=2.0)

    assert seen and seen == sorted(seen)
    assert len(context.commands) == 1

    cancelled = ToolContext(tmp_path / "scratch2", is_cancelled=lambda: True)
    with pytest.raises(ToolCancelled):
        cancelled.ffmpeg(["-i", str(source), str(cancelled.new_path(".mp4"))])
