from __future__ import annotations

import re
import stat
import subprocess
import time
from pathlib import Path

from src.platform.media_tools import FfmpegRunner, ToolContext, ToolMedia
from src.platform.util.video_transcode import find_ffmpeg

_FRAME_RE = re.compile(r"frame=\s*(\d+)")

SLOW_SCRIPT = "\n".join(
    [
        chr(35) + "!/bin/sh",
        "for last; do :; done",
        'echo partial > "$last"',
        "exec sleep 10",
        "",
    ]
)


def count_video_frames(path: Path) -> int:
    result = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    return int(_FRAME_RE.findall(result.stderr)[-1])


def slow_runner(tmp_path: Path) -> FfmpegRunner:
    script = tmp_path / "slow-ffmpeg.sh"
    script.write_text(SLOW_SCRIPT)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return FfmpegRunner(ffmpeg=str(script))


class CancelAfter:
    def __init__(self, seconds: float):
        self.deadline = time.monotonic() + seconds

    def __call__(self) -> bool:
        return time.monotonic() >= self.deadline


def slow_context(tmp_path: Path, seconds: float = 0.3) -> ToolContext:
    return ToolContext(
        tmp_path / "scratch",
        runner=slow_runner(tmp_path),
        is_cancelled=CancelAfter(seconds),
    )


def files_under(path: Path) -> list[Path]:
    return [item for item in path.rglob("*") if item.is_file()] if path.exists() else []


def fake_video(path: Path, **overrides) -> ToolMedia:
    fields = dict(kind="video", width=64, height=48, duration=2.0, fps=10.0, has_audio=True, has_alpha=False)
    fields.update(overrides)
    return ToolMedia(path=path, **fields)
