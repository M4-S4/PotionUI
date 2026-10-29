from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image

from src.platform.util.video_transcode import find_ffmpeg


def ffmpeg_available() -> bool:
    return find_ffmpeg() is not None


def run_ffmpeg(*args: str) -> None:
    subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", *args],
        check=True,
        capture_output=True,
    )


def make_video(
    path: Path,
    seconds: float = 1.0,
    size: tuple[int, int] = (64, 48),
    fps: int = 10,
    audio: bool = True,
    freq: int = 440,
) -> Path:
    args = ["-f", "lavfi", "-i", f"testsrc=size={size[0]}x{size[1]}:rate={fps}:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=44100:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    args += [str(path)]
    run_ffmpeg(*args)
    return path


def make_audio(path: Path, seconds: float = 2.0, freq: int = 440) -> Path:
    run_ffmpeg("-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=22050:duration={seconds}", str(path))
    return path


def make_image(
    path: Path,
    size: tuple[int, int] = (40, 30),
    mode: str = "RGB",
    color: tuple | str | int = (200, 40, 40),
) -> Path:
    Image.new(mode, size, color).save(path)
    return path
