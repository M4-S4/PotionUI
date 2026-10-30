from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.platform.media_tools import ToolContext, ToolMedia, probe, run_chain, validate_params
from src.platform.media_tools.ops import core_specs
from tests.platform.media_tools.media_factory import run_ffmpeg

SPECS = {spec.id: spec for spec in core_specs()}


def spec_of(tool_id: str):
    return SPECS[tool_id]


def tool_of(tool_id: str):
    return SPECS[tool_id].impl


def resolved(tool_id: str, **params) -> dict:
    return validate_params(SPECS[tool_id], params)


def run_steps(steps: list[tuple[str, dict]], path: Path, ctx: ToolContext) -> ToolMedia:
    chain = [(SPECS[tool_id], params) for tool_id, params in steps]
    outputs = run_chain(chain, [probe(path)], ctx)
    assert len(outputs) == 1
    return outputs[0]


def run_one(tool_id: str, params: dict, path: Path, ctx: ToolContext) -> ToolMedia:
    return run_steps([(tool_id, params)], path, ctx)


def marker_image(size: tuple[int, int], marker: tuple[int, int], mode: str = "RGB") -> Image.Image:
    background = (0, 0, 0, 255) if mode == "RGBA" else (0, 0, 0)
    image = Image.new(mode, size, background)
    image.putpixel(marker, (255, 0, 0, 255) if mode == "RGBA" else (255, 0, 0))
    return image


def make_marker_video(
    path: Path,
    size: tuple[int, int] = (64, 48),
    box: tuple[int, int] = (16, 12),
    seconds: float = 1.0,
    audio: bool = True,
) -> Path:
    args = ["-f", "lavfi", "-i", f"color=c=blue:size={size[0]}x{size[1]}:rate=10:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
    args += ["-vf", f"drawbox=x=0:y=0:w={box[0]}:h={box[1]}:color=red:t=fill"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    args += [str(path)]
    run_ffmpeg(*args)
    return path


def make_odd_video(path: Path, size: tuple[int, int] = (65, 49), seconds: float = 1.0) -> Path:
    run_ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size={size[0]}x{size[1]}:rate=10:duration={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv444p",
        "-c:a", "aac", "-b:a", "64k", "-shortest", str(path),
    )
    return path


def first_frame(video: Path, target: Path) -> Image.Image:
    run_ffmpeg("-i", str(video), "-frames:v", "1", str(target))
    with Image.open(target) as frame:
        return frame.convert("RGB")


def is_red(pixel: tuple) -> bool:
    return pixel[0] > 150 and pixel[2] < 100


def is_blue(pixel: tuple) -> bool:
    return pixel[2] > 150 and pixel[0] < 100
