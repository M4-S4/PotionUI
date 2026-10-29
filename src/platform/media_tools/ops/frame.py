from __future__ import annotations

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.types import (
    InvalidToolParams,
    ToolContext,
    ToolFailed,
    ToolMedia,
    ToolSpec,
)

LAST_FRAME_LOOKBACK_SECONDS = 1.0

SPEC = dict(
    id="frame",
    label="Frame",
    icon="frame",
    accepts=("video",),
    produces="image",
    cardinality="one_to_one",
    editor="frame",
    name_suffix="-frame",
    params=[
        {"name": "time", "type": "float", "label": "Time", "default": 0, "min": 0, "unit": "s"},
    ],
)


def _written(path) -> bool:
    return path.is_file() and path.stat().st_size > 0


class Frame(MediaTool):
    def run(self, inputs: list[ToolMedia], params: dict, ctx: ToolContext) -> list[ToolMedia]:
        return [self._extract(media, float(params.get("time") or 0.0), ctx) for media in inputs]

    def _extract(self, media: ToolMedia, time: float, ctx: ToolContext) -> ToolMedia:
        if media.duration is None:
            raise ToolFailed("Could not read the video's duration")
        if time < 0:
            raise InvalidToolParams("time", "The frame time cannot be negative")
        if time >= media.duration:
            raise InvalidToolParams(
                "time",
                f"The frame time ({time:g}s) is past the end of the video ({media.duration:.3f}s)",
            )
        out_path = ctx.new_path(".png", stem=media.path.stem[:40] or "frame")
        try:
            ctx.ffmpeg(
                ["-ss", f"{time:.6f}", "-i", str(media.path), "-frames:v", "1", str(out_path)],
            )
            if not _written(out_path):
                out_path.unlink(missing_ok=True)
                lookback = max(0.0, media.duration - LAST_FRAME_LOOKBACK_SECONDS)
                ctx.ffmpeg(
                    ["-ss", f"{lookback:.6f}", "-i", str(media.path), "-update", "1", str(out_path)],
                )
            if not _written(out_path):
                raise ToolFailed("The encoder produced an empty file")
            return ctx.probe(out_path, "image")
        except BaseException:
            out_path.unlink(missing_ok=True)
            raise


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Frame())
