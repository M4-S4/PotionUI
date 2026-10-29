from __future__ import annotations

import math
import shutil

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.fastpath import AUDIO_OUTPUT_SUFFIXES, FALLBACK_AUDIO_SUFFIX
from src.platform.media_tools.types import (
    InvalidToolParams,
    ToolContext,
    ToolFailed,
    ToolMedia,
    ToolSpec,
)

MAX_SPLIT_PARTS = 200

SPEC = dict(
    id="split",
    label="Split",
    icon="split",
    accepts=("audio",),
    produces="audio",
    cardinality="one_to_many",
    editor="split",
    name_suffix="-part",
    params=[
        {"name": "part_seconds", "type": "float", "label": "Part length", "min": 0.1, "unit": "s"},
    ],
)


class Split(MediaTool):
    def run(self, inputs: list[ToolMedia], params: dict, ctx: ToolContext) -> list[ToolMedia]:
        parts: list[ToolMedia] = []
        for media in inputs:
            parts.extend(self._split(media, float(params["part_seconds"]), ctx))
        return parts

    def _split(self, media: ToolMedia, part_seconds: float, ctx: ToolContext) -> list[ToolMedia]:
        if not math.isfinite(part_seconds) or part_seconds <= 0:
            raise InvalidToolParams("part_seconds", "The part length must be a positive number of seconds")
        if media.duration is None:
            raise ToolFailed("Could not read the media's duration")
        if part_seconds >= media.duration:
            raise InvalidToolParams(
                "part_seconds",
                f"The part length ({part_seconds:g}s) is not shorter than the media "
                f"({media.duration:.3f}s), so there is nothing to split",
            )
        part_count = math.ceil(media.duration / part_seconds)
        if part_count > MAX_SPLIT_PARTS:
            raise InvalidToolParams(
                "part_seconds",
                f"Splitting into {part_count} parts exceeds the {MAX_SPLIT_PARTS}-part limit",
            )

        suffix = media.path.suffix.lower()
        if suffix not in AUDIO_OUTPUT_SUFFIXES:
            suffix = FALLBACK_AUDIO_SUFFIX
        parts_dir = ctx.new_path("", stem="parts")
        parts_dir.mkdir(parents=True, exist_ok=True)
        pattern = parts_dir / f"part_%03d{suffix}"
        try:
            ctx.ffmpeg(
                [
                    "-i", str(media.path),
                    "-f", "segment",
                    "-segment_time", f"{part_seconds:.6f}",
                    "-reset_timestamps", "1",
                    "-vn",
                    str(pattern),
                ],
                duration=media.duration,
            )
            produced = sorted(parts_dir.iterdir())
            if not produced or any(part.stat().st_size == 0 for part in produced):
                raise ToolFailed("The encoder produced an incomplete split")
            return [ctx.probe(part, "audio") for part in produced]
        except BaseException:
            shutil.rmtree(parts_dir, ignore_errors=True)
            raise


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Split())
