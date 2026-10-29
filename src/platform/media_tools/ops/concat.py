from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.platform.media_tools.base import MediaTool
from src.platform.media_tools.types import (
    InvalidToolParams,
    ToolContext,
    ToolFailed,
    ToolMedia,
    ToolSpec,
    UnsupportedMedia,
)

SAMPLE_RATE = 48000
DEFAULT_FPS = 24.0

SPEC = dict(
    id="concat",
    label="Concat",
    icon="concat",
    accepts=("video",),
    produces="video",
    cardinality="many_to_one",
    selection={"min": 2, "max": 8},
    params=[
        {
            "name": "overlap",
            "type": "select",
            "label": "Overlap",
            "default": "cut",
            "options": [
                {"label": "Cut", "value": "cut"},
                {"label": "Crossfade", "value": "crossfade"},
            ],
        },
        {
            "name": "overlap_frames",
            "type": "integer",
            "label": "Overlap frames",
            "default": 0,
            "min": 0,
            "max": 48,
            "when": {"field": "overlap", "equals": "crossfade"},
        },
        {
            "name": "fit",
            "type": "select",
            "label": "Fit",
            "default": "pad",
            "options": [
                {"label": "Pad", "value": "pad"},
                {"label": "Cover", "value": "cover"},
            ],
        },
        {"name": "keep_audio", "type": "checkbox", "label": "Keep audio", "default": True},
    ],
)


def _even(value: int) -> int:
    return value - value % 2


def fit_filter(width: int, height: int, fit: str) -> str:
    if fit == "cover":
        return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )


def check_layout(lengths: Sequence[float], crossfades: Sequence[float]) -> None:
    joined = lengths[0]
    for fade, length in zip(crossfades, lengths[1:]):
        if fade < 0 or fade >= min(joined, length):
            raise InvalidToolParams(
                "overlap_frames",
                f"A {fade:.3f}s crossfade is longer than the clips it joins",
            )
        joined += length - fade


def blend_chain(lengths: Sequence[float], crossfades: Sequence[float], with_audio: bool) -> list[str]:
    filters: list[str] = []
    video, sound = "v0", "a0"
    elapsed = lengths[0]
    last = len(lengths) - 1
    for index in range(1, len(lengths)):
        fade = crossfades[index - 1]
        video_out = "v" if index == last else f"vj{index}"
        sound_out = "a" if index == last else f"aj{index}"
        if fade > 0:
            filters.append(
                f"[{video}][v{index}]xfade=transition=fade:duration={fade:.6f}:offset={elapsed - fade:.6f}[{video_out}]"
            )
            if with_audio:
                filters.append(f"[{sound}][a{index}]acrossfade=d={fade:.6f}[{sound_out}]")
        else:
            filters.append(f"[{video}][v{index}]concat=n=2:v=1:a=0[{video_out}]")
            if with_audio:
                filters.append(f"[{sound}][a{index}]concat=n=2:v=0:a=1[{sound_out}]")
        elapsed += lengths[index] - fade
        video, sound = video_out, sound_out
    return filters


def concat_args(
    clips: Sequence[ToolMedia],
    out_path: Path,
    *,
    crossfades: Sequence[float],
    audio: bool,
    fit: str,
) -> list[str]:
    first = clips[0]
    fps = first.fps if first.fps and first.fps > 0 else DEFAULT_FPS
    width, height = _even(int(first.width)), _even(int(first.height))
    with_audio = audio and any(clip.has_audio for clip in clips)
    blended = any(fade > 0 for fade in crossfades)
    lengths = [float(clip.duration) for clip in clips]

    args: list[str] = []
    for clip in clips:
        args += ["-i", str(clip.path)]
    silence_inputs: dict[int, int] = {}
    for index, clip in enumerate(clips):
        if with_audio and not clip.has_audio:
            silence_inputs[index] = len(clips) + len(silence_inputs)
            args += [
                "-f", "lavfi", "-t", f"{lengths[index]:.3f}",
                "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
            ]

    filters: list[str] = []
    joined = ""
    timebase = ",settb=AVTB" if blended else ""
    for index in range(len(clips)):
        filters.append(
            f"[{index}:v]{fit_filter(width, height, fit)},setsar=1,fps={fps:g},format=yuv420p{timebase}[v{index}]"
        )
        joined += f"[v{index}]"
        if with_audio:
            source = f"{silence_inputs[index]}:a" if index in silence_inputs else f"{index}:a"
            filters.append(
                f"[{source}]aresample={SAMPLE_RATE},aformat=channel_layouts=stereo[a{index}]"
            )
            joined += f"[a{index}]"
    if blended:
        filters += blend_chain(lengths, crossfades, with_audio)
    else:
        filters.append(
            f"{joined}concat=n={len(clips)}:v=1:a={1 if with_audio else 0}[v]" + ("[a]" if with_audio else "")
        )
    args += ["-filter_complex", ";".join(filters), "-map", "[v]"]
    if with_audio:
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out_path),
    ]
    return args


class Concat(MediaTool):
    def run(self, inputs: list[ToolMedia], params: dict, ctx: ToolContext) -> list[ToolMedia]:
        clips = list(inputs)
        if not clips:
            raise InvalidToolParams("inputs", "There are no clips to join")
        for clip in clips:
            if not clip.width or not clip.height or not clip.duration or clip.duration <= 0:
                raise UnsupportedMedia(f"The clip {clip.path.name} cannot be joined: its size or length is unknown")

        first = clips[0]
        rate = first.fps if first.fps and first.fps > 0 else DEFAULT_FPS
        overlap_frames = int(params.get("overlap_frames") or 0)
        fade = overlap_frames / rate if params.get("overlap") == "crossfade" and overlap_frames > 0 else 0.0
        crossfades = [fade] * (len(clips) - 1)
        lengths = [float(clip.duration) for clip in clips]
        check_layout(lengths, crossfades)

        out_path = ctx.new_path(".mp4", stem="joined")
        try:
            ctx.ffmpeg(
                concat_args(
                    clips,
                    out_path,
                    crossfades=crossfades,
                    audio=bool(params.get("keep_audio", True)),
                    fit=str(params.get("fit") or "pad"),
                ),
                duration=sum(lengths) - sum(crossfades),
            )
            if not out_path.is_file() or out_path.stat().st_size == 0:
                raise ToolFailed("The encoder produced an empty file")
            return [ctx.probe(out_path, "video")]
        except BaseException:
            out_path.unlink(missing_ok=True)
            raise


def spec() -> ToolSpec:
    return ToolSpec(**SPEC, impl=Concat())
