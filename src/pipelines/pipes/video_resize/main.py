import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.pipelines.contracts import (
    BasePipe,
    IOType,
    PipeConfigSpec,
    PipeInput,
    PipeInputSpec,
    PipeOutput,
    PipeOutputSpec,
    logger,
)
from src.pipelines.outputs import Icon, ProgressGenerationOutput
from src.pipelines.pipes._shared.media.video_concat import (
    FFMPEG_TIMEOUT_SECONDS,
    VideoConcatCancelled,
    run_cancellable,
    video_paths,
)
from src.pipelines.pipes._shared.video.frames import probe_video_stream
from src.platform.util.video_transcode import find_ffmpeg

QUALITY_CRF = {"high": 18, "balanced": 23, "small": 28}
ENCODER_PRESET = "medium"


class VideoResizeError(RuntimeError):
    pass


def parse_size(value: object) -> Tuple[int, int]:
    text = str(value or "").strip().lower().replace("×", "x")
    parts = text.split("x")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(f"The size must look like 1280x720, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width < 2 or height < 2:
        raise ValueError(f"The size must be at least 2x2, got {value!r}")
    return width, height


def _even(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def fit_size(source: Tuple[int, int], requested: Tuple[int, int]) -> Tuple[int, int, bool]:
    src_w, src_h = source
    scale = min(requested[0] / src_w, requested[1] / src_h)
    clamped = scale > 1.0
    scale = min(scale, 1.0)
    return _even(round(src_w * scale)), _even(round(src_h * scale)), clamped


def resize_command(
    ffmpeg: str, clip: str, out_path: Path, width: int, height: int, crf: int, audio_codec: str,
) -> List[str]:
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(clip),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale={width}:{height}:flags=lanczos,setsar=1",
        "-c:v", "libx264", "-preset", ENCODER_PRESET, "-crf", str(crf), "-pix_fmt", "yuv420p",
    ]
    command += ["-c:a", "copy"] if audio_codec == "copy" else ["-c:a", "aac", "-b:a", "192k"]
    return command + ["-movflags", "+faststart", str(out_path)]


def resize_video(
    clip: str,
    out_path: Path,
    width: int,
    height: int,
    crf: int,
    *,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise VideoResizeError("ffmpeg is needed to resize a video")
    stderr = b""
    for audio_codec in ("copy", "aac"):
        command = resize_command(ffmpeg, clip, out_path, width, height, crf, audio_codec)
        try:
            result = run_cancellable(command, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS, is_cancelled=is_cancelled)
        except (OSError, subprocess.SubprocessError) as error:
            raise VideoResizeError(f"ffmpeg could not resize the video: {type(error).__name__}") from error
        if result.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
            return out_path
        stderr = (result.stderr or b"")[-400:]
        logger.warning(f"[VIDEO_RESIZE] ffmpeg failed with audio {audio_codec}: {stderr!r}")
    raise VideoResizeError(f"ffmpeg could not resize the video: {stderr!r}")


class VideoResizePipe(BasePipe):
    name = "video_resize"
    description = "Make a video smaller, keeping its shape, frame rate and sound"

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec(
                name="size",
                param_type=str,
                default="",
                description="Box the video is scaled to fit inside, as WIDTHxHEIGHT; never larger than the source",
                required=True,
            ),
            PipeConfigSpec(
                name="quality",
                param_type=str,
                default="balanced",
                description="H.264 quality: high keeps more detail, small makes the smallest file",
                required=False,
                choices=list(QUALITY_CRF),
            ),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [PipeInputSpec("video", IOType.VIDEO, True, "Videos to resize", is_array=True)]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("video", IOType.VIDEO, "Resized videos, one per input", is_array=True)]

    def process(
        self, pipe_input: PipeInput, generation_outputs: callable, is_cancelled: Optional[callable] = None,
    ) -> PipeOutput:
        videos = video_paths(pipe_input.input.get("video"))
        if not videos:
            raise ValueError("video_resize needs at least one video")
        requested = parse_size(self.config.get("size"))
        quality = str(self.config.get("quality") or "balanced")
        if quality not in QUALITY_CRF:
            raise ValueError(f"Unknown quality {quality!r}; choose one of {', '.join(QUALITY_CRF)}")
        resized: List[str] = []
        for video in videos:
            src_w, src_h, _, _ = probe_video_stream(video)
            width, height, clamped = fit_size((src_w, src_h), requested)
            if clamped:
                generation_outputs(ProgressGenerationOutput(
                    state=f"The video is {src_w}x{src_h}, smaller than {requested[0]}x{requested[1]}, so it keeps its size",
                    icon=Icon("alert-triangle", "beat"),
                ))
            generation_outputs(ProgressGenerationOutput(
                state=f"Resizing the video to {width}x{height}", icon=Icon("film", "pulse"),
            ))
            with tempfile.NamedTemporaryFile(prefix="potionui-resize-", suffix=".mp4", delete=False) as handle:
                out_path = Path(handle.name)
            try:
                resize_video(video, out_path, width, height, QUALITY_CRF[quality], is_cancelled=is_cancelled)
            except VideoConcatCancelled:
                out_path.unlink(missing_ok=True)
                break
            except Exception:
                out_path.unlink(missing_ok=True)
                raise
            logger.info(f"[VIDEO_RESIZE] {video} {src_w}x{src_h} -> {width}x{height}")
            resized.append(str(out_path))
        return PipeOutput(output={"video": resized})
