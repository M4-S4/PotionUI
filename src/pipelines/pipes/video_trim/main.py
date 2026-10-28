import tempfile
from pathlib import Path
from typing import List, Optional

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
from src.pipelines.pipes._shared.media.video_concat import VideoConcatCancelled, as_flag, trim_video, video_paths


class VideoTrimPipe(BasePipe):
    name = "video_trim"
    description = "Keep the first or last frames of a video"

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec(
                name="keep",
                param_type=str,
                default="tail",
                description="Keep the start (head) or the end (tail) of the video",
                required=False,
                choices=["head", "tail"],
            ),
            PipeConfigSpec(
                name="frames",
                param_type=int,
                default=25,
                description="How many frames to keep",
                required=False,
                min_value=1,
            ),
            PipeConfigSpec(
                name="fps",
                param_type=float,
                default=0.0,
                description="Frame rate to resample to before counting frames; 0 keeps the source rate",
                required=False,
                min_value=0.0,
            ),
            PipeConfigSpec(
                name="keep_audio",
                param_type=bool,
                default=True,
                description="Keep the sound of the kept frames",
                required=False,
            ),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [PipeInputSpec("video", IOType.VIDEO, True, "Videos to trim", is_array=True)]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("video", IOType.VIDEO, "Trimmed videos, one per input", is_array=True)]

    def process(
        self, pipe_input: PipeInput, generation_outputs: callable, is_cancelled: Optional[callable] = None,
    ) -> PipeOutput:
        videos = video_paths(pipe_input.input.get("video"))
        if not videos:
            raise ValueError("video_trim needs at least one video")
        keep = str(self.config.get("keep", "tail"))
        frames = int(self.config.get("frames") or 1)
        fps = float(self.config.get("fps") or 0.0)
        audio = as_flag(self.config.get("keep_audio", True))
        trimmed: List[str] = []
        for video in videos:
            generation_outputs(ProgressGenerationOutput(
                state=f"Taking the {'end' if keep == 'tail' else 'start'} of the video", icon=Icon("scissors"),
            ))
            with tempfile.NamedTemporaryFile(prefix="potionui-trim-", suffix=".mp4", delete=False) as handle:
                out_path = Path(handle.name)
            try:
                trim_video(video, out_path, keep=keep, frames=frames, fps=fps, audio=audio, is_cancelled=is_cancelled)
            except VideoConcatCancelled:
                out_path.unlink(missing_ok=True)
                break
            except Exception:
                out_path.unlink(missing_ok=True)
                raise
            logger.info(f"[VIDEO_TRIM] kept the {keep} {frames} frame(s) of {video}")
            trimmed.append(str(out_path))
        return PipeOutput(output={"video": trimmed})
