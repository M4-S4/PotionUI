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
from src.pipelines.pipes._shared.media.video_concat import (
    ClipTrim,
    VideoConcatCancelled,
    as_flag,
    concat_videos,
    video_paths,
)


class VideoConcatPipe(BasePipe):
    name = "video_concat"
    description = "Join video clips end to end, keeping their sound"

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec(
                name="extension_position",
                param_type=str,
                default="after",
                description="Where each extension clip joins the video",
                required=False,
                choices=["after", "before"],
            ),
            PipeConfigSpec(
                name="overlap_frames",
                param_type=int,
                default=0,
                description="Frames at the seam of each extension clip that repeat the video",
                required=False,
                min_value=0,
            ),
            PipeConfigSpec(
                name="overlap",
                param_type=str,
                default="crossfade",
                description="Blend the repeated frames into the video, or cut them",
                required=False,
                choices=["crossfade", "cut"],
            ),
            PipeConfigSpec(
                name="fps",
                param_type=float,
                default=24.0,
                description="Frame rate of the extension clips, to turn overlap frames into time",
                required=False,
            ),
            PipeConfigSpec(
                name="keep_audio",
                param_type=bool,
                default=True,
                description="Keep the clips' sound",
                required=False,
            ),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("video", IOType.VIDEO, True, "Clips joined in order", is_array=True),
            PipeInputSpec(
                "extension", IOType.VIDEO, False,
                "Clips joined to the video one at a time; each makes its own joined video", is_array=True,
            ),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("video", IOType.VIDEO, "Joined videos", is_array=True)]

    def _plan(self, videos: List[str], extension: Optional[str]):
        if extension is None:
            return videos, None, None, 0
        overlap = int(self.config.get("overlap_frames") or 0)
        fps = float(self.config.get("fps") or 24.0)
        seconds = overlap / fps if overlap > 0 and fps > 0 else 0.0
        before = self.config.get("extension_position", "after") == "before"
        cut = self.config.get("overlap", "crossfade") == "cut"
        clips = [extension, *videos] if before else [*videos, extension]
        trims = [ClipTrim()] * len(clips)
        crossfades = [0.0] * (len(clips) - 1)
        if seconds > 0 and cut and before:
            trims[0] = ClipTrim(end=seconds)
        elif seconds > 0 and cut:
            trims[-1] = ClipTrim(start=seconds)
        elif seconds > 0:
            crossfades[0 if before else -1] = seconds
        return clips, trims, crossfades, 1 if before else 0

    def process(
        self, pipe_input: PipeInput, generation_outputs: callable, is_cancelled: Optional[callable] = None,
    ) -> PipeOutput:
        videos = video_paths(pipe_input.input.get("video"))
        extensions = video_paths(pipe_input.input.get("extension"))
        if not videos:
            raise ValueError("video_concat needs at least one video")
        audio = as_flag(self.config.get("keep_audio", True))
        joined: List[str] = []
        for extension in extensions or [None]:
            clips, trims, crossfades, canvas = self._plan(videos, extension)
            generation_outputs(ProgressGenerationOutput(state="Joining the clips", icon=Icon("film", "pulse")))
            with tempfile.NamedTemporaryFile(prefix="potionui-joined-", suffix=".mp4", delete=False) as handle:
                out_path = Path(handle.name)
            try:
                concat_videos(
                    clips, out_path, trims=trims, crossfades=crossfades, audio=audio, canvas=canvas,
                    is_cancelled=is_cancelled,
                    notify=lambda text: generation_outputs(
                        ProgressGenerationOutput(state=text, icon=Icon("alert-triangle", "beat"))),
                )
            except VideoConcatCancelled:
                out_path.unlink(missing_ok=True)
                break
            except Exception:
                out_path.unlink(missing_ok=True)
                raise
            logger.info(f"[VIDEO_CONCAT] joined {len(clips)} clip(s) into {out_path}")
            joined.append(str(out_path))
        return PipeOutput(output={"video": joined})
