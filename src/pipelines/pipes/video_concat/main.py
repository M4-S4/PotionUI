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
from src.pipelines.pipes._shared.media.inputs import as_flag, media_paths
from src.pipelines.pipes._shared.media.tools import as_pipe_error, join_video_files
from src.pipelines.pipes._shared.video.frames import probe_video_stream
from src.platform.media_tools import MediaToolError, ToolCancelled


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
                name="fit",
                param_type=str,
                default="pad",
                description="Pad a clip of another size with black bars, or fill the frame and crop the overflow",
                required=False,
                choices=["pad", "cover"],
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
            PipeInputSpec(
                "overlap", IOType.VIDEO, False,
                "The clip each extension repeats at its seam; its length sets the overlap", is_array=True,
            ),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("video", IOType.VIDEO, "Joined videos", is_array=True)]

    def _overlap_seconds(self, overlap_clips: List[str]) -> float:
        if overlap_clips:
            _, _, fps, frames = probe_video_stream(overlap_clips[0])
            return frames / fps if fps > 0 else 0.0
        overlap = int(self.config.get("overlap_frames") or 0)
        fps = float(self.config.get("fps") or 24.0)
        return overlap / fps if overlap > 0 and fps > 0 else 0.0

    def _plan(self, videos: List[str], extension: Optional[str], seconds: float = 0.0):
        if extension is None:
            return videos, None, None, 0
        before = self.config.get("extension_position", "after") == "before"
        cut = self.config.get("overlap", "crossfade") == "cut"
        clips = [extension, *videos] if before else [*videos, extension]
        trims = [(0.0, 0.0)] * len(clips)
        crossfades = [0.0] * (len(clips) - 1)
        if seconds > 0 and cut and before:
            trims[0] = (0.0, seconds)
        elif seconds > 0 and cut:
            trims[-1] = (seconds, 0.0)
        elif seconds > 0:
            crossfades[0 if before else -1] = seconds
        return clips, trims, crossfades, 1 if before else 0

    def process(
        self, pipe_input: PipeInput, generation_outputs: callable, is_cancelled: Optional[callable] = None,
    ) -> PipeOutput:
        videos = media_paths(pipe_input.input.get("video"))
        extensions = media_paths(pipe_input.input.get("extension"))
        if not videos:
            raise ValueError("video_concat needs at least one video")
        audio = as_flag(self.config.get("keep_audio", True))
        seconds = self._overlap_seconds(media_paths(pipe_input.input.get("overlap"))) if extensions else 0.0
        joined: List[str] = []
        for extension in extensions or [None]:
            clips, trims, crossfades, canvas = self._plan(videos, extension, seconds)
            with tempfile.NamedTemporaryFile(prefix="potionui-joined-", suffix=".mp4", delete=False) as handle:
                out_path = Path(handle.name)
            try:
                join_video_files(
                    clips, out_path, emit=generation_outputs, is_cancelled=is_cancelled, trims=trims,
                    crossfades=crossfades, canvas=canvas, fit=self.config.get("fit", "pad"), audio=audio,
                )
            except ToolCancelled:
                out_path.unlink(missing_ok=True)
                break
            except MediaToolError as error:
                out_path.unlink(missing_ok=True)
                raise as_pipe_error(error) from error
            except Exception:
                out_path.unlink(missing_ok=True)
                raise
            logger.info(f"[VIDEO_CONCAT] joined {len(clips)} clip(s) into {out_path}")
            joined.append(str(out_path))
        return PipeOutput(output={"video": joined})
