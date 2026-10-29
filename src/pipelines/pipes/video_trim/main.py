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
from src.pipelines.pipes._shared.media.inputs import media_paths
from src.pipelines.pipes._shared.media.tools import as_pipe_error, keep_file, tool_session
from src.platform.media_tools import (
    FilterFragment,
    MediaTool,
    MediaToolError,
    ToolCancelled,
    ToolFailed,
    ToolMedia,
    ToolSpec,
    run_chain,
)
from src.platform.media_tools.ops.trim import DEFAULT_FPS, head_tail_frame_count

KEPT_CRF = 12
WINDOW_PAD_SECONDS = 1.0


class HeadTail(MediaTool):
    def video_filter(self, media: ToolMedia, params: dict) -> FilterFragment:
        if media.duration is None:
            raise ToolFailed("Could not read the media's duration")
        keep = params["keep"]
        fps = float(params.get("fps") or 0.0)
        seconds = float(params.get("seconds") or 0.0) or None
        rate = fps if fps > 0 else float(media.fps or DEFAULT_FPS)
        count = head_tail_frame_count(int(params.get("frames") or 1), seconds, rate, int(params.get("frame_step") or 1))
        kept = count / rate
        reach = kept + 2 / rate + WINDOW_PAD_SECONDS
        duration = float(media.duration)
        start, end = (0.0, min(duration, reach)) if keep == "head" else (max(0.0, duration - reach), duration)
        resample = [f"fps={fps:g}"] if fps > 0 else []
        if keep == "head":
            video = [f"trim=end_frame={count}"]
            sound = [f"atrim=end={kept:.6f}", "asetpts=PTS-STARTPTS"]
        else:
            video = ["reverse", f"trim=end_frame={count}", "reverse"]
            sound = ["areverse", f"atrim=end={kept:.6f}", "areverse", "asetpts=PTS-STARTPTS"]
        return FilterFragment(
            start=start,
            end=end,
            duration=min(kept, end - start),
            vf=[*resample, *video, "setpts=PTS-STARTPTS", f"fps={rate:g}"],
            af=sound,
            drop_audio=not params.get("keep_audio", True),
            crf=KEPT_CRF,
        )


HEAD_TAIL = ToolSpec(
    id="head_tail",
    label="Keep the head or tail",
    accepts=("video",),
    produces="same",
    cardinality="one_to_one",
    params=[
        {
            "name": "keep", "type": "select", "default": "tail",
            "options": [{"label": "Head", "value": "head"}, {"label": "Tail", "value": "tail"}],
        },
        {"name": "frames", "type": "integer", "default": 25, "min": 1},
        {"name": "seconds", "type": "float", "default": 0.0, "min": 0.0},
        {"name": "frame_step", "type": "integer", "default": 1, "min": 1},
        {"name": "fps", "type": "float", "default": 0.0, "min": 0.0},
        {"name": "keep_audio", "type": "checkbox", "default": True},
    ],
    impl=HeadTail(),
)


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
                name="seconds",
                param_type=float,
                default=0.0,
                description="When above 0, keep this many seconds instead of a frame count",
                required=False,
                min_value=0.0,
            ),
            PipeConfigSpec(
                name="frame_step",
                param_type=int,
                default=1,
                description="Round the kept frame count down to 1 plus a multiple of this step",
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
        videos = media_paths(pipe_input.input.get("video"))
        if not videos:
            raise ValueError("video_trim needs at least one video")
        params = {spec.name: self.config.get(spec.name) for spec in self.configuration()}
        keep = str(params["keep"])
        label = f"Taking the {'end' if keep == 'tail' else 'start'} of the video"
        trimmed: List[str] = []
        try:
            with tool_session(generation_outputs, label, is_cancelled) as ctx:
                for video in videos:
                    (result,) = run_chain([(HEAD_TAIL, params)], [ctx.probe(video, "video")], ctx)
                    trimmed.append(keep_file(result.path, "potionui-trim-"))
                    logger.info(f"[VIDEO_TRIM] kept the {keep} frames of {video}")
        except ToolCancelled:
            self._discard(trimmed)
            return PipeOutput(output={"video": []})
        except MediaToolError as error:
            self._discard(trimmed)
            raise as_pipe_error(error) from error
        except BaseException:
            self._discard(trimmed)
            raise
        return PipeOutput(output={"video": trimmed})

    @staticmethod
    def _discard(paths: List[str]) -> None:
        for path in paths:
            Path(path).unlink(missing_ok=True)
