from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

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
from src.pipelines.outputs import GenerationExecutionError
from src.pipelines.pipes._shared.media.inputs import as_flag, media_paths
from src.pipelines.pipes._shared.media.tools import (
    ProgressRelay,
    keep_file,
    resolve_registry,
    scratch_dir,
)
from src.platform.media_tools import (
    InvalidToolParams,
    MediaToolError,
    ToolCancelled,
    ToolContext,
    ToolMedia,
    ToolSpec,
    UnsupportedMedia,
    run_chain,
    validate_params,
)
from src.platform.media_tools.chain import check_accepts, check_requires
from src.platform.media_tools.probe import has_alpha_mode

KINDS = ("image", "video", "audio")
KIND_TYPES = {"image": IOType.IMAGE, "video": IOType.VIDEO, "audio": IOType.AUDIO}
IN_MEMORY_NAME = "in-memory"


def describe_image(image: Image.Image) -> ToolMedia:
    return ToolMedia(
        path=Path(IN_MEMORY_NAME),
        kind="image",
        width=image.width,
        height=image.height,
        has_alpha=has_alpha_mode(image),
    )


def load_image(path) -> Image.Image:
    with Image.open(path) as opened:
        opened.load()
        return opened.copy()


class MediaToolPipe(BasePipe):
    name = "media_tool"
    description = "Run a media tool, built in or from a plugin, on images, videos or audio"

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec(
                name="tool",
                param_type=str,
                default=None,
                description="Id of the media tool to run, such as trim, resize, crop_subject or a plugin's id.tool",
                required=True,
            ),
            PipeConfigSpec(
                name="params",
                param_type=dict,
                default={},
                description="The tool's parameters; anything left out takes the tool's default",
                required=False,
            ),
            PipeConfigSpec(
                name="skip_unfit",
                param_type=bool,
                default=False,
                description=(
                    "Pass media the tool cannot work on (an image without transparency for crop_subject, "
                    "a video without sound for a tool that needs it) through unchanged instead of failing"
                ),
                required=False,
            ),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("image", IOType.IMAGE, False, "Images, for a tool that accepts images", is_array=True),
            PipeInputSpec("video", IOType.VIDEO, False, "Video files, for a tool that accepts video", is_array=True),
            PipeInputSpec("audio", IOType.AUDIO, False, "Audio files, for a tool that accepts audio", is_array=True),
            PipeInputSpec(
                "MEDIA_TOOLS", IOType.SERVICE, False,
                "Media tool registry, so a plugin's tools can run; without it only the built-in tools do",
                is_array=False,
            ),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("image", IOType.IMAGE, "Images the tool produced", is_array=True),
            PipeOutputSpec("video", IOType.VIDEO, "Video files the tool produced", is_array=True),
            PipeOutputSpec("audio", IOType.AUDIO, "Audio files the tool produced", is_array=True),
        ]

    def _spec(self, pipe_input: PipeInput) -> ToolSpec:
        tool_id = str(self.config.get("tool") or "").strip()
        if not tool_id:
            raise ValueError("media_tool needs a 'tool'")
        spec = resolve_registry(pipe_input.input.get("MEDIA_TOOLS")).get(tool_id)
        if spec is None:
            raise ValueError(f"media_tool: there is no media tool '{tool_id}'")
        if spec.impl is None:
            raise ValueError(f"media_tool: '{tool_id}' has nothing to run on the server")
        return spec

    def _params(self, spec: ToolSpec) -> dict:
        given = self.config.get("params")
        try:
            return validate_params(spec, given if isinstance(given, dict) else {})
        except InvalidToolParams as error:
            raise GenerationExecutionError(f"{spec.label}: {error.message}") from error

    @staticmethod
    def _inputs(pipe_input: PipeInput, spec: ToolSpec) -> List[Tuple[str, Any]]:
        items: List[Tuple[str, Any]] = []
        for kind in KINDS:
            raw = pipe_input.input.get(kind)
            if kind == "image":
                values = [item for item in (raw if isinstance(raw, (list, tuple)) else [raw]) if item is not None]
            else:
                values = media_paths(raw)
            if values and kind not in spec.accepts:
                raise GenerationExecutionError(f"{spec.label} cannot be applied to {kind}")
            items.extend((kind, value) for value in values)
        if not items:
            raise ValueError(f"media_tool '{spec.id}' needs {' or '.join(spec.accepts)} input")
        return items

    @staticmethod
    def _media(kind: str, item: Any, ctx: ToolContext) -> ToolMedia:
        if isinstance(item, Image.Image):
            path = ctx.new_path(".png", stem="input")
            item.save(path)
            return ctx.probe(path, "image")
        return ctx.probe(item, kind)

    @staticmethod
    def _collect(results: List[ToolMedia], spec: ToolSpec, produced: Dict[str, list], kept: List[str]) -> None:
        for result in results:
            if result.kind == "image":
                produced["image"].append(load_image(result.path))
                continue
            stored = keep_file(result.path, f"potionui-{spec.id.replace('.', '-')}-")
            kept.append(stored)
            produced[result.kind].append(stored)

    def _run(
        self,
        spec: ToolSpec,
        params: dict,
        items: List[Tuple[str, Any]],
        ctx: ToolContext,
        produced: Dict[str, list],
        kept: List[str],
    ) -> None:
        if spec.cardinality == "many_to_one":
            if len({kind for kind, _ in items}) != 1:
                raise UnsupportedMedia(f"{spec.label} joins media of one kind at a time")
            media = [self._media(kind, item, ctx) for kind, item in items]
            self._collect(run_chain([(spec, params)], media, ctx, validate=False), spec, produced, kept)
            return

        skip_unfit = as_flag(self.config.get("skip_unfit", False))
        total = len(items)
        for index, (kind, item) in enumerate(items):
            stage = ctx.scoped(index / total, (index + 1) / total)
            in_memory = isinstance(item, Image.Image)
            media = describe_image(item) if in_memory else stage.probe(item, kind)
            check_accepts(spec, media)
            try:
                check_requires(spec, media)
            except UnsupportedMedia:
                if not skip_unfit:
                    raise
                produced[kind].append(item)
                stage.progress(1.0, spec.label)
                continue

            if (
                in_memory
                and spec.cardinality == "one_to_one"
                and spec.output_kind("image") == "image"
                and spec.impl.has_image_path()
            ):
                produced["image"].append(spec.impl.image(item, params))
                stage.progress(1.0, spec.label)
                continue

            source = self._media(kind, item, stage) if in_memory else media
            self._collect(run_chain([(spec, params)], [source], stage, validate=False), spec, produced, kept)

    def process(
        self, pipe_input: PipeInput, generation_outputs: callable, is_cancelled: Optional[callable] = None,
    ) -> PipeOutput:
        spec = self._spec(pipe_input)
        params = self._params(spec)
        items = self._inputs(pipe_input, spec)
        relay = ProgressRelay(generation_outputs, spec.label)
        produced: Dict[str, list] = {kind: [] for kind in KINDS}
        kept: List[str] = []
        try:
            with scratch_dir() as scratch:
                self._run(spec, params, items, relay.context(scratch, is_cancelled), produced, kept)
        except ToolCancelled:
            self._discard(kept)
            return PipeOutput(output={kind: [] for kind in KINDS})
        except MediaToolError as error:
            self._discard(kept)
            raise GenerationExecutionError(error.message) from error
        except BaseException:
            self._discard(kept)
            raise
        relay.flush()
        logger.info(f"[MEDIA_TOOL] {spec.id} ran on {len(items)} item(s)")
        return PipeOutput(output=produced)

    @staticmethod
    def _discard(paths: List[str]) -> None:
        for path in paths:
            Path(path).unlink(missing_ok=True)
