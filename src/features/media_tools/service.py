import asyncio
import io
import logging
import shutil
import tempfile
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

from PIL import Image
from pydantic import ValidationError

from src.features.media_tools.dto import (
    TERMINAL_STATUSES,
    InputRef,
    OutputSpec,
    RunDTO,
    RunError,
    RunResult,
    StepRequest,
    ToolDTO,
)
from src.features.media_tools.jobs import MediaToolJobManager, RunControl, RunOutcome
from src.features.media_tools.publisher import MediaToolPublisher, to_item
from src.features.media_tools.repository import MediaToolParamsRepository
from src.features.media_tools.sources import InputResolver, ResolvedInput
from src.platform.http.errors import InvalidInputError, NotFoundError
from src.platform.media_tools import (
    InvalidSelection,
    InvalidToolParams,
    MediaToolError,
    MediaToolRegistry,
    ToolCancelled,
    ToolContext,
    ToolFailed,
    ToolSpec,
    UnsupportedMedia,
    run_chain,
    validate_params,
)
from src.platform.media_tools.fastpath import prepare_image
from src.platform.media_tools.probe import has_alpha_mode

if TYPE_CHECKING:
    from src.features.generation.file_repository import FileRepository
    from src.features.media.media_types import MediaTypeResolver
    from src.features.media.upload_repository import UploadRepository
    from src.features.plans.guard import LimitGuard
    from src.platform.filesystem.storage_driver import FileStorageDriver

logger = logging.getLogger(__name__)

MAX_STEPS = 8
MAX_INPUTS = 50
PREVIEW_SOURCE_SIDE = 1024
PREVIEW_OUTPUT_SIDE = 512
PREVIEW_QUALITY = 80
DEFAULT_SYNC_TIMEOUT_SECONDS = 600.0
SAVE_PROGRESS = 0.96
CHAIN_PROGRESS_SHARE = 0.95

StepLike = Union[StepRequest, Mapping[str, Any]]
InputLike = Union[InputRef, Mapping[str, Any]]
OutputLike = Union[OutputSpec, Mapping[str, Any], None]


@dataclass
class _Plan:
    steps: List[Tuple[ToolSpec, dict]]
    inputs: List[ResolvedInput]
    mode: str


def _user_id(user: Any) -> str:
    return str(getattr(user, "id", user))


def _model(cls, value, field: str):
    if isinstance(value, cls):
        return value
    try:
        return cls.model_validate(value)
    except ValidationError as exc:
        raise InvalidInputError("invalid_params", f"Malformed {field}", field=field) from exc


def _refusal(exc: MediaToolError, **extra: Any) -> InvalidInputError:
    if isinstance(exc, InvalidToolParams):
        details = {"field": exc.field, **extra}
        if exc.step_index is not None:
            details["step_index"] = exc.step_index
        return InvalidInputError("invalid_params", exc.message, **details)
    return InvalidInputError(exc.code, exc.message, **extra)


class MediaToolService:

    def __init__(
        self,
        registry: MediaToolRegistry,
        params_repository: MediaToolParamsRepository,
        upload_repository: "UploadRepository",
        file_repository: "FileRepository",
        media_type_resolver: "MediaTypeResolver",
        storage_driver: "FileStorageDriver",
        limit_guard: Optional["LimitGuard"] = None,
        jobs: Optional[MediaToolJobManager] = None,
        scratch_root: Optional[Path] = None,
    ):
        self.registry = registry
        self.params_repo = params_repository
        self.resolver = InputResolver(upload_repository, file_repository, storage_driver)
        self.publisher = MediaToolPublisher(
            upload_repository, media_type_resolver, storage_driver, limit_guard
        )
        self.jobs = jobs if jobs is not None else MediaToolJobManager()
        self.scratch_root = Path(scratch_root) if scratch_root else Path(tempfile.gettempdir())

    def catalog(
        self, user: Any, kind: Optional[str] = None, scope: Optional[str] = None
    ) -> List[ToolDTO]:
        specs = self.registry.list(kind=kind or None, scope=scope or None)
        remembered = self.params_repo.get_many(_user_id(user), [spec.id for spec in specs])
        return [ToolDTO(**spec.to_dict(), last_params=remembered.get(spec.id)) for spec in specs]

    def last_params(self, user: Any, tool_id: str) -> Optional[dict]:
        return self.params_repo.get(_user_id(user), tool_id)

    def start_run(
        self,
        user: Any,
        steps: Sequence[StepLike],
        inputs: Sequence[InputLike],
        output: OutputLike = None,
    ) -> str:
        user_id = _user_id(user)
        plan = self._plan(user_id, steps, inputs, output)
        if plan.mode != "replace":
            self.publisher.precheck(user_id)
        return self.jobs.submit(user_id, self._work(user_id, plan))

    def get_run(self, user: Any, run_id: str) -> RunDTO:
        run = self.jobs.get(_user_id(user), run_id)
        if run is None:
            raise NotFoundError("not_found", "Run not found")
        return run

    def cancel_run(self, user: Any, run_id: str) -> RunDTO:
        run = self.jobs.cancel(_user_id(user), run_id)
        if run is None:
            raise NotFoundError("not_found", "Run not found")
        return run

    def run_blocking(
        self,
        user: Any,
        steps: Sequence[StepLike],
        inputs: Sequence[InputLike],
        output: OutputLike = None,
        timeout: float = DEFAULT_SYNC_TIMEOUT_SECONDS,
    ) -> RunResult:
        user_id = _user_id(user)
        run_id = self.start_run(user_id, steps, inputs, output)
        run = self.jobs.wait(user_id, run_id, timeout)
        return self._settle(user_id, run_id, run)

    async def run_sync(
        self,
        user: Any,
        steps: Sequence[StepLike],
        inputs: Sequence[InputLike],
        output: OutputLike = None,
        timeout: float = DEFAULT_SYNC_TIMEOUT_SECONDS,
    ) -> RunResult:
        user_id = _user_id(user)
        starter = asyncio.ensure_future(
            asyncio.to_thread(self.start_run, user_id, steps, inputs, output)
        )
        try:
            run_id = await asyncio.shield(starter)
        except asyncio.CancelledError:
            started = (await asyncio.gather(starter, return_exceptions=True))[0]
            if isinstance(started, str):
                await asyncio.shield(asyncio.to_thread(self.jobs.cancel, user_id, started))
            raise
        try:
            run = await asyncio.to_thread(self.jobs.wait, user_id, run_id, timeout)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.to_thread(self.jobs.cancel, user_id, run_id))
            raise
        return await asyncio.to_thread(self._settle, user_id, run_id, run)

    def preview(
        self,
        user: Any,
        tool: str,
        params: Mapping[str, Any],
        input_ref: InputLike,
        time_seconds: Optional[float] = None,
    ) -> bytes:
        user_id = _user_id(user)
        spec = self._spec(tool)
        if not spec.preview or spec.impl is None or not spec.impl.has_image_path():
            raise InvalidInputError("unsupported_media", f"{spec.label} has no preview")
        resolved = self.resolver.resolve(user_id, _model(InputRef, input_ref, "input"))
        if resolved.kind not in spec.accepts or resolved.kind == "audio":
            raise InvalidInputError(
                "unsupported_media", f"{spec.label} cannot be applied to {resolved.kind}"
            )
        try:
            validated = validate_params(spec, dict(params or {}))
        except InvalidToolParams as exc:
            raise _refusal(exc) from exc

        scratch = self.scratch_root / f"media-tool-{uuid.uuid4().hex}"
        try:
            scratch.mkdir(parents=True)
            with self.resolver.stage(resolved) as path:
                source = self._preview_source(resolved, path, scratch, time_seconds)
            source.thumbnail((PREVIEW_SOURCE_SIDE, PREVIEW_SOURCE_SIDE), Image.Resampling.LANCZOS)
            try:
                result = spec.impl.image(source, validated)
            except (InvalidToolParams, UnsupportedMedia, InvalidSelection) as exc:
                raise _refusal(exc) from exc
            result.thumbnail((PREVIEW_OUTPUT_SIDE, PREVIEW_OUTPUT_SIDE), Image.Resampling.LANCZOS)
            if result.mode not in ("RGB", "RGBA"):
                result = result.convert("RGBA" if has_alpha_mode(result) else "RGB")
            buffer = io.BytesIO()
            result.save(buffer, format="WEBP", quality=PREVIEW_QUALITY)
            return buffer.getvalue()
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _preview_source(
        self, resolved: ResolvedInput, path: Path, scratch: Path, time_seconds: Optional[float]
    ) -> Image.Image:
        if resolved.kind == "image":
            return prepare_image(path)
        ctx = ToolContext(scratch)
        frame = ctx.new_path(".png", stem="frame")
        at = max(0.0, float(time_seconds or 0.0))
        ctx.ffmpeg(["-ss", f"{at:.3f}", "-i", str(path), "-frames:v", "1", str(frame)])
        if not frame.is_file():
            raise ToolFailed("Could not read a frame from the video")
        return prepare_image(frame)

    def _settle(self, user_id: str, run_id: str, run: Optional[RunDTO]) -> RunResult:
        if run is None:
            return RunResult(
                status="failed", run_id=run_id, error=RunError(code="not_found", message="Run not found")
            )
        if run.status not in TERMINAL_STATUSES:
            run = self.jobs.cancel(user_id, run_id) or run
            return RunResult(
                status="cancelled",
                run_id=run_id,
                error=RunError(code="timeout", message="The tool took too long and was cancelled"),
            )
        return RunResult(
            items=list(run.items),
            replaced=run.replaced,
            error=run.error,
            status=run.status,
            run_id=run_id,
        )

    def _spec(self, tool: str, index: Optional[int] = None) -> ToolSpec:
        spec = self.registry.get(tool)
        if spec is None or spec.impl is None:
            extra = {} if index is None else {"step_index": index}
            raise InvalidInputError("unknown_tool", f"Unknown tool: {tool}", **extra)
        return spec

    def _plan(
        self, user_id: str, steps: Sequence[StepLike], inputs: Sequence[InputLike], output: OutputLike
    ) -> _Plan:
        step_models = [_model(StepRequest, step, "steps") for step in steps or []]
        if not step_models or len(step_models) > MAX_STEPS:
            raise InvalidInputError(
                "invalid_params", f"A run needs between 1 and {MAX_STEPS} steps", field="steps"
            )
        specs = [self._spec(step.tool, index) for index, step in enumerate(step_models)]

        refs = [_model(InputRef, ref, "inputs") for ref in inputs or []]
        if not refs or len(refs) > MAX_INPUTS:
            raise InvalidInputError(
                "invalid_selection", f"Pick between 1 and {MAX_INPUTS} items to run a tool on"
            )
        resolved = [self.resolver.resolve(user_id, ref) for ref in refs]

        self._check_media(specs, resolved)
        validated = self._validate_steps(specs, step_models)

        mode = _model(OutputSpec, output if output is not None else {}, "output").mode
        if mode == "replace":
            self._check_replace(specs, resolved)
        return _Plan(steps=list(zip(specs, validated)), inputs=resolved, mode=mode)

    def _check_media(self, specs: List[ToolSpec], resolved: List[ResolvedInput]) -> None:
        kinds = [item.kind for item in resolved]
        count_known = True
        for index, spec in enumerate(specs):
            for kind in kinds:
                if kind not in spec.accepts:
                    raise InvalidInputError(
                        "unsupported_media",
                        f"{spec.label} cannot be applied to {kind}",
                        step_index=index,
                    )
            if count_known:
                low, high = spec.selection_bounds()
                if len(kinds) < low or (high is not None and len(kinds) > high):
                    span = f"{low}" if high == low else (f"{low} to {high}" if high else f"at least {low}")
                    raise InvalidInputError(
                        "invalid_selection",
                        f"{spec.label} needs {span} item(s), got {len(kinds)}",
                        step_index=index,
                    )
            if spec.cardinality == "many_to_one":
                kinds = [spec.output_kind(kinds[0])]
            else:
                kinds = [spec.output_kind(kind) for kind in kinds]
                if spec.cardinality == "one_to_many":
                    count_known = False

    def _validate_steps(self, specs: List[ToolSpec], steps: List[StepRequest]) -> List[dict]:
        validated: List[dict] = []
        for index, (spec, step) in enumerate(zip(specs, steps)):
            try:
                validated.append(validate_params(spec, dict(step.params)))
            except InvalidToolParams as exc:
                raise _refusal(exc, step_index=index) from exc
        return validated

    def _check_replace(self, specs: List[ToolSpec], resolved: List[ResolvedInput]) -> None:
        kind = resolved[0].kind
        for spec in specs:
            kind = spec.output_kind(kind)
        allowed = (
            len(resolved) == 1
            and resolved[0].is_library_item
            and all(spec.cardinality == "one_to_one" for spec in specs)
            and kind == resolved[0].kind
        )
        if not allowed:
            raise InvalidInputError(
                "replace_not_allowed",
                "Replace works on a single library item and a tool that keeps its kind",
            )

    def _work(self, user_id: str, plan: _Plan) -> Callable[[RunControl], RunOutcome]:
        def work(control: RunControl) -> RunOutcome:
            scratch = self.scratch_root / f"media-tool-{uuid.uuid4().hex}"
            try:
                scratch.mkdir(parents=True)
                base = ToolContext(
                    scratch, progress=control.report, is_cancelled=control.is_cancelled
                )
                ctx = base.scoped(0.0, CHAIN_PROGRESS_SHARE)
                with ExitStack() as stack:
                    paths = [stack.enter_context(self.resolver.stage(item)) for item in plan.inputs]
                    media = [ctx.probe(path, item.kind) for path, item in zip(paths, plan.inputs)]
                    outputs = run_chain(plan.steps, media, ctx, validate=False)
                    if control.is_cancelled():
                        raise ToolCancelled()
                    control.report(SAVE_PROGRESS, "Saving")
                    outcome = self._publish(user_id, plan, outputs)
                self._remember(user_id, plan)
                return outcome
            finally:
                shutil.rmtree(scratch, ignore_errors=True)

        return work

    def _publish(self, user_id: str, plan: _Plan, outputs: list) -> RunOutcome:
        if plan.mode == "replace":
            if len(outputs) != 1:
                raise ToolFailed("The tool did not produce exactly one result")
            source = plan.inputs[0]
            updated = self.publisher.publish_replace(user_id, source.upload, source.key, outputs[0])
            return RunOutcome(items=[to_item(updated)], replaced=True)
        rows = self.publisher.publish_new(user_id, outputs, self._names(plan, outputs))
        return RunOutcome(items=[to_item(row) for row in rows], replaced=False)

    def _names(self, plan: _Plan, outputs: list) -> List[Optional[str]]:
        last = plan.steps[-1][0]
        total = len(outputs)
        names: List[Optional[str]] = []
        for index, media in enumerate(outputs):
            matched = last.cardinality == "one_to_one" and len(plan.inputs) == total
            source = plan.inputs[index] if matched else plan.inputs[0]
            stem = Path(source.original_filename or "").stem or "media"
            extension = media.path.suffix
            if last.cardinality == "one_to_many":
                names.append(f"{stem} — part {index + 1}/{total}{extension}")
            elif last.cardinality == "many_to_one":
                names.append(f"{stem}{last.name_suffix or '-joined'}{extension}")
            else:
                names.append(f"{stem}{last.name_suffix}{extension}")
        return names

    def _remember(self, user_id: str, plan: _Plan) -> None:
        if len(plan.steps) != 1:
            return
        spec, params = plan.steps[0]
        try:
            self.params_repo.save(user_id, spec.id, params)
        except Exception:
            logger.exception("Could not remember the parameters of %s", spec.id)
