import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.pipelines.contracts import (
    BasePipe,
    IOType,
    PipeConfigSpec,
    PipeInput,
    PipeInputSpec,
    PipeOutput,
    PipeOutputSpec,
)
from src.pipelines.outputs import Icon, ModelFileGenerationOutput, Progress, ProgressGenerationOutput

from src.pipelines.pipes.model_tools.errors import ModelToolsRefused
from src.pipelines.pipes._shared.generation.config_specs import loras_config_spec

MODES = ["quantize", "merge_into_model", "merge_and_quantize", "merge_loras"]
MODEL_MODES = frozenset({"quantize", "merge_into_model", "merge_and_quantize"})
LORA_MODES = frozenset({"merge_into_model", "merge_and_quantize", "merge_loras"})
MERGE_METHODS = ["concat", "reduce"]
OUTPUT_DTYPES = ["F32", "BF16", "F16"]
ADMIN_MESSAGE = "Model Tools is available to administrators only"
LOCAL_ONLY_MESSAGE = (
    "Model Tools runs only on the local native backend, because it writes into this host's model folders; "
    "remote workers and ComfyUI servers are not supported"
)
MAX_KEEP_PATTERNS = 64
MAX_PATTERN_LENGTH = 200
MIN_LORAS_TO_MERGE = 2
PROGRESS_SCALE = 1000
PHASE_TEXT = {
    "hashing": "Hashing the source files",
    "writing": "Writing <<MODEL:{name}>>",
    "verifying": "Verifying <<MODEL:{name}>>",
}


def _split_patterns(raw: Any) -> List[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        items = raw.replace(",", "\n").splitlines()
    elif isinstance(raw, (list, tuple)):
        items = [str(item) for item in raw]
    else:
        raise ModelToolsRefused("Extra keep patterns must be text")
    patterns = [item.strip() for item in items if item.strip()]
    if len(patterns) > MAX_KEEP_PATTERNS:
        raise ModelToolsRefused(f"At most {MAX_KEEP_PATTERNS} keep patterns are allowed")
    for pattern in patterns:
        if len(pattern) > MAX_PATTERN_LENGTH or "\x00" in pattern:
            raise ModelToolsRefused("A keep pattern is too long or contains an invalid character")
    return patterns


def _lora_rows(raw: Any) -> List[Dict[str, Any]]:
    if raw is None or raw == "":
        return []
    if not isinstance(raw, (list, tuple)):
        raise ModelToolsRefused("The LoRA list must be a list of entries")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            raise ModelToolsRefused("Each LoRA entry must carry a file and a strength")
        rows.append(item)
    return rows


def _strength(row: Dict[str, Any], label: str) -> float:
    try:
        value = float(row.get("strength", row.get("weight", 1.0)))
    except (TypeError, ValueError):
        raise ModelToolsRefused(f"{label}: the strength is not a number") from None
    if not math.isfinite(value) or value == 0.0:
        raise ModelToolsRefused(f"{label}: the strength must be a non-zero number")
    return value


def _existing_ancestor(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


class ModelToolsPipe(BasePipe):
    name = "model_tools"
    description = (
        "Quantizes a model file to scaled fp8, bakes LoRAs into it, or merges LoRAs into one file, "
        "writing the result next to the source. Administrators on the local native backend only."
    )
    display_title = "Processing model files"

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("mode", str, "quantize", "Which tool to run", required=True, choices=MODES),
            PipeConfigSpec("is_admin", bool, False,
                           "Must be true; set from runtime.user.is_admin so the server decides, never the form",
                           required=False),
            PipeConfigSpec("model", str, "", "Absolute path of the source model file (quantize and bake modes)",
                           required=False),
            loras_config_spec("LoRA rows [{model, strength}] (bake and merge modes)"),
            PipeConfigSpec("keep_patterns", str, "",
                           "Extra tensor-name globs kept in full precision (quantize modes), one per line or comma separated",
                           required=False),
            PipeConfigSpec("merge_method", str, "concat",
                           "'concat' stacks the LoRAs exactly; 'reduce' compresses them to target_rank",
                           required=False, choices=MERGE_METHODS),
            PipeConfigSpec("target_rank", int, 32, "Rank of the merged LoRA when reducing, and for LoKr conversion",
                           required=False, min_value=1, max_value=4096),
            PipeConfigSpec("min_energy", float, 0.9,
                           "Refuse when a reduced layer keeps less than this share of the delta energy",
                           required=False, min_value=0.0, max_value=1.0),
            PipeConfigSpec("lora_dtype", str, "F32", "Precision of a merged LoRA file", required=False,
                           choices=OUTPUT_DTYPES),
            PipeConfigSpec("output_name", str, "",
                           "Output file name without folders; empty derives it from the inputs", required=False),
            PipeConfigSpec("overwrite", bool, False, "Replace the output file when it already exists",
                           required=False),
            PipeConfigSpec("family", str, "", "Force the model family instead of detecting it", required=False),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("MODELS", IOType.SERVICE, False,
                          "Model lifecycle service; its model roots locate and contain every path", is_array=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return []

    def process(
        self,
        pipe_input: PipeInput,
        generation_outputs: Callable,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> PipeOutput:
        from src.platform.runtime.native.errors import SamplingCancelled
        from src.platform.runtime.native.ops.model_tools.errors import ModelToolsError, WriteCancelledError

        cancelled = is_cancelled or (lambda: False)
        try:
            self._run(pipe_input, generation_outputs, cancelled)
        except WriteCancelledError as exc:
            raise SamplingCancelled() from exc
        except ModelToolsRefused:
            raise
        except ModelToolsError as exc:
            raise ModelToolsRefused(str(exc)) from exc
        return PipeOutput(output={})

    def _mode(self) -> str:
        mode = str(self.config.get("mode") or "")
        if mode not in MODES:
            raise ModelToolsRefused(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}")
        return mode

    def _run(self, pipe_input: PipeInput, generation_outputs: Callable, cancelled: Callable[[], bool]) -> None:
        if self.config.get("is_admin") is not True:
            raise ModelToolsRefused(ADMIN_MESSAGE)
        roots = getattr(pipe_input.input.get("MODELS"), "roots", None)
        if roots is None:
            raise ModelToolsRefused(LOCAL_ONLY_MESSAGE)
        if cancelled():
            from src.platform.runtime.native.ops.model_tools.errors import WriteCancelledError

            raise WriteCancelledError("cancelled before starting")

        from src.platform.runtime.native.ops.model_tools.lora_io import LoraInput

        from src.pipelines.pipes.model_tools import naming, placement, runner

        mode = self._mode()
        source = None
        if mode in MODEL_MODES:
            source = placement.locate_model_file(roots, self.config.get("model"), "Model", placement.MODEL_TYPES)

        lora_files: List[Any] = []
        lora_inputs: List[LoraInput] = []
        if mode in LORA_MODES:
            rows = _lora_rows(self.config.get("loras"))
            minimum = MIN_LORAS_TO_MERGE if mode == "merge_loras" else 1
            if len(rows) < minimum:
                raise ModelToolsRefused(
                    "Pick at least two LoRAs to merge" if mode == "merge_loras" else "Pick at least one LoRA"
                )
            for index, row in enumerate(rows, start=1):
                label = f"LoRA {index}"
                located = placement.locate_model_file(
                    roots, row.get("model") or row.get("file_path"), label, frozenset({placement.LORA_TYPE})
                )
                lora_files.append(located)
                lora_inputs.append(LoraInput(str(located.path), _strength(row, label)))

        merge_method = str(self.config.get("merge_method") or "concat")
        if merge_method not in MERGE_METHODS:
            raise ModelToolsRefused(f"Unknown merge method {merge_method!r}")
        dtype = str(self.config.get("lora_dtype") or "F32")
        if dtype not in OUTPUT_DTYPES:
            raise ModelToolsRefused(f"Unsupported output precision {dtype!r}")
        min_energy = float(self.config.get("min_energy") if self.config.get("min_energy") is not None else 0.9)
        if not 0.0 <= min_energy <= 1.0:
            raise ModelToolsRefused("The minimum energy must be between 0 and 1")
        rank_raw = self.config.get("target_rank")
        rank = int(rank_raw) if rank_raw not in (None, "") else None
        if rank is not None and rank < 1:
            raise ModelToolsRefused("The target rank must be at least 1")
        overwrite = self.config.get("overwrite") is True

        stem = naming.resolve_stem(
            self.config.get("output_name"),
            mode,
            source.path.name if source else None,
            [f.path.name for f in lora_files],
        )
        name = naming.file_name(stem)
        if mode == "merge_loras":
            destination = placement.destination_in_lora_root(roots, name)
        else:
            destination = placement.destination_beside(roots, source, name)
        protected = [f.path for f in lora_files] + ([source.path] if source else [])
        placement.check_overwrite(destination, protected, overwrite)

        request = runner.JobRequest(
            mode=mode,
            model=source.path if source else None,
            loras=tuple(lora_inputs),
            keep_extra=tuple(_split_patterns(self.config.get("keep_patterns"))) if mode != "merge_loras" else (),
            merge_method=merge_method,
            rank=rank,
            min_energy=min_energy,
            dtype=dtype,
            family=(str(self.config.get("family")).strip() or None) if self.config.get("family") else None,
        )

        notifier = _Notifier(generation_outputs, name)
        job = runner.prepare(request)
        if source is not None:
            job.model_type = source.model_type
        placement.check_free_space(_existing_ancestor(destination.directory), job.planned_bytes)

        digests = runner.hash_sources(_unique(job.sources), cancelled, notifier)
        published = runner.publish(
            job, request, destination.path, digests, notify=notifier, is_cancelled=cancelled, overwrite=overwrite
        )

        generation_outputs(ProgressGenerationOutput(
            state=f"Saved <<MODEL:{name}>>",
            icon=Icon(name="check"),
            progress=Progress(PROGRESS_SCALE, PROGRESS_SCALE),
        ))
        generation_outputs(ModelFileGenerationOutput(
            path=str(published.path),
            kind=job.kind,
            model_type=job.model_type or "diffusion_model",
            file_name=name,
            size_bytes=published.size_bytes,
            source_size_bytes=job.source_size_bytes,
            format=job.format_label,
            summary=published.summary,
        ))


def _unique(paths: List[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


class _Notifier:
    def __init__(self, emit: Callable, name: str, min_interval: float = 0.25) -> None:
        self._emit = emit
        self._name = name
        self._min_interval = min_interval
        self._last_phase: Optional[str] = None
        self._last_value = -1
        self._last_time = 0.0

    def __call__(self, phase: str, current: int, total: int) -> None:
        scaled = int(PROGRESS_SCALE * current / total) if total else 0
        now = time.monotonic()
        changed = phase != self._last_phase
        if not changed and (scaled == self._last_value or now - self._last_time < self._min_interval):
            return
        self._last_phase = phase
        self._last_value = scaled
        self._last_time = now
        self._emit(ProgressGenerationOutput(
            state=PHASE_TEXT[phase].format(name=self._name),
            icon=Icon(name="box"),
            progress=Progress(scaled, PROGRESS_SCALE),
        ))
