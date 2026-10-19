from __future__ import annotations

import os
import tempfile
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
from src.pipelines.outputs import (
    GalleryGenerationOutput,
    Icon,
    Progress,
    ProgressGenerationOutput,
    SplatGenerationOutput,
    TextGenerationOutput,
)
from src.platform.runtime.native.arch.image_to_splat import (
    DEFAULT_DETAIL,
    DETAILS,
    STAGE_ESTIMATE,
    STAGE_LIFT,
    LiftedSplat,
    render_splat_preview,
    run_image_to_splat,
    write_splat_ply,
)
from src.platform.runtime.native.errors import SamplingCancelled

_STAGE_LABELS = {
    STAGE_ESTIMATE: "Estimating depth with MoGe-2",
    STAGE_LIFT: "Building the 3D scene",
}
_WRITE_LABEL = "Writing the splat"


class GeneratorImageToSplatPipe(BasePipe):
    name = "generator"
    description = "Native Image to Splat generator (one image to a two-layer 3D Gaussian splat .ply, MoGe-2 depth)"

    _family_pipe = "generator/image_to_splat"
    _loader_pipe = "model_loader/image_to_splat"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"detail": DEFAULT_DETAIL, "device": "cuda"}

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("detail", str, DEFAULT_DETAIL,
                           "Inference resolution: standard runs MoGe-2 and the lift at 1024 px on the long "
                           "side, high at 1536 px. The image is never enlarged, and the Gaussian count "
                           "follows the pixel count (about 1.1 to 1.4 per pixel with the background layer).",
                           required=False, choices=list(DETAILS)),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("model", IOType.MODEL, True, "Image to Splat model bundle", is_array=False),
            PipeInputSpec("image", IOType.IMAGE, True, "Source image(s), one splat each", is_array=True),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("splat", IOType.SPLAT, "Paths to the generated 3DGS .ply files", is_array=True)]

    def process(
        self,
        pipe_input: PipeInput,
        generation_outputs: callable,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> PipeOutput:
        is_cancelled = is_cancelled or (lambda: False)
        images = list(pipe_input.input.get("image") or [])
        if not images:
            raise ValueError(f"{self._family_pipe} needs a source image, but none was provided")
        bundle = pipe_input.input.get("model")
        if bundle is None:
            raise ValueError(f"{self._family_pipe} needs the model bundle from {self._loader_pipe}")

        estimator = bundle.components()
        device = self.config.get("device") or getattr(bundle, "device", "cuda")
        detail = self._detail()
        paths: List[str] = []
        splats: List[SplatGenerationOutput] = []
        try:
            for index, image in enumerate(images):
                if is_cancelled():
                    raise SamplingCancelled()
                lifted = run_image_to_splat(
                    estimator, image, detail=detail, device=device,
                    on_stage=self._stage_reporter(generation_outputs, index, len(images)),
                    is_cancelled=is_cancelled,
                )
                if is_cancelled():
                    raise SamplingCancelled()
                generation_outputs(ProgressGenerationOutput(
                    state=self._label(_WRITE_LABEL, index, len(images)),
                    icon=Icon(name="cube", effect="pulse"), progress=Progress(index, len(images)),
                ))
                path = self._write(lifted)
                paths.append(path)
                splats.append(SplatGenerationOutput(
                    splat_path=Path(path),
                    temporary=False,
                    splat_count=lifted.splats.count,
                    preview=render_splat_preview(lifted.splats, lifted.view),
                ))
                generation_outputs(TextGenerationOutput(title="3D scene", text=self._summary(lifted), index=index))
            if is_cancelled():
                raise SamplingCancelled()
        except BaseException:
            self._discard(paths)
            raise

        generation_outputs(GalleryGenerationOutput(images=[], splats=splats))
        generation_outputs(ProgressGenerationOutput(
            state="Built the 3D scene" if len(splats) == 1 else f"Built <<NUMBER:{len(splats)}>> 3D scenes",
            icon=Icon(name="check"),
        ))
        return PipeOutput(output={"splat": paths})

    def _detail(self) -> str:
        detail = str(self.config.get("detail") or DEFAULT_DETAIL)
        if detail not in DETAILS:
            raise ValueError(f"unknown detail {detail!r}; expected one of {list(DETAILS)}")
        return detail

    @staticmethod
    def _label(label: str, index: int, total: int) -> str:
        if total > 1:
            return f"{label} (<<NUMBER:{index + 1}>> of <<NUMBER:{total}>>)"
        return label

    def _stage_reporter(self, generation_outputs: callable, index: int, total: int) -> Callable[[str], None]:
        def report(stage: str) -> None:
            generation_outputs(ProgressGenerationOutput(
                state=self._label(_STAGE_LABELS.get(stage, stage), index, total),
                icon=Icon(name="cube", effect="pulse"),
                progress=Progress(index, total),
            ))

        return report

    @staticmethod
    def _summary(lifted: LiftedSplat) -> str:
        view = lifted.view
        return (
            f"{lifted.splats.count:,} Gaussians ({lifted.front_count:,} visible, {lifted.back_count:,} behind "
            f"the foreground) from a {view.width}x{view.height} image, {view.fov_y:.1f}° vertical field of view"
        )

    @staticmethod
    def _write(lifted: LiftedSplat) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        handle.close()
        try:
            write_splat_ply(handle.name, lifted.splats, lifted.view)
        except BaseException:
            os.unlink(handle.name)
            raise
        return handle.name

    @staticmethod
    def _discard(paths: List[str]) -> None:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass
