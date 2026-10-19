from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.pipelines.contracts import (
    IOType,
    PipeConfigSpec,
    PipeInput,
    PipeInputSpec,
    PipeOutput,
    PipeOutputSpec,
)
from src.pipelines.outputs import ModelGenerationOutput, ModelsGenerationOutput
from src.pipelines.pipes._shared.generation.geometry_estimator import (
    geometry_estimator_fingerprint,
    geometry_estimator_key,
    geometry_estimator_size_gb,
    load_geometry_estimator,
)
from src.pipelines.pipes._shared.generation.loader_base import BaseModelLoaderPipe
from src.pipelines.pipes._shared.generation.loader_helpers import ComponentProgress, path_of
from src.pipelines.pipes._shared.generation.loader_lifecycle import Component, ComponentLifecycle
from src.pipelines.pipes.model_loader.image_to_splat.bundle import ImageToSplatModelBundle

ESTIMATOR_LABEL = "MoGe-2 geometry estimator"


class ModelLoaderImageToSplatPipe(BaseModelLoaderPipe):
    name = "model_loader"
    description = "Load MoGe-2 for Image to Splat (the depth, validity mask and camera of one image)"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {"geometry_estimator": None, "device": "cuda"}

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("geometry_estimator", dict, None,
                           "MoGe-2 checkpoint (moge_2_vitl_normal_fp16.safetensors from Comfy-Org/MoGe, "
                           "geometry_estimation/). It is cached under the same entry as Pixal3D's camera "
                           "estimator, so switching between the two presets does not reload it.",
                           required=True),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("MODELS", IOType.SERVICE, False,
                          "Model lifecycle service for reuse across generations", is_array=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [PipeOutputSpec("model", IOType.MODEL, "Image to Splat model bundle (MoGe-2)", is_array=False)]

    def progress_message(self) -> str:
        path = path_of(self.config.get("geometry_estimator")) or "?"
        return f"Loading MoGe-2 <<MODEL:{Path(path).stem}>>"

    def describe_models(self) -> List[ModelGenerationOutput]:
        component = self.config.get("geometry_estimator")
        path = path_of(component)
        if not path:
            return []
        return [ModelGenerationOutput(name=component.get("name") or Path(path).stem, type="geometry_estimation")]

    def _estimator_path(self) -> str:
        path = path_of(self.config.get("geometry_estimator"))
        if not path:
            raise ValueError(
                "model_loader/image_to_splat needs the MoGe-2 geometry estimator selected: "
                "moge_2_vitl_normal_fp16.safetensors (Comfy-Org/MoGe, geometry_estimation/)."
            )
        return path

    def process(self, pipe_input: PipeInput, generation_outputs: callable) -> PipeOutput:
        path = self._estimator_path()
        generation_outputs(ModelsGenerationOutput(models=self.describe_models()))
        models = pipe_input.input.get("MODELS", None)
        progress = ComponentProgress(generation_outputs, models, self.progress_message(), 1)
        lifecycle = ComponentLifecycle(models, progress)
        estimator = lifecycle.acquire(Component(
            ESTIMATOR_LABEL,
            geometry_estimator_key(path),
            geometry_estimator_fingerprint(path),
            lambda: load_geometry_estimator(path),
            geometry_estimator_size_gb(path),
        ))
        bundle = ImageToSplatModelBundle(estimator=estimator, device=self.config.get("device") or "cuda")
        return PipeOutput(output={"model": bundle})
