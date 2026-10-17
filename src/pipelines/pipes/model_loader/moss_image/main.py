from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipelines.outputs import (
    ModelGenerationOutput,
    ModelsGenerationOutput,
)
from src.platform.runtime.model_lifecycle.lifecycle import file_size_gb
from src.platform.runtime.native.engine import NativeEngineLoader, NativeModel
from src.pipelines.contracts import (
    IOType,
    PipeInput,
    PipeInputSpec,
    PipeOutput,
    PipeOutputSpec,
    PipeConfigSpec,
)
from src.pipelines.pipes._shared.generation.loader_base import BaseModelLoaderPipe
from src.pipelines.pipes._shared.generation.loader_helpers import (
    ComponentProgress,
    path_of as _path_of,
    vram_budget as _vram_budget_fn,
)
from src.pipelines.pipes._shared.generation.loader_lifecycle import (
    Component,
    ComponentLifecycle,
)
from src.pipelines.pipes.model_loader.moss_image.bundle import MossImageModelBundle
from src.pipelines.pipes.model_loader.moss_image.moss_image_clip import MossImageClipTextEncoder


class ModelLoaderMossImagePipe(BaseModelLoaderPipe):
    name = "model_loader"
    description = "Load a native MOSS-Image checkpoint set (DiT + Qwen3.5-9B TE + FLUX 2 AE)"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "diffusion_model": None,
            "text_encoder": None,
            "vae": None,
            "device": "cuda",
            "dtype": "bfloat16",
        }

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("diffusion_model", dict, None, "MOSS-Image DiT (SFT or Base, bf16 or int8 ConvRot)", required=True),
            PipeConfigSpec("text_encoder", dict, None, "Qwen3.5-9B text encoder (int8 ConvRot, one prompt per pass)", required=True),
            PipeConfigSpec("vae", dict, None, "FLUX 2 autoencoder (flux2-vae.safetensors)", required=True),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
            PipeConfigSpec("dtype", str, "bfloat16", "Compute dtype", required=False,
                           choices=["bfloat16", "float16", "float32"]),
            PipeConfigSpec("vram_limit_gb", float, None, "VRAM budget hint (backend-injected)", required=False),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("MODELS", IOType.SERVICE, False, "Model lifecycle service for per-component reuse", is_array=False),
            PipeInputSpec("GPU", IOType.SERVICE, False, "GPU manager for the VRAM budget", is_array=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("model", IOType.MODEL, "MOSS-Image model bundle (DiT + TE + VAE)", is_array=False),
            PipeOutputSpec("text_encoder", IOType.TEXT_ENCODER, "Qwen3.5-9B text encoder (ClipTextEncoder ABC)", is_array=False),
        ]

    def progress_message(self) -> str:
        dit_path = _path_of(self.config.get("diffusion_model")) or "?"
        return f"Loading MOSS-Image model <<MODEL:{Path(dit_path).stem}>>"

    def describe_models(self) -> List[ModelGenerationOutput]:
        out: List[ModelGenerationOutput] = []
        for key, mtype in (
            ("diffusion_model", "moss_image_dit"),
            ("text_encoder", "moss_image_text_encoder"),
            ("vae", "moss_image_vae"),
        ):
            cfg = self.config.get(key)
            if _path_of(cfg):
                out.append(ModelGenerationOutput(name=cfg.get("name") or Path(_path_of(cfg)).stem, type=mtype))
        return out

    def process(self, pipe_input: PipeInput, generation_outputs: callable) -> PipeOutput:
        self.validate()
        generation_outputs(ModelsGenerationOutput(models=self.describe_models()))

        dit_path = _path_of(self.config.get("diffusion_model"))
        te_path = _path_of(self.config.get("text_encoder"))
        vae_path = _path_of(self.config.get("vae"))
        if not (dit_path and te_path and vae_path):
            raise ValueError("model_loader/moss_image requires diffusion_model, text_encoder and vae file paths")

        device = self.config.get("device", "cuda")
        dtype = self.config.get("dtype", "bfloat16")

        vram_gb = self._vram_budget(pipe_input)
        loader = NativeEngineLoader(device=device, vram_gb=vram_gb)

        te_fp = f"{te_path}|{dtype}|moss_image"
        vae_fp = f"{vae_path}|{dtype}"
        dit_fp = f"{dit_path}|{dtype}"

        def load_te() -> NativeModel:
            return loader.load(te_path, "text_encoder")

        def load_vae() -> NativeModel:
            return loader.load(vae_path, "vae")

        def load_dit() -> NativeModel:
            return loader.load(dit_path, "diffusion_model")

        models = pipe_input.input.get("MODELS", None)
        progress = ComponentProgress(generation_outputs, models, self.progress_message(), total=3)
        lifecycle = ComponentLifecycle(models, progress)

        te_key = f"native/te/{te_path}|moss_image"
        te = Component("text encoder", te_key, te_fp, load_te, file_size_gb(te_path))
        vae = Component("VAE", f"native/vae/{vae_path}", vae_fp, load_vae, file_size_gb(vae_path))
        dit = Component("DiT", f"native/dit/{dit_path}", dit_fp, load_dit, file_size_gb(dit_path))

        if not lifecycle.caching:
            te_model = lifecycle.acquire(te)
            vae_model = lifecycle.acquire(vae)
            dit_model = lifecycle.acquire(dit)
            return PipeOutput(output={
                "model": MossImageModelBundle(dit=dit_model, te=te_model, vae=vae_model, te_cache_key=te_key),
                "text_encoder": MossImageClipTextEncoder(
                    te_model.module, device=device, model_fingerprint=f"{te_fp}|{dit_fp}",
                ),
            })

        vae_model = lifecycle.acquire(vae)
        dit_model = lifecycle.acquire(dit)
        bundle = MossImageModelBundle(dit=dit_model, te=None, vae=vae_model, te_cache_key=te_key)
        clip = MossImageClipTextEncoder(
            device=device, model_fingerprint=f"{te_fp}|{dit_fp}",
            te_loader=lifecycle.deferred_module(te),
        )
        return PipeOutput(output={"model": bundle, "text_encoder": clip})

    def _vram_budget(self, pipe_input: PipeInput) -> Optional[float]:
        return _vram_budget_fn(pipe_input, self.config.get("vram_limit_gb", None), "MODEL LOADER MOSS")
