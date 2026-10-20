"""Model loader for the native Flux family (Flux1 / Flux2 / Klein / Krea-2).

One pipe handles every Flux variant — the native engine detects the variant from
the checkpoint, so Flux1 (T5-XXL + CLIP-L) and Klein/Flux2 (Qwen3, no CLIP-L)
differ only by which text-encoder file(s) the form supplies.

Unlike the single-``acquire`` ``BaseModelLoaderPipe`` flow, this pipe acquires
each heavy component (text encoder, VAE, DiT) under its OWN ``MODELS`` cache key
(per the plan): a shared T5-XXL is reused across Flux presets.

The DiT's ``MODELS`` fingerprint is LoRA-INDEPENDENT (path + dtype only),
mirroring ``model_loader/krea2``: a LoRA-set change is a cache HIT on the
already-resident DiT, never a fingerprint bust that re-reads the checkpoint
(~24GB for Flux1) from disk. ``_sync_loras`` below reconciles an already-cached
DiT's applied LoRA stack with the requested one on every acquire, in place, via
``lora/apply.py``'s ``apply_loras``/``remove_loras`` -- which already dispatch
per-Linear between an in-place weight patch (fp32/fp16/bf16 storage) and a
runtime delta (fp8/nvfp4 storage, applied fresh each forward without ever
touching the quantized weight), so Flux's fp8-scaled DiT variants patch
correctly with no family-specific handling here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.platform.runtime.model_headers import read_safetensors_header
from src.platform.runtime.model_headers.signatures import FLUX3_SIG
from src.pipelines.contracts import (
    IOType,
    PipeOutputSpec,
    PipeConfigSpec,
)
from src.pipelines.pipes._shared.generation.three_component_loader import ThreeComponentLoaderPipe
from src.pipelines.pipes._shared.generation.loader_helpers import path_of as _path_of
from src.pipelines.pipes._shared.generation.three_component_loader import LoaderContext
from src.platform.runtime.model_lifecycle.lifecycle import file_size_gb
from src.platform.runtime.native.engine import NativeModel
from src.pipelines.pipes.model_loader.flux.flux_clip import FluxClipTextEncoder


def _is_flux3_checkpoint(path: str) -> bool:
    try:
        keys = read_safetensors_header(path)
    except (OSError, ValueError):
        return False
    return any(k.endswith(FLUX3_SIG) for k in keys)


class ModelLoaderFluxPipe(ThreeComponentLoaderPipe):
    name = "model_loader"
    description = "Load a native Flux-family checkpoint set (DiT + text encoder + VAE)"
    log_tag = "MODEL LOADER FLUX"
    family = "flux"
    display_name = "Flux"
    component_labels = (
        ("diffusion_model", "flux_dit"),
        ("text_encoder", "flux_text_encoder"),
        ("clip_l", "flux_clip_l"),
        ("vae", "flux_vae"),
    )
    clip_cls = FluxClipTextEncoder
    sync_loras = True

    # -- declaration -------------------------------------------------------

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "diffusion_model": None,
            "text_encoder": None,
            "clip_l": None,
            "vae": None,
            "loras": [],
            "device": "cuda",
            "dtype": "bfloat16",
        }

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("diffusion_model", dict, None, "Flux DiT checkpoint", required=True),
            PipeConfigSpec("text_encoder", dict, None, "Text encoder (T5-XXL for Flux1, Qwen3 for Klein)", required=True),
            PipeConfigSpec("clip_l", dict, None, "CLIP-L (Flux1 only; omit for Klein/Flux2)", required=False),
            PipeConfigSpec("vae", dict, None, "Flux VAE", required=True),
            PipeConfigSpec("loras", list, [], "LoRA adapters (patched in place on an already-cached DiT)", required=False),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
            PipeConfigSpec("dtype", str, "bfloat16", "Compute dtype", required=False,
                           choices=["bfloat16", "float16", "float32"]),
            PipeConfigSpec("vram_limit_gb", float, None, "VRAM budget hint (backend-injected)", required=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("model", IOType.MODEL, "Flux model bundle (DiT + TE + VAE)", is_array=False),
            PipeOutputSpec("text_encoder", IOType.TEXT_ENCODER, "Flux text encoder (ClipTextEncoder ABC)", is_array=False),
        ]




    def _clip_l_path(self):
        return _path_of(self.config.get("clip_l"))

    def _check_paths(self, dit_path: str) -> None:
        if _is_flux3_checkpoint(dit_path):
            raise ValueError(
                f"'{Path(dit_path).name}' is a FLUX 3 Image Klein checkpoint, which the FLUX loader cannot run: "
                "select the FLUX 3 preset for it"
            )

    def _te_key(self, ctx: LoaderContext) -> str:
        return f"native/te/{ctx.te_path}|{self._clip_l_path() or ''}"

    def _te_fingerprint(self, ctx: LoaderContext) -> str:
        return f"{self._te_key(ctx)}|{ctx.dtype}"

    def _te_size_gb(self, ctx: LoaderContext):
        te_gb = file_size_gb(ctx.te_path)
        clip_l_path = self._clip_l_path()
        if clip_l_path:
            clip_l_gb = file_size_gb(clip_l_path)
            if clip_l_gb is not None:
                te_gb = (te_gb or 0.0) + clip_l_gb
        return te_gb

    def _load_te(self, ctx: LoaderContext) -> NativeModel:
        clip_l_path = self._clip_l_path()
        te_arg: Any = [ctx.te_path, clip_l_path] if clip_l_path else ctx.te_path
        return ctx.loader.load(te_arg, "text_encoder")
