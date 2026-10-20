from __future__ import annotations

from typing import Any, Dict, List

from src.pipelines.contracts import (
    IOType,
    PipeOutputSpec,
    PipeConfigSpec,
)
from src.pipelines.pipes._shared.generation.three_component_loader import ThreeComponentLoaderPipe
from src.pipelines.pipes.model_loader.flux3.flux3_clip import Flux3ClipTextEncoder


class ModelLoaderFlux3Pipe(ThreeComponentLoaderPipe):
    name = "model_loader"
    description = "Load a native FLUX 3 Image Klein checkpoint set (DiT + Qwen3-VL-4B TE + FLUX 2 AE)"
    log_tag = "MODEL LOADER FLUX 3"
    family = "flux3"
    display_name = "FLUX 3"
    component_labels = (
        ("diffusion_model", "flux3_dit"),
        ("text_encoder", "flux3_text_encoder"),
        ("vae", "flux3_vae"),
    )
    clip_cls = Flux3ClipTextEncoder
    te_suffix = "|flux3"
    te_load_kwargs = {"te_variant": "flux3"}

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "diffusion_model": None,
            "text_encoder": None,
            "vae": None,
            "loras": [],
            "device": "cuda",
            "dtype": "bfloat16",
        }

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("diffusion_model", dict, None, "FLUX 3 Image Klein DiT (distilled flow-map or base)", required=True),
            PipeConfigSpec("text_encoder", dict, None, "Qwen3-VL-4B text encoder (4 hidden layers, one prompt per pass)", required=True),
            PipeConfigSpec("vae", dict, None, "FLUX 3 autoencoder (ae.safetensors)", required=True),
            PipeConfigSpec("loras", list, [], "LoRA adapters (busts only the DiT cache)", required=False),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
            PipeConfigSpec("dtype", str, "bfloat16", "Compute dtype", required=False,
                           choices=["bfloat16", "float16", "float32"]),
            PipeConfigSpec("vram_limit_gb", float, None, "VRAM budget hint (backend-injected)", required=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("model", IOType.MODEL, "FLUX 3 model bundle (DiT + TE + VAE)", is_array=False),
            PipeOutputSpec("text_encoder", IOType.TEXT_ENCODER, "Qwen3-VL-4B text encoder (ClipTextEncoder ABC)", is_array=False),
        ]
