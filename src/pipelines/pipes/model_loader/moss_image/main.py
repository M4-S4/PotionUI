from __future__ import annotations

from typing import Any, Dict, List

from src.pipelines.contracts import (
    IOType,
    PipeOutputSpec,
    PipeConfigSpec,
)
from src.pipelines.pipes._shared.generation.three_component_loader import ThreeComponentLoaderPipe
from src.pipelines.pipes.model_loader.moss_image.moss_image_clip import MossImageClipTextEncoder


_LOG_TAG = "MODEL LOADER MOSS"


class ModelLoaderMossImagePipe(ThreeComponentLoaderPipe):
    name = "model_loader"
    description = "Load a native MOSS-Image checkpoint set (DiT + Qwen3.5-9B TE + FLUX 2 AE)"
    log_tag = _LOG_TAG
    family = "moss_image"
    display_name = "MOSS-Image"
    component_labels = (
        ("diffusion_model", "moss_image_dit"),
        ("text_encoder", "moss_image_text_encoder"),
        ("vae", "moss_image_vae"),
    )
    clip_cls = MossImageClipTextEncoder
    te_suffix = "|moss_image"
    sync_loras = True

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
            PipeConfigSpec("diffusion_model", dict, None, "MOSS-Image DiT (SFT or Base, bf16 or int8 ConvRot)", required=True),
            PipeConfigSpec("text_encoder", dict, None, "Qwen3.5-9B text encoder (int8 ConvRot, one prompt per pass)", required=True),
            PipeConfigSpec("vae", dict, None, "FLUX 2 autoencoder (flux2-vae.safetensors)", required=True),
            PipeConfigSpec("loras", list, [],
                           "LoRA adapters (patched in place on an already-cached DiT)", required=False),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
            PipeConfigSpec("dtype", str, "bfloat16", "Compute dtype", required=False,
                           choices=["bfloat16", "float16", "float32"]),
            PipeConfigSpec("vram_limit_gb", float, None, "VRAM budget hint (backend-injected)", required=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("model", IOType.MODEL, "MOSS-Image model bundle (DiT + TE + VAE)", is_array=False),
            PipeOutputSpec("text_encoder", IOType.TEXT_ENCODER, "Qwen3.5-9B text encoder (ClipTextEncoder ABC)", is_array=False),
        ]
