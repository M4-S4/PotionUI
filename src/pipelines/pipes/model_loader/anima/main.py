"""Model loader for the native Anima family (txt2img).

Anima is a single-checkpoint MiniTrainDIT (with an in-model LLMAdapter) plus ONE
text encoder (Qwen3-0.6B) and the Wan-2.1 causal-3D VAE (16ch, shared with
Qwen-Image / Krea-2). The native engine detects the DiT variant from the
checkpoint, so this pipe just supplies the three component files.

Like the Qwen loader, each heavy component (text encoder, VAE, DiT) is acquired
under its OWN ``MODELS`` cache key: a shared Qwen3-0.6B / VAE is reused across
Anima presets, and a LoRA change re-acquires ONLY the DiT (LoRAs are in the DiT
fingerprint, not the TE/VAE ones).
"""

from __future__ import annotations

from typing import List

from src.pipelines.contracts import (
    IOType,
    PipeOutputSpec,
    PipeConfigSpec,
)
from src.pipelines.pipes._shared.generation.three_component_loader import ThreeComponentLoaderPipe
from src.pipelines.pipes.model_loader.anima.anima_clip import AnimaClipTextEncoder
from src.pipelines.pipes._shared.generation.config_specs import loras_config_spec


class ModelLoaderAnimaPipe(ThreeComponentLoaderPipe):
    name = "model_loader"
    description = "Load a native Anima checkpoint set (MiniTrainDIT + Qwen3-0.6B TE + causal-3D VAE)"
    log_tag = "MODEL LOADER ANIMA"
    family = "anima"
    display_name = "Anima"
    component_labels = (
        ("diffusion_model", "anima_dit"),
        ("text_encoder", "anima_text_encoder"),
        ("vae", "anima_vae"),
    )
    clip_cls = AnimaClipTextEncoder

    # -- declaration -------------------------------------------------------

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("diffusion_model", dict, None, "Anima MiniTrainDIT checkpoint", required=True),
            PipeConfigSpec("text_encoder", dict, None, "Qwen3-0.6B text encoder (single TE, no CLIP-L)", required=True),
            PipeConfigSpec("vae", dict, None, "Anima (Wan-2.1 causal-3D) VAE", required=True),
            loras_config_spec("LoRA adapters (busts only the DiT cache)"),
            *cls.runtime_config_specs(),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("model", IOType.MODEL, "Anima model bundle (DiT + TE + VAE)", is_array=False),
            PipeOutputSpec("text_encoder", IOType.TEXT_ENCODER, "Qwen3-0.6B text encoder (ClipTextEncoder ABC)", is_array=False),
        ]



