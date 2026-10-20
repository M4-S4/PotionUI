from __future__ import annotations

from typing import Any, Dict, List

from src.pipelines.contracts import (
    IOType,
    PipeOutputSpec,
    PipeConfigSpec,
)
from src.pipelines.pipes._shared.generation.three_component_loader import ThreeComponentLoaderPipe
from src.pipelines.pipes._shared.generation.loader_helpers import path_of as _path_of
from src.pipelines.pipes._shared.generation.three_component_loader import LoaderContext
from src.platform.runtime.model_lifecycle.lifecycle import file_size_gb
from src.pipelines.pipes.model_loader.qwen_image21.qwen_image21_clip import QwenImage21ClipTextEncoder


class ModelLoaderQwenImage21Pipe(ThreeComponentLoaderPipe):
    name = "model_loader"
    description = "Load a native Qwen-Image-2.1 checkpoint set (MMDiT + Qwen3-VL-8B TE + RGBA 16x VAE)"
    log_tag = "MODEL LOADER QWEN IMAGE 2.1"
    family = "qwen_image21"
    display_name = "Qwen-Image-2.1"
    component_labels = (
        ("diffusion_model", "qwen_image21_dit"),
        ("text_encoder", "qwen_image21_text_encoder"),
        ("vae", "qwen_image21_vae"),
        ("control_model", "qwen_image21_fun_control"),
    )
    clip_cls = QwenImage21ClipTextEncoder
    te_vision = True

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "diffusion_model": None,
            "text_encoder": None,
            "vae": None,
            "loras": [],
            "device": "cuda",
            "dtype": "bfloat16",
            "vision": False,
            "control_model": None,
        }

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            PipeConfigSpec("diffusion_model", dict, None, "Qwen-Image-2.1 MMDiT checkpoint", required=True),
            PipeConfigSpec("text_encoder", dict, None, "Qwen3-VL-8B text encoder (single TE, no CLIP-L)", required=True),
            PipeConfigSpec("vae", dict, None, "Qwen-Image-2.1 RGBA 16x VAE", required=True),
            PipeConfigSpec("loras", list, [], "LoRA adapters (busts only the DiT cache)", required=False),
            PipeConfigSpec("device", str, "cuda", "Compute device", required=False, choices=["cuda", "cpu"]),
            PipeConfigSpec("dtype", str, "bfloat16", "Compute dtype", required=False,
                           choices=["bfloat16", "float16", "float32"]),
            PipeConfigSpec("vram_limit_gb", float, None, "VRAM budget hint (backend-injected)", required=False),
            PipeConfigSpec("vision", bool, False, "Load the vision tower for image-conditioned editing", required=False),
            PipeConfigSpec("control_model", dict, None, "Fun ControlNet Union patch loaded with the DiT", required=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("model", IOType.MODEL, "Qwen-Image-2.1 model bundle (DiT + TE + VAE)", is_array=False),
            PipeOutputSpec("text_encoder", IOType.TEXT_ENCODER, "Qwen3-VL-8B text encoder (ClipTextEncoder ABC)", is_array=False),
        ]

    def _control_path(self):
        return _path_of(self.config.get("control_model"))

    def _dit_key(self, ctx: LoaderContext) -> str:
        control_path = self._control_path()
        key = super()._dit_key(ctx)
        return f"{key}+control={control_path}" if control_path else key

    def _dit_fingerprint(self, ctx: LoaderContext) -> str:
        control_path = self._control_path()
        fp = super()._dit_fingerprint(ctx)
        return f"{fp}|control={control_path}" if control_path else fp

    def _dit_size_gb(self, ctx: LoaderContext):
        control_path = self._control_path()
        dit_gb = super()._dit_size_gb(ctx)
        if control_path and dit_gb is not None:
            dit_gb += file_size_gb(control_path) or 0.0
        return dit_gb

    def _dit_load_kwargs(self, ctx: LoaderContext):
        control_path = self._control_path()
        return {"model_patch": control_path} if control_path else {}
