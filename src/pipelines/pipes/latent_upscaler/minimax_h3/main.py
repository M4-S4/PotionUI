from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.pipelines.contracts import BasePipe
from src.pipelines.contracts import (
    IOType,
    PipeConfigSpec,
    PipeInput,
    PipeInputSpec,
    PipeOutput,
    PipeOutputSpec,
)
from src.pipelines.outputs import Icon, ProgressGenerationOutput
from src.platform.observability.profiling import get_profiler
from src.platform.runtime.device import clear_gpu_memory
from src.platform.runtime.model_lifecycle.lifecycle import empty_pinned_host_cache
from src.platform.runtime.native.memory.residency import (
    effective_free_vram_gb,
    get_residency_registry,
    minimum_inference_memory_gb,
)
from src.platform.runtime.native.vae.minimax_h3_latent_upsampler import denormalize_h3_latent, normalize_h3_latent
from src.platform.runtime.native.vae.minimax_h3_video import VaeCannotEncodeError, video_latent_frame_count
from src.pipelines.pipes._shared.video.frames import load_video_frames
from src.pipelines.pipes._shared.video.idle_encoder import release_idle_te
from src.pipelines.pipes.generator.video_minimax_h3.geometry import align_num_frames, resolve_canvas_size
from src.pipelines.pipes.latent_upscaler.minimax_h3.geometry import (
    UPSAMPLE_CHUNK,
    UPSAMPLE_CONTEXT,
    resolve_target_geometry,
    upsample_chunked,
)
from src.pipelines.pipes.latent_upscaler.minimax_h3.diagnostics import diagnostics_dir, dump_upscale_previews
from src.pipelines.pipes._shared.vae.minimax_h3_encode import (
    encode_activation_gb,
    encode_compute_dtype,
    encode_video,
)
from src.pipelines.pipes._shared.generation.config_specs import device_config_spec

logger = logging.getLogger(__name__)

_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)
_UPSAMPLE_LIVE_TENSORS = 6
_PREP_FRAMES_PER_BATCH = 32
UPSCALE_NEEDS_ENCODER_MESSAGE = "This VAE can only decode; pick the standard MiniMax-H3 video VAE for Upscale"


def _resize_cover_center_crop_01(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    chw = frames.permute(0, 3, 1, 2)
    _, _, h0, w0 = chw.shape
    scale = max(width / w0, height / h0)
    chw = F.interpolate(chw, size=(round(h0 * scale), round(w0 * scale)), mode="bilinear", align_corners=False)
    _, _, h1, w1 = chw.shape
    top, left = (h1 - height) // 2, (w1 - width) // 2
    chw = chw[:, :, top:top + height, left:left + width]
    return chw.permute(1, 0, 2, 3).unsqueeze(0)


def _unload_idle_te(bundle: Any, models: Any) -> float:
    return release_idle_te(bundle, models, "H3_UPSCALE")


def _upsample_activation_gb(upsampler: Any, latent_frames: int, latent_height: int, latent_width: int) -> float:
    channels = getattr(upsampler.module, "channels", None)
    if not channels:
        return 0.0
    frames = min(latent_frames, UPSAMPLE_CHUNK + 2 * UPSAMPLE_CONTEXT)
    element = torch.empty((), dtype=getattr(upsampler, "compute_dtype", torch.float32)).element_size()
    voxels = frames * latent_height * latent_width
    return voxels * channels * element * _UPSAMPLE_LIVE_TENSORS / (1 << 30)


def _upscale_need_gb(
    bundle: Any, geometry: Any, latent_frames: int, encode_canvas: Optional[Tuple[int, int]],
) -> float:
    upsampler = bundle.upsampler
    need = (getattr(upsampler, "estimated_vram_gb", None) or 0.0) + _upsample_activation_gb(
        upsampler, latent_frames, geometry.latent_height, geometry.latent_width,
    )
    if encode_canvas is not None:
        vae = bundle.video_vae
        height, width = encode_canvas
        encode = (getattr(vae, "estimated_vram_gb", None) or 0.0) + encode_activation_gb(
            vae.module, height, width, encode_compute_dtype(vae.module),
        )
        need = max(need, encode)
    return need


def _free_room_for_upscale(bundle: Any, device: str, models: Any = None, need_gb: Optional[float] = None) -> None:
    te_unloaded_gb = _unload_idle_te(bundle, models)
    get_profiler().census_now("h3_upscale.te_evicted")
    empty_pinned_host_cache()
    pinned_emptied = True

    if not str(device).startswith("cuda"):
        get_profiler().mark(
            "h3_upscale.free_room", device=str(device), dit_was_resident=False, dit_offloaded=False,
            te_unloaded_gb=round(te_unloaded_gb, 2), pinned_emptied=pinned_emptied,
        )
        return
    if need_gb is None:
        need_gb = getattr(bundle.upsampler, "estimated_vram_gb", None) or 0.0
    need_gb += minimum_inference_memory_gb()
    alloc0 = torch.cuda.memory_allocated(device) / (1 << 30) if torch.cuda.is_available() else 0.0
    dit = getattr(bundle, "dit", None)
    dit_dev = str(getattr(dit, "device", "<no dit>"))
    dit_was_resident = dit is not None and dit_dev.startswith("cuda")
    dit_offloaded = False
    free_gb = effective_free_vram_gb(device)
    if dit_was_resident and (free_gb is None or free_gb < need_gb):
        dit.offload()
        dit_offloaded = True
    own_models = tuple(
        m for m in (
            getattr(bundle, "video_vae", None),
            getattr(bundle, "audio_vae", None),
            getattr(bundle, "upsampler", None),
            dit if dit_was_resident and not dit_offloaded else None,
        ) if m is not None
    )
    get_residency_registry().offload_all(device, exclude=own_models)
    clear_gpu_memory()
    alloc1 = torch.cuda.memory_allocated(device) / (1 << 30) if torch.cuda.is_available() else 0.0
    logger.info(
        "[H3_UPSCALE] eviction pass: dit.device=%s, dit_offloaded=%s, need_gb=%.2f, free_gb=%s, "
        "allocated %.2fGB -> %.2fGB, te_unloaded_gb=%.2f",
        dit_dev, dit_offloaded, need_gb, "n/a" if free_gb is None else f"{free_gb:.2f}", alloc0, alloc1,
        te_unloaded_gb,
    )
    get_profiler().mark(
        "h3_upscale.free_room", device=str(device), dit_was_resident=dit_was_resident, dit_offloaded=dit_offloaded,
        need_gb=round(need_gb, 2), free_gb=None if free_gb is None else round(free_gb, 2),
        alloc_before_gb=round(alloc0, 2), alloc_after_gb=round(alloc1, 2),
        te_unloaded_gb=round(te_unloaded_gb, 2), pinned_emptied=pinned_emptied,
    )


def _offload_resident_dit(bundle: Any, device: str) -> bool:
    dit = getattr(bundle, "dit", None)
    if dit is None or not str(getattr(dit, "device", "")).startswith("cuda"):
        return False
    dit.offload()
    clear_gpu_memory()
    get_profiler().mark("h3_upscale.oom_dit_offload", device=str(device))
    logger.warning("[H3_UPSCALE] VAE encode ran out of VRAM beside the resident DiT; offloaded it and retrying")
    return True


class LatentUpscalerMinimaxH3Pipe(BasePipe):
    name = "latent_upscaler"
    description = "Upsample a MiniMax-H3 video latent spatially, via the model's optional upscaler checkpoint"

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            device_config_spec(),
            PipeConfigSpec("target_mode", str, "megapixels", "How to size the upscale target: a decimal-megapixel "
                           "area, or a straight axis multiplier", required=False, choices=["megapixels", "scale"]),
            PipeConfigSpec("megapixels", float, 2.1, "Target area in megapixels (target_mode='megapixels')",
                           required=False, min_value=0.5, max_value=8.0),
            PipeConfigSpec("scale", float, 2.0, "Target axis multiplier (target_mode='scale')",
                           required=False, min_value=1.0, max_value=4.0),
            PipeConfigSpec("max_frames", int, 601,
                           "Max frames to read from a 'video' input (standalone upscale mode)",
                           required=False, min_value=1, max_value=601),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("model", IOType.MODEL, True,
                          "MiniMax-H3 model bundle (needs the video VAE plus the upscale_model slot)",
                          is_array=False),
            PipeInputSpec("latent", IOType.LATENT, False,
                          "Direct latent to upsample (in-flow two-stage)", is_array=False),
            PipeInputSpec("video", IOType.VIDEO, False,
                          "Existing video file to VAE-encode then upsample (standalone upscale mode)",
                          is_array=True),
            PipeInputSpec("MODELS", IOType.SERVICE, False,
                          "Model lifecycle service, to release the idle TE's host RAM",
                          is_array=False),
        ]

    @classmethod
    def outputs(cls) -> List[PipeOutputSpec]:
        return [
            PipeOutputSpec("latent", IOType.LATENT, "Upsampled latent", is_array=False),
            PipeOutputSpec("source_frame_count", IOType.INT,
                           "Original (pre-temporal-padding) frame count of the 'video' input, before "
                           "the pipe repeated its last frame up to the video VAE's 17*n+5 lattice "
                           "-- lets a downstream refine trim that padding back out of its own decoded output "
                           "before mux. None on the 'latent'-input path (in-flow two-stage: no video was "
                           "decoded here, so no padding was ever added)", is_array=False),
        ]

    def process(self, pipe_input: PipeInput, generation_outputs: callable) -> PipeOutput:
        bundle = pipe_input.input["model"]
        upsampler = getattr(bundle, "upsampler", None)
        if upsampler is None:
            raise ValueError(
                "latent_upscaler/minimax_h3: the model bundle has no latent upsampler loaded -- set "
                "model_loader/minimax_h3's 'upscale_model' config to a MiniMax-H3 3D latent-upscaler checkpoint"
            )

        device = self.config.get("device", "cuda")
        models = pipe_input.input.get("MODELS")

        latent = pipe_input.input.get("latent")
        source_frame_count: Optional[int] = None
        pixels: Optional[torch.Tensor] = None
        if latent is None:
            videos = pipe_input.input.get("video") or []
            if not videos:
                raise ValueError("latent_upscaler/minimax_h3 requires either a 'latent' or a 'video' input")
            if not getattr(bundle.video_vae.module, "can_encode", True):
                raise VaeCannotEncodeError(UPSCALE_NEEDS_ENCODER_MESSAGE)
            pixels, source_frame_count = self._prepare_pixels(videos[0])
            source_height, source_width = int(pixels.shape[-2]), int(pixels.shape[-1])
            latent_frames = video_latent_frame_count(int(pixels.shape[2]))
        else:
            _, _, latent_frames, lat_h, lat_w = latent.shape
            source_height, source_width = lat_h * 16, lat_w * 16

        target_mode = str(self.config.get("target_mode", "megapixels"))
        geometry = resolve_target_geometry(
            source_height, source_width, mode=target_mode,
            megapixels=float(self.config.get("megapixels", 2.1)), scale=float(self.config.get("scale", 2.0)),
        )

        encode_canvas = (source_height, source_width) if pixels is not None else None
        _free_room_for_upscale(
            bundle, device, models, need_gb=_upscale_need_gb(bundle, geometry, int(latent_frames), encode_canvas),
        )

        dumping = diagnostics_dir() is not None
        source_frame: Optional[torch.Tensor] = None
        if pixels is not None:
            if dumping:
                source_frame = pixels[0, :, 0].clone()
            generation_outputs(ProgressGenerationOutput(
                state="Encoding source video", icon=Icon(name="film", effect="pulse")))
            latent = self._encode_pixels(bundle, pixels, device)
            del pixels

        generation_outputs(ProgressGenerationOutput(
            state="Upsampling latent", icon=Icon(name="bolt", effect="pulse")))
        upsampled = self._upsample(upsampler, latent, geometry, device)
        if dumping:
            dump_upscale_previews(bundle, device, source_frame, latent, upsampled)
        return PipeOutput(output={"latent": upsampled, "source_frame_count": source_frame_count})

    def _prepare_pixels(self, video_path: str) -> Tuple[torch.Tensor, int]:
        max_frames = int(self.config.get("max_frames", 601))
        frames = load_video_frames(video_path, max_frames)
        n0, h0, w0 = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        height, width = resolve_canvas_size(w0, h0)
        total = align_num_frames(n0)

        pixels = torch.empty(1, 3, total, height, width, dtype=torch.float32)
        for start in range(0, n0, _PREP_FRAMES_PER_BATCH):
            batch = frames[start:start + _PREP_FRAMES_PER_BATCH]
            pixels[0, :, start:start + batch.shape[0]] = _resize_cover_center_crop_01(batch, height, width)[0]
        del frames
        if total > n0:
            pixels[:, :, n0:] = pixels[:, :, n0 - 1:n0]

        pixels.sub_(torch.tensor(_PIXEL_MEAN, dtype=torch.float32).view(1, -1, 1, 1, 1))
        pixels.div_(torch.tensor(_PIXEL_STD, dtype=torch.float32).view(1, -1, 1, 1, 1))
        return pixels, n0

    @staticmethod
    def _encode_pixels(bundle: Any, pixels: torch.Tensor, device: str) -> torch.Tensor:
        vae = bundle.video_vae
        vae.move_to(device)
        try:
            pixels = pixels.to(dtype=encode_compute_dtype(vae.module))
            return encode_video(vae.module, pixels, device, free_more=lambda: _offload_resident_dit(bundle, device))
        finally:
            vae.offload()

    @staticmethod
    def _upsample(upsampler: Any, latent: torch.Tensor, geometry: Any, device: str) -> torch.Tensor:
        upsampler.move_to(device)
        try:
            with torch.no_grad():
                z = latent.to(device=device, dtype=upsampler.compute_dtype)
                z = normalize_h3_latent(z)
                z = upsample_chunked(
                    upsampler.module, z, geometry.effective_scale,
                    (geometry.latent_height, geometry.latent_width),
                    chunk=UPSAMPLE_CHUNK, overlap=UPSAMPLE_CONTEXT,
                )
                z = denormalize_h3_latent(z)
        finally:
            upsampler.offload()
        return z
