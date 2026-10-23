
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import torch

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
from src.platform.runtime.native.memory.residency import get_residency_registry
from src.platform.runtime.native.resolution import snap_resolution
from src.platform.runtime.native.vae.ltx_tiling import LtxTilingConfig
from src.pipelines.pipes._shared.vae.ltx_latent_upsample import upsample_ltx_latent
from src.pipelines.pipes._shared.vae.ltx_tiled_encode import encode_with_oom_retry
from src.pipelines.pipes._shared.video.frames import load_video_frames, resize_cover_center_crop
from src.pipelines.pipes._shared.video.idle_encoder import release_idle_te
from src.pipelines.pipes._shared.video.ltx_constants import LTX_SPATIAL_DOWNSCALE, LTX_TEMPORAL_DOWNSCALE
from src.pipelines.pipes.latent_upscaler.ltx.geometry import temporal_upsample_out_frames
from src.pipelines.pipes._shared.generation.config_specs import device_config_spec

logger = logging.getLogger(__name__)


def _pad_frames_to_temporal_grid(frames: torch.Tensor, temporal_downscale: int) -> torch.Tensor:
    """Pad ``frames`` (``(n, H, W, 3)``) up to the next valid ``1 + k*temporal_downscale``
    count by repeating its last frame -- NEVER truncates, so all source content
    is preserved. A no-op when ``frames`` is already on the grid."""
    n0 = int(frames.shape[0])
    td = max(1, int(temporal_downscale))
    pad = (td - (n0 - 1) % td) % td
    if pad == 0:
        return frames
    last = frames[-1:].expand(pad, *frames.shape[1:])
    return torch.cat([frames, last], dim=0)


def _resolve_upsampler(bundle: Any, mode: str) -> Any:
    """The bundle component ``mode`` asks for, cross-checked against the
    checkpoint that was actually loaded into it.

    The slot and the checkpoint are independent facts -- a preset can point
    ``temporal_upscale_model`` at a spatial file -- and getting them crossed
    is silently wrong rather than loud: a spatial checkpoint run in temporal
    mode returns the frame count untouched, so the stage that follows renders
    half the clip the preset asked for. Both directions are checked here,
    before any GPU work.
    """
    if mode == "temporal":
        component = getattr(bundle, "temporal_upsampler", None)
        slot = "temporal_upscale_model"
    elif mode == "spatial":
        component = getattr(bundle, "upsampler", None)
        slot = "upscale_model"
    else:
        raise ValueError(f"latent_upscaler/ltx: unknown mode {mode!r} (expected 'spatial' or 'temporal')")

    if component is None:
        raise ValueError(
            f"latent_upscaler/ltx: mode='{mode}' but the model bundle has no {mode} upscaler loaded -- "
            f"set model_loader/ltx's '{slot}' config to an LTX-2.3/2.5 {mode} latent-upscaler checkpoint"
        )

    is_temporal = bool(getattr(component.module, "temporal_upsample", False))
    if mode == "temporal" and not is_temporal:
        raise ValueError(
            f"latent_upscaler/ltx: mode='temporal' but the checkpoint in model_loader/ltx's '{slot}' "
            "declares temporal_upsample=false -- it is a spatial upscaler and would leave the frame "
            "count unchanged; point that slot at the LTX-2.5 temporal x2 upscaler"
        )
    if mode == "spatial" and is_temporal:
        raise ValueError(
            f"latent_upscaler/ltx: mode='spatial' but the checkpoint in model_loader/ltx's '{slot}' "
            "declares temporal_upsample=true -- running it here would change the frame count; "
            "use mode='temporal', or point that slot at a spatial upscaler"
        )
    return component


def _unload_idle_te(bundle: Any, models: Any) -> float:
    return release_idle_te(bundle, models, "LTX_UPSCALE")


def _free_room_for_upscale(bundle: Any, device: str, models: Any = None) -> None:
    """Evict a resident DiT and every other foreign GPU-resident component
    before this pipe's own VAE-encode/upsampler-forward GPU work (see the module
    docstring), plus fully unload the idle TE's HOST RAM via
    :func:`_unload_idle_te`.

    Unconditional except for the DiT's own move: ``bundle.dit`` is only
    offloaded when it's actually GPU-resident right now (``dit_restore.py``
    may have left it on CPU already, e.g. a partial-residency generation), but
    ``GpuResidencyRegistry.offload_all``/``clear_gpu_memory``/the TE unload
    always run -- mirrors ``dit_placement.py``'s ``_ensure_room_for`` idiom,
    reused rather than reinvented. ``bundle.vae`` and both upsampler slots are
    excluded from the GPU eviction sweep since they're what THIS pipe is about
    to move onto the GPU itself (excluding the slot this call won't use costs
    nothing -- skipping an absent or CPU-resident component is a no-op).
    GPU eviction is a no-op on a non-CUDA device
    (mirrors ``place_dit_for_sequence``'s identical early return) so
    CPU-configured tests/deploys never touch the residency manager at all --
    but the TE unload is HOST-RAM work, independent of ``device``, so it still
    runs even in that branch (a CPU-configured pipe still benefits from
    freeing 22GB of idle RAM).
    """
    te_unloaded_gb = _unload_idle_te(bundle, models)
    # Census right after the eviction: a SIGKILL'd run never reaches
    # GenerationProfiler.stop(), so this is the only chance to record what is
    # still holding memory after an eviction that reported success (a stray
    # strong reference to the TE can keep ~22GB resident -- see
    # `model_loader/ltx/ltx_clip.py`). Coarse (once per call, never per-step).
    get_profiler().census_now("ltx_upscale.te_evicted")
    # Release any stale pinned-host pool left by an earlier phase's torn-down
    # partial-residency placement -- same rationale as engine.py's stream_to().
    # ``empty_pinned_host_cache`` is best-effort (never raises).
    empty_pinned_host_cache()
    pinned_emptied = True

    if not str(device).startswith("cuda"):
        get_profiler().mark(
            "ltx_upscale.free_room", device=str(device), dit_was_resident=False,
            te_unloaded_gb=round(te_unloaded_gb, 2), pinned_emptied=pinned_emptied,
        )
        return
    alloc0 = torch.cuda.memory_allocated(device) / (1 << 30) if torch.cuda.is_available() else 0.0
    dit = getattr(bundle, "dit", None)
    dit_dev = str(getattr(dit, "device", "<no dit>"))
    dit_was_resident = dit is not None and dit_dev.startswith("cuda")
    if dit_was_resident:
        dit.offload()
    own_models = tuple(
        m for m in (
            getattr(bundle, "vae", None),
            getattr(bundle, "upsampler", None),
            getattr(bundle, "temporal_upsampler", None),
        ) if m is not None
    )
    get_residency_registry().offload_all(device, exclude=own_models)
    clear_gpu_memory()
    alloc1 = torch.cuda.memory_allocated(device) / (1 << 30) if torch.cuda.is_available() else 0.0
    logger.debug(
        "[LTX_UPSCALE] eviction pass: dit.device=%s, allocated %.2fGB -> %.2fGB, te_unloaded_gb=%.2f",
        dit_dev, alloc0, alloc1, te_unloaded_gb,
    )
    get_profiler().mark(
        "ltx_upscale.free_room", device=str(device), dit_was_resident=dit_was_resident,
        alloc_before_gb=round(alloc0, 2), alloc_after_gb=round(alloc1, 2),
        te_unloaded_gb=round(te_unloaded_gb, 2), pinned_emptied=pinned_emptied,
    )
    # If eviction barely moved the needle, name what is actually holding CUDA
    # memory: walk live nn.Modules with cuda-resident parameters. This is the
    # runtime twin of the lifecycle manager's tensor-level referrer diagnostic
    # (which only fires on ITS OWN eviction paths, not this pipe's sweep).
    if alloc1 > 5.0:
        import gc as _gc
        seen: dict = {}
        for obj in _gc.get_objects():
            try:
                if isinstance(obj, torch.nn.Module) and not isinstance(obj, torch.nn.modules.container.ModuleList):
                    p = next(obj.parameters(recurse=False), None)
                    if p is not None and p.is_cuda:
                        gb = sum(q.numel() * q.element_size() for q in obj.parameters(recurse=False)) / (1 << 30)
                        cls = type(obj).__name__
                        seen[cls] = seen.get(cls, 0.0) + gb
            except Exception:
                continue
        top = sorted(seen.items(), key=lambda kv: -kv[1])[:8]
        logger.warning(
            "[LTX_UPSCALE] %.2fGB still allocated after eviction; cuda-resident module classes (leaf, GB): %s",
            alloc1, ", ".join(f"{c}={g:.2f}" for c, g in top) or "<none found via gc>",
        )


class LatentUpscalerLtxPipe(BasePipe):
    name = "latent_upscaler"
    description = ("Upsample an LTX-2/2.3/2.5 video latent in latent space -- spatially (x1.5/x2.0) "
                   "or temporally (x2 frames) -- via the model's optional upscaler checkpoints")

    @classmethod
    def configuration(cls) -> List[PipeConfigSpec]:
        return [
            device_config_spec(),
            PipeConfigSpec("mode", str, "spatial",
                           "Which upscaler to run: 'spatial' (model_loader/ltx's upscale_model, x1.5/x2.0, "
                           "frame count unchanged) or 'temporal' (its temporal_upscale_model, LTX-2.5 x2, "
                           "T -> 2T-1 latent frames)",
                           required=False, choices=["spatial", "temporal"]),
            PipeConfigSpec("max_frames", int, 1001,
                           "Max frames to read from a 'video' input (standalone upscale mode)",
                           required=False, min_value=1, max_value=1001),
        ]

    @classmethod
    def inputs(cls) -> List[PipeInputSpec]:
        return [
            PipeInputSpec("model", IOType.MODEL, True,
                          "LTX model bundle (needs the VAE plus the upsampler slot 'mode' selects)",
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
                           "_pad_frames_to_temporal_grid repeated its last frame up to the VAE's 1+8k "
                           "lattice -- lets a downstream generator trim that padding back out of its own "
                           "decoded output before mux. In mode='temporal' this is mapped through the same "
                           "T -> 2T-1 the upsampler applies, so it still names the source content's extent "
                           "on the NEW timeline. None on the 'latent'-input path (in-flow two-stage: "
                           "no video was decoded here, so no padding was ever added)", is_array=False),
        ]

    def process(self, pipe_input: PipeInput, generation_outputs: callable) -> PipeOutput:
        bundle = pipe_input.input["model"]
        mode = str(self.config.get("mode", "spatial"))
        upsampler = _resolve_upsampler(bundle, mode)

        device = self.config.get("device", "cuda")
        models = pipe_input.input.get("MODELS")

        # Make room BEFORE any of this pipe's own GPU work -- covers both the
        # video-input encode below and the upsampler forward in `_upsample`.
        # See `_free_room_for_upscale`. `models` additionally lets it release
        # the idle TE's host RAM.
        _free_room_for_upscale(bundle, device, models)

        latent = pipe_input.input.get("latent")
        source_frame_count: Optional[int] = None
        if latent is None:
            videos = pipe_input.input.get("video") or []
            if not videos:
                raise ValueError("latent_upscaler/ltx requires either a 'latent' or a 'video' input")
            generation_outputs(ProgressGenerationOutput(
                state="Encoding source video", icon=Icon(name="film", effect="pulse")))
            latent, source_frame_count = self._encode_video(bundle, videos[0], device)

        generation_outputs(ProgressGenerationOutput(
            state=f"Upsampling latent ({mode})", icon=Icon(name="bolt", effect="pulse")))
        upsampled = self._upsample(bundle, upsampler, latent, device)

        if mode == "temporal" and source_frame_count is not None:
            # The source content now spans twice the timeline (module
            # docstring), so the trim hint has to move with it or a downstream
            # generator would cut the clip in half.
            source_frame_count = temporal_upsample_out_frames(source_frame_count)
        return PipeOutput(output={"latent": upsampled, "source_frame_count": source_frame_count})

    # -- helpers -------------------------------------------------------------

    def _encode_video(self, bundle: Any, video_path: str, device: str) -> Tuple[torch.Tensor, int]:
        max_frames = int(self.config.get("max_frames", 1001))
        frames = load_video_frames(video_path, max_frames)
        n0, h0, w0 = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        width, height = snap_resolution(w0, h0, LTX_SPATIAL_DOWNSCALE, 1)

        frames = frames.to(device=device)
        frames = _pad_frames_to_temporal_grid(frames, LTX_TEMPORAL_DOWNSCALE)
        pixels = resize_cover_center_crop(frames, height, width)
        del frames

        bundle.vae.move_to(device)
        try:
            latent = self._encode_with_oom_retry(bundle, pixels, device)
        finally:
            bundle.vae.offload()
            del pixels
        return latent, n0

    @staticmethod
    def _encode_with_oom_retry(
        bundle: Any, pixels: torch.Tensor, device: str,
        tiling_config: LtxTilingConfig | None = None,
    ) -> torch.Tensor:
        """``bundle.vae.module.encode(pixels)``, with the shared VRAM-aware
        ladder (``_shared/vae/ltx_tiled_encode.py`` -- see the module
        docstring)."""
        return encode_with_oom_retry(
            bundle.vae, pixels, device, tiling_config=tiling_config,
            profiler_mark="ltx_upscale.encode", log_prefix="latent_upscaler/ltx",
        )

    @staticmethod
    def _upsample(bundle: Any, upsampler: Any, latent: torch.Tensor, device: str) -> torch.Tensor:
        """Thin wrapper over the shared recipe
        (``_shared/vae/ltx_latent_upsample.py`` -- see its module docstring for
        the normalization sandwich and the temporal frame-count contract);
        ``upsampler`` is whichever slot ``mode`` resolved to
        (:func:`_resolve_upsampler`). Kept as a method for call-site/test
        stability, like ``_encode_with_oom_retry`` above."""
        return upsample_ltx_latent(bundle, upsampler, latent, device)
