from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from src.pipelines.pipes._shared.vae.minimax_h3_decode import decode_video
from src.platform.observability.profiling import get_profiler, profiling_enabled
from src.platform.runtime.native.vae.minimax_h3_latent_upsampler import normalize_h3_latent

logger = logging.getLogger(__name__)

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
PREVIEW_MAX_SIDE = 768
SOURCE_PREVIEW = "h3_upscale_0_source.png"
ENCODED_PREVIEW = "h3_upscale_1_encoded.png"
UPSCALED_PREVIEW = "h3_upscale_2_upscaled.png"


def diagnostics_dir() -> Optional[Path]:
    if not profiling_enabled():
        return None
    out_dir = get_profiler().out_dir
    return Path(out_dir) if out_dir is not None else None


def latent_stats(latent: torch.Tensor) -> dict[str, float]:
    raw = latent.detach().float()
    normalized = normalize_h3_latent(raw)
    return {
        "raw_mean": round(raw.mean().item(), 4),
        "raw_std": round(raw.std().item(), 4),
        "norm_mean": round(normalized.mean().item(), 4),
        "norm_std": round(normalized.std().item(), 4),
        "norm_abs_max": round(normalized.abs().max().item(), 4),
        "finite": bool(torch.isfinite(raw).all().item()),
    }


def preview_latent_frames(latent_frames: int, tokens_chunk_size: int, token_drop: int) -> int:
    smallest = tokens_chunk_size + (tokens_chunk_size - token_drop)
    return smallest if latent_frames >= smallest else 0


def to_preview_image(frame: torch.Tensor) -> Image.Image:
    mean = torch.tensor(PIXEL_MEAN, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(PIXEL_STD, dtype=torch.float32).view(-1, 1, 1)
    rgb = (frame.detach().float().cpu() * std + mean).clamp(0.0, 1.0)
    array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(array)
    scale = PREVIEW_MAX_SIDE / max(image.size)
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR)
    return image


def _decode_first_frame(vae: Any, latent: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    vae_module = vae.module
    frames = preview_latent_frames(int(latent.shape[2]), vae_module.tokens_chunk_size, vae_module.token_drop)
    if frames == 0:
        return None
    on_cuda = str(device).startswith("cuda")
    head = latent[:, :, :frames].to(device=device, dtype=vae.compute_dtype)
    with torch.no_grad(), torch.autocast(device_type="cuda" if on_cuda else "cpu", dtype=torch.float16, enabled=on_cuda):
        pixels = decode_video(vae_module, head, str(device), log_prefix="[H3_UPSCALE] preview")
    return pixels[0, :, 0]


def dump_upscale_previews(
    bundle: Any, device: str, source_frame: Optional[torch.Tensor],
    encoded: Optional[torch.Tensor], upscaled: torch.Tensor,
) -> list[str]:
    out_dir = diagnostics_dir()
    if out_dir is None:
        return []
    written: list[str] = []
    profiler = get_profiler()
    try:
        for stage, latent in (("encoded", encoded), ("upscaled", upscaled)):
            if latent is not None:
                profiler.mark("h3_upscale.latent_stats", stage=stage, shape=list(latent.shape), **latent_stats(latent))
        if source_frame is not None:
            path = out_dir / SOURCE_PREVIEW
            to_preview_image(source_frame).save(path)
            written.append(str(path))
        vae = bundle.video_vae
        vae.move_to(device)
        try:
            for name, latent in ((ENCODED_PREVIEW, encoded), (UPSCALED_PREVIEW, upscaled)):
                if latent is None:
                    continue
                frame = _decode_first_frame(vae, latent, device)
                if frame is None:
                    continue
                path = out_dir / name
                to_preview_image(frame).save(path)
                written.append(str(path))
        finally:
            vae.offload()
    except Exception:
        logger.warning("[H3_UPSCALE] preview dump to %s failed", out_dir, exc_info=True)
    profiler.mark("h3_upscale.previews", files=[Path(p).name for p in written])
    if written:
        logger.info("[H3_UPSCALE] wrote upscale previews: %s", ", ".join(written))
    return written
