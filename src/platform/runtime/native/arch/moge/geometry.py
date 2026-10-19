from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "depth_to_points",
    "focal_to_fx_fy",
    "fov_from_focal",
    "normalized_view_plane_uv",
    "recover_focal",
    "recover_focal_shift",
    "solve_focal_shift",
]

_DOWNSAMPLE = (64, 64)


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device=None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def solve_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> Tuple[float, float]:
    from scipy.optimize import least_squares

    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def residual(shift):
        projected = xy / (z + shift)[:, None]
        focal = (projected * uv).sum() / np.square(projected).sum()
        return (focal * projected - uv).ravel()

    solution = least_squares(residual, x0=0, ftol=1e-3, method="lm")
    shift = solution["x"].squeeze().astype(np.float32)
    projected = xy / (z + shift)[:, None]
    focal = (projected * uv).sum() / np.square(projected).sum()
    return float(shift), float(focal)


def recover_focal(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    downsample: Tuple[int, int] = _DOWNSAMPLE,
) -> List[Optional[float]]:
    return [pair[0] if pair is not None else None for pair in recover_focal_shift(points, mask, downsample)]


def recover_focal_shift(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    downsample: Tuple[int, int] = _DOWNSAMPLE,
) -> List[Optional[Tuple[float, float]]]:
    height, width = points.shape[-3], points.shape[-2]
    points = points.reshape(-1, height, width, 3).float()
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)
    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample, mode="nearest").permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample, mode="nearest").squeeze(0).permute(1, 2, 0)
    points_np = points_lr.detach().cpu().numpy()
    uv_np = uv_lr.detach().cpu().numpy()
    if mask is None:
        mask_np = None
    else:
        mask = mask.reshape(-1, height, width).to(torch.float32).unsqueeze(1)
        mask_np = (F.interpolate(mask, downsample, mode="nearest").squeeze(1) > 0).cpu().numpy()

    solutions: List[Optional[Tuple[float, float]]] = []
    for index in range(points_np.shape[0]):
        xyz = points_np[index] if mask_np is None else points_np[index][mask_np[index]]
        grid = uv_np if mask_np is None else uv_np[mask_np[index]]
        xyz = xyz.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
        finite = np.isfinite(xyz).all(axis=-1)
        xyz, grid = xyz[finite], grid[finite]
        if grid.shape[0] < 2:
            solutions.append(None)
            continue
        shift, focal = solve_focal_shift(grid, xyz)
        usable = math.isfinite(focal) and focal > 0 and math.isfinite(shift)
        solutions.append((focal, shift) if usable else None)
    return solutions


def focal_to_fx_fy(focal: float, aspect_ratio: float) -> Tuple[float, float]:
    diagonal = (1 + aspect_ratio ** 2) ** 0.5
    return focal / 2 * diagonal / aspect_ratio, focal / 2 * diagonal


def fov_from_focal(focal: float, aspect_ratio: float) -> Tuple[float, float]:
    fx, fy = focal_to_fx_fy(focal, aspect_ratio)
    return math.degrees(2 * math.atan(0.5 / fx)), math.degrees(2 * math.atan(0.5 / fy))


def depth_to_points(depth: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    height, width = depth.shape[-2:]
    u = (torch.arange(width, dtype=depth.dtype, device=depth.device) + 0.5) / width
    v = (torch.arange(height, dtype=depth.dtype, device=depth.device) + 0.5) / height
    v, u = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([(u - 0.5) / fx * depth, (v - 0.5) / fy * depth, depth], dim=-1)
