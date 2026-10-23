"""
Tensor Utilities - Centralized PIL/numpy/torch conversions.

Provides consistent conversion functions between PIL Images, numpy arrays,
and torch tensors used across multiple pipes in the generation pipeline.
"""

import numpy as np
import torch
from PIL import Image


def pil_to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to RGB numpy array.

    Args:
        image: PIL Image in any mode (RGB, RGBA, L, etc.)

    Returns:
        numpy array with shape (H, W, 3) and dtype uint8
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.

    Args:
        array: numpy array with shape (H, W, C) or (H, W) and dtype uint8

    Returns:
        PIL Image
    """
    return Image.fromarray(array)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor (B, C, H, W) to numpy image (H, W, C) uint8.

    Takes the first element in the batch dimension.

    Args:
        tensor: Torch tensor in range [0, 1] with shape (B, C, H, W)

    Returns:
        Numpy array in range [0, 255] with shape (H, W, C) and dtype uint8
    """
    image = tensor[0].permute(1, 2, 0)  # CHW -> HWC
    image = (image * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    return image
