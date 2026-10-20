import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def release_lm(bundle: Any, models: Any) -> None:
    key = getattr(bundle, "lm_cache_key", None)
    if not key or models is None:
        return
    evict = getattr(models, "evict_dead_weight", None)
    if callable(evict):
        evict(key)


def write_wav(waveform: torch.Tensor, sample_rate: int) -> Path:
    samples = waveform[0].detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0).numpy()
    samples = np.ascontiguousarray(samples.T)
    out_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    sf.write(str(out_path), samples, sample_rate)
    return out_path
