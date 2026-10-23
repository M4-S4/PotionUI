from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.sampling.algorithms import (
    sample_dpmpp_2m,
    sample_euler,
    sample_euler_ancestral,
    sample_euler_ancestral_cfg_pp,
    sample_unipc,
)
from src.platform.runtime.native.sampling.algorithms.dpmpp_2m_sde import sample_dpmpp_2m_sde
from src.platform.runtime.native.sampling.algorithms.dpmpp_3m import sample_dpmpp_3m
from src.platform.runtime.native.sampling.algorithms.euler_cfg_pp import sample_euler_cfg_pp
from src.platform.runtime.native.sampling.algorithms.euler_sde import sample_euler_sde
from src.platform.runtime.native.sampling.algorithms.lcm import sample_lcm
from src.platform.runtime.native.sampling.algorithms.res_multistep import sample_res_multistep
from src.platform.runtime.native.sampling.cfg import NoCFG
from src.platform.runtime.native.sampling.hooks import BaseStepHook

SAMPLERS = {
    "euler": (sample_euler, None),
    "euler_ancestral": (sample_euler_ancestral, {"eta": 1.0}),
    "euler_ancestral_cfg_pp": (sample_euler_ancestral_cfg_pp, {"eta": 1.0}),
    "euler_cfg_pp": (sample_euler_cfg_pp, None),
    "euler_sde": (sample_euler_sde, {"eta": 1.0}),
    "dpmpp_2m": (sample_dpmpp_2m, None),
    "dpmpp_2m_sde": (sample_dpmpp_2m_sde, None),
    "dpmpp_3m": (sample_dpmpp_3m, None),
    "res_multistep": (sample_res_multistep, None),
    "lcm": (sample_lcm, None),
    "unipc": (sample_unipc, None),
}


class _Counter(BaseStepHook):
    def __init__(self):
        self.starts = 0
        self.steps = 0
        self.ends = 0
        self.total = None
        self.step_indices = []
        self.x0s = []

    def on_start(self, total_steps):
        self.starts += 1
        self.total = total_steps

    def on_step(self, step_index, total_steps, x, sigma, denoised_x0):
        self.steps += 1
        self.step_indices.append(step_index)
        self.x0s.append(denoised_x0)

    def on_end(self):
        self.ends += 1


def _const_velocity_model(v0):
    def model_fn(x, sigma, cond):
        return torch.full_like(x, v0)

    return model_fn


@pytest.mark.parametrize("name", SAMPLERS)
def test_hooks_fire_once_per_step(name):
    sampler, options = SAMPLERS[name]
    counter = _Counter()
    sigmas = torch.linspace(1.0, 0.0, 6)
    kwargs = {"hooks": [counter]}
    if options is not None:
        kwargs["sampler_options"] = dict(options)
    sampler(_const_velocity_model(1.0), torch.zeros(1, 2), sigmas, NoCFG(), {}, None, **kwargs)
    assert (counter.starts, counter.steps, counter.total, counter.ends) == (1, 5, 5, 1)
    assert counter.step_indices == list(range(5))
    assert all(x0 is not None for x0 in counter.x0s)
