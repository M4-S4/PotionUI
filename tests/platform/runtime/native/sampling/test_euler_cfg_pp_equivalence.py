from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.errors import SamplingCancelled
from src.platform.runtime.native.sampling.algorithms.euler_cfg_pp import sample_euler_cfg_pp
from src.platform.runtime.native.sampling.cfg import TrueCFG
from src.platform.runtime.native.sampling.hooks import (
    BaseStepHook,
    apply_latent_filters,
    run_hooks,
)
from src.platform.runtime.native.sampling.registry import sampler_registry


def frozen_sample_euler_cfg_pp(
    model_fn, x, sigmas, guidance, cond, uncond=None, hooks=(), is_cancelled=None, sampler_options=None,
):
    total_steps = len(sigmas) - 1
    run_hooks(hooks, "on_start", total_steps)
    zero_init_steps = getattr(guidance, "zero_init_steps", 0)
    s_in = x.new_ones((x.shape[0],))
    try:
        for i in range(total_steps):
            if is_cancelled is not None and is_cancelled():
                raise SamplingCancelled(step_index=i)
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            v_cfg = guidance(model_fn, x, sigma * s_in, cond, uncond, i)
            x0_est = x - sigma * v_cfg
            no_split = True
            if i < zero_init_steps:
                v_dir = v_cfg
            else:
                v_dir = getattr(guidance, "last_uncond_v", None)
                if v_dir is None:
                    v_dir = v_cfg
                else:
                    no_split = False
            if sigma_next == 0:
                x = x0_est
            elif no_split:
                x = x + (sigma_next - sigma) * v_cfg
            else:
                eps = x + (1.0 - sigma) * v_dir
                x = (1.0 - sigma_next) * x0_est + sigma_next * eps
            x = apply_latent_filters(hooks, i, total_steps, x, float(sigma_next))
            run_hooks(hooks, "on_step", i, total_steps, x, float(sigma), x0_est)
    finally:
        run_hooks(hooks, "on_end")
    return x


class _Recorder(BaseStepHook):
    def __init__(self):
        self.events = []

    def on_start(self, total_steps):
        self.events.append(("start", total_steps))

    def on_step(self, step_index, total_steps, x, sigma, denoised_x0):
        self.events.append(("step", step_index, x.clone(), sigma, denoised_x0.clone()))

    def on_end(self):
        self.events.append(("end",))


def _model_fn(x, sigma, cond):
    s = sigma.view(-1, *([1] * (x.dim() - 1)))
    return torch.tanh(x * cond["w"]) + cond["b"] * s


def _schedule(steps):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return 3.0 * s / (1.0 + 2.0 * s)


def _same_events(a, b):
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert ea[0] == eb[0]
        for fa, fb in zip(ea[1:], eb[1:]):
            if torch.is_tensor(fa):
                assert torch.equal(fa, fb)
            else:
                assert fa == fb


@pytest.mark.parametrize("cfg_zero_star", [True, False])
@pytest.mark.parametrize("zero_init_steps", [0, 2])
def test_new_matches_frozen_implementation_bitwise(cfg_zero_star, zero_init_steps):
    gen = torch.Generator().manual_seed(7)
    x_init = torch.randn(2, 4, 5, generator=gen)
    cond = {"w": 0.8, "b": 0.3}
    uncond = {"w": 0.5, "b": -0.2}
    sigmas = _schedule(8)

    def run(fn):
        guidance = TrueCFG(3.5, cfg_zero_star=cfg_zero_star, zero_init_steps=zero_init_steps)
        hook = _Recorder()
        out = fn(_model_fn, x_init.clone(), sigmas, guidance, cond, uncond, hooks=(hook,))
        return out, hook.events

    out_old, events_old = run(frozen_sample_euler_cfg_pp)
    out_new, events_new = run(sample_euler_cfg_pp)

    assert torch.equal(out_old, out_new)
    _same_events(events_old, events_new)


def test_registered_name_still_resolves_to_the_deterministic_sampler():
    definition = sampler_registry.get("euler_cfg_pp")
    assert not definition.stochastic
    x_init = torch.randn(1, 3, 4)
    sigmas = _schedule(5)
    cond = {"w": 0.8, "b": 0.3}
    uncond = {"w": 0.5, "b": -0.2}

    via_registry = definition.sample(_model_fn, x_init.clone(), sigmas, TrueCFG(2.5), cond, uncond)
    frozen = frozen_sample_euler_cfg_pp(_model_fn, x_init.clone(), sigmas, TrueCFG(2.5), cond, uncond)
    assert torch.equal(via_registry, frozen)


def test_caller_eta_and_generator_options_cannot_make_it_stochastic():
    x_init = torch.randn(1, 3, 4)
    sigmas = _schedule(5)
    cond = {"w": 0.8, "b": 0.3}
    uncond = {"w": 0.5, "b": -0.2}
    plain = sample_euler_cfg_pp(_model_fn, x_init.clone(), sigmas, TrueCFG(2.5), cond, uncond)
    noisy = sample_euler_cfg_pp(
        _model_fn, x_init.clone(), sigmas, TrueCFG(2.5), cond, uncond,
        sampler_options={"eta": 1.0, "generator": torch.Generator().manual_seed(1)},
    )
    assert torch.equal(plain, noisy)
