from __future__ import annotations

import pytest
import torch

from src.platform.runtime.native.errors import SamplingCancelled, SamplingNumericsError
from src.platform.runtime.native.sampling.algorithms import (
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m,
    sample_er_sde,
    sample_euler,
    sample_euler_ancestral,
    sample_euler_ancestral_cfg_pp,
    sample_euler_cfg_pp,
    sample_euler_restart,
    sample_euler_sde,
    sample_lcm,
    sample_res_multistep,
    sample_unipc,
)
from src.platform.runtime.native.sampling.cfg import NoCFG
from src.platform.runtime.native.sampling.hooks import BaseStepHook, StepLoop
from src.platform.runtime.native.sampling.noise import fresh_noise

SAMPLERS = [
    sample_euler,
    sample_euler_sde,
    sample_euler_ancestral,
    sample_euler_ancestral_cfg_pp,
    sample_euler_cfg_pp,
    sample_euler_restart,
    sample_lcm,
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m,
    sample_er_sde,
    sample_res_multistep,
    sample_unipc,
]


class _Events(BaseStepHook):
    def __init__(self):
        self.events = []

    def on_start(self, total_steps):
        self.events.append(("start", total_steps))

    def on_step(self, step_index, total_steps, x, sigma, denoised_x0):
        self.events.append(("step", step_index))

    def on_end(self):
        self.events.append(("end",))


def _model(x, sigma, cond):
    return torch.zeros_like(x)


def test_iterates_global_indices_from_start_step():
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0])
    with StepLoop(sigmas, (), None, 2) as loop:
        seen = [(i, total, float(s), float(sn)) for i, total, s, sn in loop]
    assert [i for i, *_ in seen] == [2, 3]
    assert {total for _, total, *_ in seen} == {4}
    assert seen[0][2:] == pytest.approx((0.4, 0.1))


def test_finish_step_dispatches_on_step_and_returns_latent():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    hook = _Events()
    x = torch.zeros(1, 2)
    with StepLoop(sigmas, (hook,)) as loop:
        for i, _, sigma, sigma_next in loop:
            x = loop.finish_step(x, i, sigma, sigma_next, x)
    assert hook.events == [("start", 2), ("step", 0), ("step", 1), ("end",)]


def test_cancel_raises_with_step_index_and_still_ends():
    sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0])
    hook = _Events()
    calls = []

    def is_cancelled():
        calls.append(1)
        return len(calls) > 1

    with pytest.raises(SamplingCancelled) as info:
        with StepLoop(sigmas, (hook,), is_cancelled) as loop:
            for i, _, sigma, sigma_next in loop:
                loop.finish_step(torch.zeros(1), i, sigma, sigma_next, None)
    assert info.value.step_index == 1
    assert hook.events[-1] == ("end",)


def test_total_steps_override_reaches_hooks():
    hook = _Events()
    with StepLoop(torch.tensor([1.0, 0.0]), (hook,), total_steps=7):
        pass
    assert hook.events == [("start", 7), ("end",)]


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_on_end_fires_when_pre_loop_setup_raises(sampler):
    hook = _Events()
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    with pytest.raises(AttributeError):
        sampler(_model, object(), sigmas, NoCFG(), {}, None, hooks=(hook,))
    assert hook.events == [("start", 2), ("end",)]


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_every_sampler_brackets_steps_with_start_and_end(sampler):
    hook = _Events()
    sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
    out = sampler(_model, torch.ones(1, 2, 3), sigmas, NoCFG(), {}, None, hooks=(hook,))
    assert out.shape == (1, 2, 3)
    assert hook.events[0][0] == "start"
    assert hook.events[-1] == ("end",)
    assert [e[1] for e in hook.events if e[0] == "step"] == list(range(hook.events[0][1]))


@pytest.mark.parametrize("sampler", SAMPLERS, ids=lambda f: f.__name__)
def test_every_sampler_cancels_with_the_failing_step_and_closes_hooks(sampler):
    hook = _Events()
    sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
    with pytest.raises(SamplingCancelled):
        sampler(
            _model, torch.ones(1, 2, 3), sigmas, NoCFG(), {}, None,
            hooks=(hook,), is_cancelled=lambda: True,
        )
    assert hook.events == [("start", 3), ("end",)]


def test_unipc_numerics_error_still_carries_solver_state():
    class _Poison(BaseStepHook):
        priority = 10

        def on_step(self, step_index, total_steps, x, sigma, denoised_x0):
            raise SamplingNumericsError(step_index, "unipc", None, tensor_name="x")

    with pytest.raises(SamplingNumericsError) as info:
        sample_unipc(
            _model, torch.ones(1, 2), torch.tensor([1.0, 0.5, 0.0]), NoCFG(), {}, None,
            hooks=(_Poison(),),
        )
    assert info.value.solver_order == 1
    assert info.value.history_depth == 1


def test_fresh_noise_with_generator_matches_randn_on_that_generator():
    x = torch.zeros(2, 3)
    a = fresh_noise(x, torch.Generator().manual_seed(5))
    b = torch.randn(x.shape, generator=torch.Generator().manual_seed(5), dtype=x.dtype)
    assert torch.equal(a, b)


def test_fresh_noise_without_generator_draws_from_global_rng():
    x = torch.zeros(2, 3)
    torch.manual_seed(5)
    a = fresh_noise(x, None)
    torch.manual_seed(5)
    assert torch.equal(a, torch.randn_like(x))
