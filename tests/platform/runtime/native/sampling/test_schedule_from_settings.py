from __future__ import annotations

import logging

import pytest
import torch

from src.platform.runtime.native.sampling.cfg import NoCFG
from src.platform.runtime.native.sampling.conditioned import conditioned_sigmas
from src.platform.runtime.native.sampling.denoise_loop import (
    fp32_trajectory_forward,
    schedule_from_settings,
    warn_discarded_guidance_scale,
    wrap_step_cache,
)
from src.platform.runtime.native.sampling.flow_schedule import build_sigmas


def test_matches_build_sigmas_for_every_schedule_knob():
    settings = {
        "shift": 3.0,
        "detail_strength": 0.1,
        "detail_start": 0.2,
        "detail_end": 0.8,
    }
    got = schedule_from_settings(12, settings, denoise_strength=0.6)
    want = build_sigmas(
        12, shift=3.0, denoise=0.6, detail_strength=0.1, detail_start=0.2, detail_end=0.8,
    )
    assert torch.equal(got, want)


def test_fixed_mu_changes_the_schedule():
    plain = schedule_from_settings(10, {"shift": 3.0})
    pinned = schedule_from_settings(10, {"shift": 3.0, "fixed_mu": 1.15})
    assert not torch.equal(plain, pinned)
    assert torch.equal(pinned, build_sigmas(10, shift=3.0, fixed_mu=1.15))


def test_conditioned_sigmas_is_the_same_schedule():
    settings = {"shift": 2.0, "fixed_mu": 0.9}
    assert torch.equal(
        conditioned_sigmas(8, settings, denoise_strength=0.5),
        schedule_from_settings(8, settings, denoise_strength=0.5),
    )


def test_fp32_trajectory_forward_casts_in_and_out():
    seen = {}

    def model_forward(x, sigma, conditioning):
        seen["x"], seen["sigma"] = x.dtype, sigma.dtype
        return torch.ones_like(x)

    wrapped = fp32_trajectory_forward(model_forward, torch.bfloat16)
    out = wrapped(torch.zeros(2, dtype=torch.float32), torch.ones(2, dtype=torch.float32), {})
    assert seen == {"x": torch.bfloat16, "sigma": torch.bfloat16}
    assert out.dtype == torch.float32


def test_warn_discarded_guidance_scale_is_silent_at_one(caplog):
    log = logging.getLogger("test.warn")
    with caplog.at_level(logging.WARNING, logger="test.warn"):
        warn_discarded_guidance_scale(log, "denoise", NoCFG(), 1.0)
    assert caplog.records == []


def test_warn_discarded_guidance_scale_names_the_caller(caplog):
    log = logging.getLogger("test.warn")
    with caplog.at_level(logging.WARNING, logger="test.warn"):
        warn_discarded_guidance_scale(log, "denoise_prenoised", NoCFG(), 4.0)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage().startswith("denoise_prenoised: guidance_override (NoCFG)")


def test_wrap_step_cache_is_a_no_op_without_options():
    guidance = NoCFG()
    wrapped, cache_set = wrap_step_cache(guidance, None, lambda *a: None, 10)
    assert wrapped is guidance and cache_set is None


def test_wrap_step_cache_wraps_when_enabled():
    guidance = NoCFG()
    wrapped, cache_set = wrap_step_cache(guidance, {"rel_threshold": 0.12}, lambda *a: None, 10)
    assert cache_set is not None and wrapped is not guidance
    assert wrapped.inner is guidance
    plain, none_set = wrap_step_cache(guidance, {"rel_threshold": 0.0}, lambda *a: None, 10)
    assert plain is guidance and none_set is None
