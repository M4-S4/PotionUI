from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.platform.runtime.native.arch.krea2.model import Krea2
from src.platform.runtime.native.arch.krea2.text_fusion import (
    CHUNK_PROFILE,
    chunk_gains,
    enhanced_fusion,
    fusion_key,
    scaled_input,
)
from src.platform.runtime.native.base import load_into_module
from src.platform.runtime.native.detect.registry import match_model_spec
from src.platform.runtime.native.engine import NativeGenerator, NativeModel
from src.platform.runtime.native.lora.step_window import LoraStepWindow, LoraStepWindowHook
from src.platform.runtime.native.kernels import pick_operations

TINY = {
    "image_model": "krea2", "features": 32, "heads": 2, "kvheads": 1,
    "channels": 4, "layers": 1, "multiplier": 1, "tdim": 16, "txtdim": 16,
    "txtheads": 2, "txtkvheads": 2, "txtlayers": 12, "patch": 2, "theta": 1000.0,
}
TAPS = 12
HALF = 8

NON_UNIT = {
    (3, 1): 2.5, (4, 0): 5.0, (4, 1): 1.1, (5, 0): 4.0,
    (9, 1): 2.5, (10, 0): 5.0, (10, 1): 1.1, (11, 0): 4.0,
}


def _randomised_state_dict(module) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for k, v in module.state_dict().items():
        if k.endswith(".scale") or k.endswith(".mod.lin") or k.endswith(".modulation.lin"):
            sd[k] = torch.zeros_like(v)
        elif v.is_floating_point():
            sd[k] = torch.randn_like(v) * 0.02
        else:
            sd[k] = v.clone()
    return sd


def _build_ready(config=None) -> Krea2:
    torch.manual_seed(0)
    config = config or TINY
    m = Krea2.from_config(config, pick_operations(torch.float32, torch.float32))
    load_into_module(m, _randomised_state_dict(m), match_model_spec(config))
    m.eval()
    return m


def _inputs(seed=0, txt_len=5, neg_txt_len=4):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 4, 8, 8, generator=g)
    te_hidden = torch.randn(1, txt_len, TAPS, 16, generator=g)
    neg_hidden = torch.randn(1, neg_txt_len, TAPS, 16, generator=g)
    return x, te_hidden, neg_hidden


def _expected_scaled(x: torch.Tensor, strength: float) -> torch.Tensor:
    out = x.clone()
    global_gain = 1.0 + 14.0 * strength
    for tap in range(TAPS):
        for half in range(2):
            gain = (1.0 + strength * (NON_UNIT.get((tap, half), 1.0) - 1.0)) * global_gain
            out[:, :, tap, half * HALF:(half + 1) * HALF] *= gain
    return out


def _expected_blend(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    ref, cand = reference.double(), candidate.double()
    out = torch.empty_like(ref)
    for b in range(ref.shape[0]):
        for t in range(ref.shape[1]):
            r, c = ref[b, t], cand[b, t]
            delta = c - r
            rel = delta.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()
            out[b, t] = r + delta * min(1.0, 0.75 / float(rel))
    return out


def _token_ratio(reference: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    delta = (other - reference).double()
    return delta.pow(2).mean(-1).sqrt() / reference.double().pow(2).mean(-1).sqrt()


def _run_scope(module):
    dit = NativeModel("diffusion_model", module)
    return dit, NativeGenerator._run_cache(SimpleNamespace(dit=dit))


class _Prepared:
    def __init__(self, module: Krea2) -> None:
        self.calls = 0
        real = module.prepare_context

        def counted(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        module.prepare_context = counted


T = torch.tensor([0.5])


@pytest.mark.parametrize("off", [None, {}, {"strength": 0.0, "scale": 1.0}, {"strength": 0.0}])
def test_a_every_spelling_of_off_leaves_the_forward_byte_identical(off):
    module = _build_ready()
    x, te_hidden, neg = _inputs(seed=3)
    nag = {"scale": 2.0, "tau": 3.5, "alpha": 0.5}

    plain = module(x, T, te_hidden)
    plain_nag = module(x, T, te_hidden, nag_context=neg, nag=nag)

    assert torch.equal(module(x, T, te_hidden, text_fusion=off), plain)
    assert torch.equal(module(x, T, te_hidden, nag_context=neg, nag=nag, text_fusion=off), plain_nag)


def test_a_off_prepare_context_is_the_plain_fusion_composition():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=5)
    mask = torch.tensor([[True, True, True, False, False]])

    by_hand = module.txtmlp(module.txtfusion(te_hidden, mask=mask[:, None, None, :]))

    assert torch.equal(module.prepare_context(te_hidden, mask), by_hand)
    assert torch.equal(module.prepare_context(te_hidden, mask, {"strength": 0.0, "scale": 1.0}), by_hand)


def test_b_the_profile_is_twelve_entries_tiled_over_twenty_four_chunks():
    assert len(CHUNK_PROFILE) == 24
    assert CHUNK_PROFILE[:12] == CHUNK_PROFILE[12:]
    assert [c for c, g in enumerate(CHUNK_PROFILE) if g != 1.0] == [7, 8, 9, 10, 19, 20, 21, 22]


def test_b_chunk_gains_follow_the_documented_formula():
    gains = chunk_gains(0.5)
    assert gains.shape == (24,)
    assert gains[8].item() == pytest.approx((1.0 + 0.5 * 4.0) * 8.0)
    assert gains[0].item() == pytest.approx(1.0 * 8.0)
    assert torch.allclose(chunk_gains(0.0), torch.ones(24))
    assert chunk_gains(1.0)[22].item() == pytest.approx(4.0 * 15.0)


@pytest.mark.parametrize("strength", [0.3, 1.0, 2.0])
def test_b_the_gains_land_on_the_right_tap_and_feature_half(strength):
    _, te_hidden, _ = _inputs(seed=7)

    got = scaled_input(te_hidden, strength)

    assert got.shape == te_hidden.shape
    assert torch.allclose(got, _expected_scaled(te_hidden, strength), rtol=1e-5, atol=1e-6)


def test_b_the_layout_survives_a_non_contiguous_context():
    g = torch.Generator().manual_seed(9)
    stacked = torch.randn(1, TAPS, 5, 16, generator=g)
    context = stacked.movedim(1, 2)
    assert not context.is_contiguous()

    assert torch.allclose(scaled_input(context, 1.0), _expected_scaled(context.contiguous(), 1.0),
                          rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strength", [0.01, 0.3, 1.0, 2.0])
def test_c_the_blend_matches_an_independent_per_token_reference(strength):
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=11)
    mask = torch.tensor([[True, True, True, True, False]])[:, None, None, :]
    reference = module.txtfusion(te_hidden, mask=mask)
    candidate = module.txtfusion(_expected_scaled(te_hidden, strength), mask=mask)

    got = enhanced_fusion(module.txtfusion, te_hidden, reference, mask, strength)

    assert torch.allclose(got.double(), _expected_blend(reference, candidate), rtol=1e-4, atol=1e-5)


def test_c_the_cap_bounds_every_token_shift_when_the_candidate_runs_away():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=13)
    reference = module.txtfusion(te_hidden)
    candidate = module.txtfusion(_expected_scaled(te_hidden, 2.0))
    assert (_token_ratio(reference, candidate) > 0.75).all(), "fixture must exercise the capped regime"

    got = enhanced_fusion(module.txtfusion, te_hidden, reference, None, 2.0)

    ratio = _token_ratio(reference, got)
    assert torch.allclose(ratio, torch.full_like(ratio, 0.75), atol=1e-4)


def test_c_a_shift_under_the_cap_passes_through_unchanged():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=15)
    reference = module.txtfusion(te_hidden)
    candidate = module.txtfusion(_expected_scaled(te_hidden, 0.002))
    assert (_token_ratio(reference, candidate) < 0.75).all(), "fixture must exercise the uncapped regime"

    got = enhanced_fusion(module.txtfusion, te_hidden, reference, None, 0.002)

    assert torch.allclose(got, candidate, rtol=1e-5, atol=1e-6)
    assert not torch.equal(got, reference)


def test_d_the_blend_is_cast_back_to_the_activation_dtype_before_txtmlp():
    module = _build_ready().to(torch.bfloat16)
    _, te_hidden, _ = _inputs(seed=17)
    te_hidden = te_hidden.to(torch.bfloat16)
    seen: list[torch.dtype] = []
    module.txtmlp.register_forward_pre_hook(lambda _m, args: seen.append(args[0].dtype))

    plain = module.prepare_context(te_hidden)
    enhanced = module.prepare_context(te_hidden, None, {"strength": 1.0, "scale": 1.5})

    assert seen == [torch.bfloat16, torch.bfloat16]
    assert plain.dtype == enhanced.dtype == torch.bfloat16
    assert torch.isfinite(enhanced.float()).all()


def _heavy_context(seed, dtype):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 5, TAPS, 16, generator=g) * 4000.0).to(dtype)


@pytest.mark.parametrize("strength", [1.0, 2.0])
def test_d_a_float16_context_of_large_magnitude_stays_finite(strength):
    module = _build_ready().to(torch.float16)
    te_hidden = _heavy_context(43, torch.float16)
    assert torch.isfinite(module.prepare_context(te_hidden)).all()

    got = module.prepare_context(te_hidden, None, {"strength": strength, "scale": 1.5})

    assert got.dtype == torch.float16
    assert torch.isfinite(got).all()


def test_d_the_scaled_input_saturates_instead_of_overflowing_in_float16():
    x = _heavy_context(45, torch.float16)

    got = scaled_input(x, 2.0)

    assert got.dtype == torch.float16
    assert torch.isfinite(got).all()
    assert got.abs().max().item() == torch.finfo(torch.float16).max


def test_d_the_scaling_is_computed_in_float32_before_the_cast():
    g = torch.Generator().manual_seed(46)
    x = (torch.randn(1, 5, TAPS, 16, generator=g) * 3.0).to(torch.float16)

    got = scaled_input(x, 0.05)

    want = _expected_scaled(x.float(), 0.05).to(torch.float16)
    assert torch.allclose(got.float(), want.float(), rtol=2e-3, atol=1e-3)


def test_d_a_token_with_a_non_finite_candidate_falls_back_to_the_reference():
    g = torch.Generator().manual_seed(47)
    x = torch.randn(1, 4, TAPS, 16, generator=g)
    reference = torch.randn(1, 4, 16, generator=g)
    candidate = reference * 1.2
    candidate[0, 1, 3] = float("nan")
    candidate[0, 2, 0] = float("inf")

    got = enhanced_fusion(lambda _x, mask=None: candidate.clone(), x, reference, None, 1.0)

    assert torch.isfinite(got).all()
    assert torch.equal(got[0, 1], reference[0, 1])
    assert torch.equal(got[0, 2], reference[0, 2])
    for token in (0, 3):
        assert torch.allclose(got[0, token], candidate[0, token], rtol=1e-5, atol=1e-6)


def test_d_a_token_whose_shift_overflows_falls_back_to_the_reference():
    g = torch.Generator().manual_seed(48)
    x = torch.randn(1, 3, TAPS, 16, generator=g)
    reference = torch.randn(1, 3, 16, generator=g)
    candidate = reference * 1.1
    reference[0, 0] = -3.0e38
    candidate[0, 0] = 3.0e38

    got = enhanced_fusion(lambda _x, mask=None: candidate.clone(), x, reference, None, 1.0)

    assert torch.isfinite(got).all()
    assert torch.equal(got[0, 0], reference[0, 0])


def test_e_text_scale_multiplies_the_txtmlp_output_exactly():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=21)

    plain = module.prepare_context(te_hidden)

    assert torch.equal(module.prepare_context(te_hidden, None, {"strength": 0.0, "scale": 1.5}), plain * 1.5)
    assert torch.equal(module.prepare_context(te_hidden, None, {"scale": 0.5}), plain * 0.5)


def test_e_the_text_scale_alone_runs_a_single_fusion_pass():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=24)
    passes = {"n": 0}
    real = module.txtfusion.forward

    def counted(*args, **kwargs):
        passes["n"] += 1
        return real(*args, **kwargs)

    module.txtfusion.forward = counted

    module.prepare_context(te_hidden, None, {"strength": 0.0, "scale": 2.0})
    assert passes["n"] == 1
    module.prepare_context(te_hidden, None, {"strength": 0.5, "scale": 2.0})
    assert passes["n"] == 3


def test_e_text_scale_applies_after_the_enhancer():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=23)

    enhanced = module.prepare_context(te_hidden, None, {"strength": 1.0, "scale": 1.0})
    scaled = module.prepare_context(te_hidden, None, {"strength": 1.0, "scale": 2.0})

    assert not torch.equal(enhanced, module.prepare_context(te_hidden))
    assert torch.equal(scaled, enhanced * 2.0)


def test_e_a_neutral_scale_with_a_live_enhancer_matches_the_hand_composition():
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=25)
    reference = module.txtfusion(te_hidden)
    candidate = module.txtfusion(_expected_scaled(te_hidden, 0.4))

    got = module.prepare_context(te_hidden, None, {"strength": 0.4, "scale": 1.0})

    want = module.txtmlp(_expected_blend(reference, candidate).float())
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_e_strength_and_scale_each_move_the_forward_output():
    module = _build_ready()
    x, te_hidden, _ = _inputs(seed=27)
    plain = module(x, T, te_hidden)

    low = module(x, T, te_hidden, text_fusion={"strength": 0.25, "scale": 1.0})
    high = module(x, T, te_hidden, text_fusion={"strength": 1.0, "scale": 1.0})
    scaled = module(x, T, te_hidden, text_fusion={"strength": 0.0, "scale": 2.0})

    for out in (low, high, scaled):
        assert torch.isfinite(out).all()
        assert not torch.equal(out, plain)
    assert (high - plain).abs().mean() > (low - plain).abs().mean()


def test_f_the_positive_and_the_nag_branch_are_both_enhanced():
    module = _build_ready()
    x, te_hidden, neg = _inputs(seed=29)
    nag = {"scale": 2.0, "tau": 3.5, "alpha": 0.5}
    fusion = {"strength": 1.0, "scale": 1.5}

    plain = module._branch_inputs(x, te_hidden, None, [], neg, None, nag)
    got = module._branch_inputs(x, te_hidden, None, [], neg, None, nag, fusion)

    assert torch.equal(got.fused, module.prepare_context(te_hidden, None, fusion))
    assert torch.equal(got.nag_fused, module.prepare_context(neg, None, fusion))
    assert not torch.equal(got.fused, plain.fused)
    assert not torch.equal(got.nag_fused, plain.nag_fused)


def test_g_equal_settings_hit_the_cache_and_a_changed_setting_never_does():
    module = _build_ready()
    x, te_hidden, _ = _inputs(seed=31)
    want_off = module(x, T, te_hidden)
    want_a = module(x, T, te_hidden, text_fusion={"strength": 1.0, "scale": 1.0})
    want_b = module(x, T, te_hidden, text_fusion={"strength": 0.5, "scale": 1.0})
    want_c = module(x, T, te_hidden, text_fusion={"strength": 0.5, "scale": 2.0})
    assert not torch.equal(want_a, want_b) and not torch.equal(want_b, want_c)

    counts = _Prepared(module)
    _, scope = _run_scope(module)
    with scope:
        assert torch.equal(module(x, T, te_hidden, text_fusion={"strength": 1.0, "scale": 1.0}), want_a)
        assert torch.equal(module(x, T, te_hidden, text_fusion={"strength": 1.0, "scale": 1.0}), want_a)
        assert counts.calls == 1
        assert torch.equal(module(x, T, te_hidden, text_fusion={"strength": 0.5, "scale": 1.0}), want_b)
        assert counts.calls == 2
        assert torch.equal(module(x, T, te_hidden, text_fusion={"strength": 0.5, "scale": 2.0}), want_c)
        assert counts.calls == 3
        assert torch.equal(module(x, T, te_hidden), want_off)
        assert counts.calls == 4


def test_g_the_enhanced_fusion_is_prepared_once_per_branch_across_steps():
    module = _build_ready()
    x, te_hidden, neg = _inputs(seed=33)
    fusion = {"strength": 1.0, "scale": 1.5}
    nag = {"scale": 2.0, "tau": 3.5, "alpha": 0.5}
    passes = {"n": 0}
    real = module.txtfusion.forward

    def counted(*args, **kwargs):
        passes["n"] += 1
        return real(*args, **kwargs)

    module.txtfusion.forward = counted
    _, scope = _run_scope(module)
    with scope:
        for s in (0.9, 0.6, 0.3):
            module(x * s, torch.tensor([s]), te_hidden, nag_context=neg, nag=nag, text_fusion=fusion)

    assert passes["n"] == 4


def _txtmlp_lora(rank=4, seed=1, scale=0.4):
    g = torch.Generator().manual_seed(seed)
    stem = "lora_unet_txtmlp_1"
    return {
        f"{stem}.lora_up.weight": torch.randn(32, rank, generator=g) * scale,
        f"{stem}.lora_down.weight": torch.randn(rank, 16, generator=g) * scale,
        f"{stem}.alpha": torch.tensor(float(rank)),
    }


def test_g_a_cached_enhanced_fusion_cannot_survive_a_lora_window_edge():
    module = _build_ready()
    x, te_hidden, _ = _inputs(seed=35)
    fusion = {"strength": 1.0, "scale": 1.5}
    dit, scope = _run_scope(module)
    hook = LoraStepWindowHook(dit, [(_txtmlp_lora(), 1.0, LoraStepWindow(3, 4))])
    counts = _Prepared(module)
    try:
        with scope:
            hook.on_start(6)
            unpatched = module(x, T, te_hidden, text_fusion=fusion)
            assert torch.equal(module(x, T, te_hidden, text_fusion=fusion), unpatched)
            assert counts.calls == 1

            hook.on_step(1, 6, None, 0.0, None)
            patched = module(x, T, te_hidden, text_fusion=fusion)
            assert counts.calls == 2
            assert not torch.equal(patched, unpatched)

            hook.on_step(3, 6, None, 0.0, None)
            restored = module(x, T, te_hidden, text_fusion=fusion)
            assert counts.calls == 3
            assert torch.allclose(restored, unpatched, atol=1e-6)
    finally:
        hook.close()


@pytest.mark.parametrize("bad", [
    {"strength": -0.1}, {"strength": float("nan")}, {"strength": float("inf")},
    {"scale": 0.0}, {"scale": -1.0}, {"scale": float("nan")},
])
def test_h_unusable_settings_are_rejected(bad):
    module = _build_ready()
    _, te_hidden, _ = _inputs(seed=37)

    with pytest.raises(ValueError):
        fusion_key(bad)
    with pytest.raises(ValueError):
        module.prepare_context(te_hidden, None, bad)


def test_h_the_key_is_none_when_neutral_and_the_normalised_pair_otherwise():
    assert fusion_key(None) is None
    assert fusion_key({}) is None
    assert fusion_key({"strength": 0.0, "scale": 1.0}) is None
    assert fusion_key({"strength": "0.5", "scale": 2}) == (0.5, 2.0)
    assert fusion_key({"strength": 0.0, "scale": 1.5}) == (0.0, 1.5)


def test_h_the_enhancer_refuses_a_tap_axis_it_has_no_profile_for():
    three_taps = torch.randn(1, 5, 3, 16)
    with pytest.raises(ValueError, match="12"):
        scaled_input(three_taps, 1.0)

    odd_width = torch.randn(1, 5, TAPS, 15)
    with pytest.raises(ValueError, match="even"):
        scaled_input(odd_width, 1.0)


def test_h_text_scale_alone_does_not_need_the_twelve_tap_layout():
    config = {**TINY, "txtlayers": 3}
    module = _build_ready(config)
    te_hidden = torch.randn(1, 5, 3, 16)

    scaled = module.prepare_context(te_hidden, None, {"strength": 0.0, "scale": 2.0})

    assert torch.equal(scaled, module.prepare_context(te_hidden) * 2.0)
    with pytest.raises(ValueError):
        module.prepare_context(te_hidden, None, {"strength": 1.0, "scale": 1.0})
