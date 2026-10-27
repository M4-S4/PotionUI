from __future__ import annotations

import importlib
import logging
import math

import pytest
import torch

from src.platform.runtime.native.kernels import FLASH, SAGE, SAGE2, SAGE3, SDPA, SPARGE
from src.platform.settings import runtime_flags
from tests.fixtures.kernel_capabilities import ada, ampere, blackwell, cpu_only, fake_capabilities, fake_device, hopper

att = importlib.import_module("src.platform.runtime.native.kernels.attention")


@pytest.fixture(autouse=True)
def _reset_cache():
    att.reset_backend_cache()
    att.set_backend_override(None)
    yield
    att.reset_backend_cache()
    att.set_backend_override(None)


def _install(monkeypatch, install_capabilities, *names, device=ampere):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    install_capabilities(fake_capabilities(device(attention_backends=frozenset({SDPA, *names}))))


def test_sdpa_always_available_no_gpu(install_capabilities):
    install_capabilities(cpu_only())
    assert att.available_backends() == ["sdpa"]


def test_flash_is_listed_only_on_a_device_that_has_it(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    assert att.available_backends() == ["flash", "sdpa"]

    install_capabilities(cpu_only())
    assert att.available_backends() == ["sdpa"]


def test_sage2_outranks_sage(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SAGE)
    assert att.available_backends() == ["sage", "sdpa"]

    _install(monkeypatch, install_capabilities, SAGE, SAGE2)
    assert att.available_backends() == ["sage2", "sage", "sdpa"]


def test_full_stack_priority_order(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH, SAGE, SAGE2, device=hopper)
    assert att.available_backends() == ["sage2", "sage", "flash", "sdpa"]


HARDWARE_DECISIONS = [
    pytest.param(ampere, (FLASH, SAGE, SAGE2, SPARGE), ["sage2", "sage", "flash", "sdpa"], "sage2", "sparge", id="ampere"),
    pytest.param(ada, (FLASH, SAGE, SAGE2, SPARGE), ["sage2", "sage", "flash", "sdpa"], "sage2", "sparge", id="ada"),
    pytest.param(hopper, (FLASH, SAGE, SAGE2, SPARGE), ["sage2", "sage", "flash", "sdpa"], "sage2", "sparge", id="hopper"),
    pytest.param(
        blackwell, (FLASH, SAGE, SAGE2, SAGE3), ["sage3", "sage2", "sage", "flash", "sdpa"], "sage3", "sage3", id="blackwell",
    ),
    pytest.param(None, (), ["sdpa"], "sdpa", "sdpa", id="cpu"),
]


@pytest.mark.parametrize("device, installed, available, auto, sparge_pin", HARDWARE_DECISIONS)
def test_backend_decisions_follow_the_snapshot_of_each_generation(
    monkeypatch, install_capabilities, device, installed, available, auto, sparge_pin
):
    if device is None:
        install_capabilities(cpu_only())
    else:
        _install(monkeypatch, install_capabilities, *installed, device=device)
    assert att.available_backends() == available
    assert att.get_attention_backend() == auto
    assert att.get_attention_backend("sparge") == sparge_pin


def test_sage3_selected_on_blackwell(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SAGE3, device=blackwell)
    assert att.available_backends() == ["sage3", "sdpa"]
    assert att.get_attention_backend() == "sage3"


def test_sage3_outranks_sage2_in_priority(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SAGE3, SAGE, SAGE2, FLASH, device=blackwell)
    assert att.available_backends() == ["sage3", "sage2", "sage", "flash", "sdpa"]
    assert att.get_attention_backend() == "sage3"


def test_sage3_override_unavailable_warns_and_falls_back(install_capabilities, caplog):
    install_capabilities(cpu_only())
    with caplog.at_level(logging.WARNING):
        chosen = att.get_attention_backend("sage3")
    assert chosen == "sdpa"
    assert any("unavailable" in r.message and "sage3" in r.message for r in caplog.records)


def test_sage3_active_still_falls_back_to_sdpa_for_mask_or_fp32(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SAGE3, device=blackwell)
    assert att.get_attention_backend() == "sage3"

    q = torch.randn(1, 2, 4, 8)
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=False)
    assert torch.equal(att.attention(q, q, q), ref)

    q16 = torch.randn(1, 2, 4, 8, dtype=torch.float16)
    mask = torch.zeros(1, 2, 4, 4, dtype=torch.float16)
    ref16 = torch.nn.functional.scaled_dot_product_attention(q16, q16, q16, attn_mask=mask, is_causal=False)
    assert torch.equal(att.attention(q16, q16, q16, mask=mask), ref16)


def test_sparge_is_pinnable_but_never_listed_for_auto_selection(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SPARGE)
    assert att.get_attention_backend("sparge") == "sparge"
    assert "sparge" not in att.available_backends()

    _install(monkeypatch, install_capabilities, SPARGE, device=hopper)
    assert att.get_attention_backend("sparge") == "sparge"


def test_sparge_unavailable_warns_and_falls_back(monkeypatch, install_capabilities, caplog):
    _install(monkeypatch, install_capabilities)
    with caplog.at_level(logging.WARNING):
        chosen = att.get_attention_backend("sparge")
    assert chosen == "sdpa"
    assert any("unavailable" in r.message and "sparge" in r.message for r in caplog.records)


def test_sparge_never_auto_selected_even_when_only_backend_available(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SPARGE)
    assert att.available_backends() == ["sdpa"]
    assert att.get_attention_backend() == "sdpa"


def test_sparge_explicit_pin_is_honored_over_auto_selection(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SPARGE, SAGE, SAGE2, FLASH)
    assert att.get_attention_backend() != "sparge"
    att.set_backend_override("sparge")
    assert att.get_attention_backend() == "sparge"


def test_sparge_via_runtime_flag_is_honored(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SPARGE, device=hopper)
    monkeypatch.setitem(runtime_flags.runtime_flag_values(), "native_attention_backend", "sparge")
    assert att.get_attention_backend() == "sparge"


def test_sparge_active_falls_back_to_sdpa_for_mask_or_fp32(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SPARGE)
    att.set_backend_override("sparge")
    assert att.get_attention_backend() == "sparge"

    q = torch.randn(1, 2, 256, 64)
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=False)
    assert torch.equal(att.attention(q, q, q), ref)

    q16 = torch.randn(1, 2, 256, 64, dtype=torch.float16)
    mask = torch.zeros(1, 2, 256, 256, dtype=torch.float16)
    ref16 = torch.nn.functional.scaled_dot_product_attention(q16, q16, q16, attn_mask=mask, is_causal=False)
    assert torch.equal(att.attention(q16, q16, q16, mask=mask), ref16)


def test_sparge_active_falls_back_to_sdpa_for_short_sequence_or_bad_headdim(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SPARGE)
    att.set_backend_override("sparge")

    q_short = torch.randn(1, 2, 64, 64, dtype=torch.float16)
    ref_short = torch.nn.functional.scaled_dot_product_attention(q_short, q_short, q_short, is_causal=False)
    assert torch.equal(att.attention(q_short, q_short, q_short), ref_short)

    q_baddim = torch.randn(1, 2, 256, 96, dtype=torch.float16)
    ref_baddim = torch.nn.functional.scaled_dot_product_attention(q_baddim, q_baddim, q_baddim, is_causal=False)
    assert torch.equal(att.attention(q_baddim, q_baddim, q_baddim), ref_baddim)


def test_unknown_backend_warning_lists_pin_only_names(install_capabilities, caplog):
    install_capabilities(cpu_only())
    with caplog.at_level(logging.WARNING):
        att.get_attention_backend("totally-not-a-backend")
    assert any("sparge" in r.message for r in caplog.records if "unknown" in r.message.lower())


def test_known_backends_is_union_of_priority_and_pin_only():
    known = att.known_backends()
    assert known == frozenset(att.BACKEND_PRIORITY) | att.PIN_ONLY_BACKENDS
    assert "sparge" in known
    for name in att.BACKEND_PRIORITY:
        assert name in known


def test_default_picks_highest_priority(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SAGE, SAGE2, FLASH, device=hopper)
    assert att.get_attention_backend() == "sage2"


def test_override_wins_when_available(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    assert att.get_attention_backend("flash") == "flash"
    assert att.get_attention_backend("sdpa") == "sdpa"


def test_runtime_flag_selects_backend(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    monkeypatch.setitem(runtime_flags.runtime_flag_values(), "native_attention_backend", "flash")
    assert att.get_attention_backend() == "flash"
    assert att.get_backend_override() == "flash"


def test_override_arg_beats_runtime_flag(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    monkeypatch.setitem(runtime_flags.runtime_flag_values(), "native_attention_backend", "sdpa")
    assert att.get_attention_backend("flash") == "flash"


def test_unavailable_override_warns_and_falls_back(install_capabilities, caplog):
    install_capabilities(cpu_only())
    with caplog.at_level(logging.WARNING):
        chosen = att.get_attention_backend("flash")
    assert chosen == "sdpa"
    assert any("unavailable" in r.message and "flash" in r.message for r in caplog.records)


def test_unknown_backend_warns_and_falls_back(install_capabilities, caplog):
    install_capabilities(cpu_only())
    with caplog.at_level(logging.WARNING):
        chosen = att.get_attention_backend("nonsense")
    assert chosen == "sdpa"
    assert any("unknown" in r.message.lower() for r in caplog.records)


def test_fallback_warns_once_per_name(install_capabilities, caplog):
    install_capabilities(cpu_only())
    with caplog.at_level(logging.WARNING):
        att.get_attention_backend("flash")
        att.get_attention_backend("flash")
    flash_warnings = [r for r in caplog.records if "flash" in r.message]
    assert len(flash_warnings) == 1


def test_backend_override_used_when_no_override_arg_or_env(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    att.set_backend_override("flash")
    assert att.get_attention_backend() == "flash"
    assert att.get_backend_override() == "flash"


def test_native_attention_env_var_is_ignored(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    att.set_backend_override("flash")
    monkeypatch.setenv("NATIVE_ATTENTION", "sdpa")
    assert att.get_attention_backend() == "flash"


def test_override_arg_beats_backend_override(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    att.set_backend_override("sdpa")
    assert att.get_attention_backend("flash") == "flash"


@pytest.mark.parametrize("cleared", ["auto", "", None])
def test_backend_override_auto_and_empty_and_none_clear_pin(monkeypatch, install_capabilities, cleared):
    _install(monkeypatch, install_capabilities, SAGE, SAGE2, FLASH, device=hopper)
    att.set_backend_override("flash")
    assert att.get_backend_override() == "flash"
    att.set_backend_override(cleared)
    assert att.get_backend_override() is None
    assert att.get_attention_backend() == "sage2"


def test_backend_override_normalizes_case_and_whitespace(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    att.set_backend_override("  Flash  ")
    assert att.get_backend_override() == "flash"
    assert att.get_attention_backend() == "flash"


def _manual_attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask
    return scores.softmax(dim=-1) @ v


def test_sdpa_matches_manual_softmax(install_capabilities):
    install_capabilities(cpu_only())
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    k = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float64)
    out = att.attention(q, k, v)
    assert out.shape == (2, 3, 5, 8)
    assert torch.allclose(out, _manual_attention(q, k, v), atol=1e-12)


def test_sdpa_matches_manual_with_additive_mask(install_capabilities):
    install_capabilities(cpu_only())
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    v = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    mask = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
    mask[..., 3] = float("-inf")
    out = att.attention(q, k, v, mask=mask)
    assert torch.allclose(out, _manual_attention(q, k, v, mask), atol=1e-12)


def test_dispatch_bit_identical_to_raw_sdpa(install_capabilities):
    install_capabilities(cpu_only())
    torch.manual_seed(2)
    q = torch.randn(2, 4, 7, 16)
    k = torch.randn(2, 4, 7, 16)
    v = torch.randn(2, 4, 7, 16)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert torch.equal(att.attention(q, k, v), ref)


def test_heads_mismatch_raises():
    q = torch.randn(1, 4, 5, 8)
    with pytest.raises(ValueError):
        att.attention(q, q, q, heads=8)


def test_accelerated_backend_falls_back_to_sdpa_for_fp32_and_mask(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, SAGE, SAGE2, device=hopper)
    assert att.get_attention_backend() == "sage2"
    q = torch.randn(1, 2, 4, 8)
    ref = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=False)
    assert torch.equal(att.attention(q, q, q), ref)


def test_cross_backend_agreement_if_installed():
    att.reset_backend_cache()
    backends = att.available_backends()
    accel = [b for b in backends if b != "sdpa"]
    if not accel or not torch.cuda.is_available():
        pytest.skip(f"no accelerated attention kernels installed (have {backends})")

    torch.manual_seed(3)
    q = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
    ref = att.attention(q, k, v, backend="sdpa")
    for b in accel:
        try:
            out = att.attention(q, k, v, backend=b)
        except ImportError as exc:
            pytest.skip(f"{b} kernel module is installed but fails to import ({exc})")
        assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2), b


class _FakeCudaTensor:

    def __init__(self, device_index, dtype=torch.float16, shape=(1, 2, 4, 8)):
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = dtype
        self.is_cuda = True
        self.device = torch.device(f"cuda:{device_index}")


def test_availability_is_resolved_per_device(install_capabilities):
    install_capabilities(fake_capabilities(
        blackwell(attention_backends=frozenset({SDPA, SAGE3})),
        fake_device((7, 5)),
    ))

    assert "sage3" in att.available_backends(device_index=0)
    assert "sage3" not in att.available_backends(device_index=1)
    assert "sage3" in att.available_backends(device_index=0)
    assert att.get_attention_backend("sage3", device_index=0) == "sage3"
    assert att.get_attention_backend("sage3", device_index=1) == "sdpa"


def test_reset_backend_cache_refreshes_the_snapshot(monkeypatch):
    calls = []
    monkeypatch.setattr(att, "refresh_capabilities", lambda: calls.append("refresh"))

    att.reset_backend_cache()
    assert calls == ["refresh"]
    att.reset_backend_cache()
    assert calls == ["refresh", "refresh"]


def test_reset_backend_cache_replaces_the_installed_snapshot_with_a_rebuilt_one(monkeypatch, install_capabilities):
    _install(monkeypatch, install_capabilities, FLASH)
    assert att.available_backends() == ["flash", "sdpa"]

    att.reset_backend_cache()

    assert att.available_backends() == ["sdpa"]


def test_reset_backend_cache_rearms_the_one_shot_warning(install_capabilities, caplog):
    install_capabilities(cpu_only())
    with caplog.at_level(logging.WARNING):
        att.get_attention_backend("flash")
        att.get_attention_backend("flash")
        assert len([r for r in caplog.records if "flash" in r.message]) == 1
        att.reset_backend_cache()
        install_capabilities(cpu_only())
        att.get_attention_backend("flash")
    assert len([r for r in caplog.records if "flash" in r.message]) == 2


def test_attention_dispatch_validates_against_q_device_not_current_device(monkeypatch):
    seen_device_indices = []

    def fake_get_attention_backend(backend=None, device_index=None):
        seen_device_indices.append(device_index)
        return "sdpa"

    monkeypatch.setattr(att, "get_attention_backend", fake_get_attention_backend)
    monkeypatch.setattr(att, "_sdpa", lambda q, k, v, mask, grouped=False, causal=False: q)

    fake_q = _FakeCudaTensor(device_index=3)
    att.attention(fake_q, fake_q, fake_q)
    assert seen_device_indices == [3]

    fake_q_other = _FakeCudaTensor(device_index=0)
    att.attention(fake_q_other, fake_q_other, fake_q_other)
    assert seen_device_indices == [3, 0]


def test_attention_dispatch_device_index_is_none_for_cpu_tensors(monkeypatch):
    seen_device_indices = []

    def fake_get_attention_backend(backend=None, device_index=None):
        seen_device_indices.append(device_index)
        return "sdpa"

    monkeypatch.setattr(att, "get_attention_backend", fake_get_attention_backend)
    q = torch.randn(1, 2, 4, 8)
    att.attention(q, q, q)
    assert seen_device_indices == [None]


class _FakeSageModule:

    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []

    def sageattn(self, q, k, v, tensor_layout="HND", is_causal=False):
        self.calls.append({"q": q, "k": k, "v": v})
        return self.return_value


def _patch_sageattention(monkeypatch, fake_module):
    import sys

    monkeypatch.setitem(sys.modules, "sageattention", fake_module)


def test_sage_passes_small_v_through_to_the_kernel_unscaled(monkeypatch):
    v = torch.full((1, 2, 4, 8), 160.0)
    fake = _FakeSageModule(return_value=torch.zeros(1, 2, 4, 8))
    _patch_sageattention(monkeypatch, fake)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    att._sage(q, k, v)

    assert len(fake.calls) == 1
    called_v = fake.calls[0]["v"]
    assert torch.equal(called_v, v)
    assert torch.equal(fake.calls[0]["q"], q)
    assert torch.equal(fake.calls[0]["k"], k)


def test_sage_small_v_returns_kernel_output_without_postscale(monkeypatch):
    kernel_out = torch.full((1, 2, 4, 8), 3.0)
    fake = _FakeSageModule(return_value=kernel_out)
    _patch_sageattention(monkeypatch, fake)

    q = k = v = torch.randn(1, 2, 4, 8)
    out = att._sage(q, k, v)

    assert torch.allclose(out, kernel_out)


def test_sage_hot_v_engages_the_256_prescale(monkeypatch):
    v = torch.full((1, 2, 4, 8), 5000.0)
    kernel_out = torch.full((1, 2, 4, 8), 3.0)
    fake = _FakeSageModule(return_value=kernel_out)
    _patch_sageattention(monkeypatch, fake)

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    out = att._sage(q, k, v)

    assert len(fake.calls) == 1
    assert torch.equal(fake.calls[0]["v"], v / 256.0)
    assert torch.equal(fake.calls[0]["q"], q)
    assert torch.equal(fake.calls[0]["k"], k)
    assert torch.allclose(out, kernel_out * 256.0)


def test_sage_round_trips_a_linear_kernel_exactly_in_both_regimes(monkeypatch):

    def linear_in_v(q, k, v, tensor_layout="HND", is_causal=False):
        return v * 2.0

    fake = _FakeSageModule(return_value=None)
    fake.sageattn = linear_in_v
    _patch_sageattention(monkeypatch, fake)

    q = k = torch.randn(1, 2, 4, 8)
    for magnitude in (10.0, 5000.0):
        v = torch.randn(1, 2, 4, 8) * magnitude
        out = att._sage(q, k, v)
        assert torch.equal(out, v * 2.0), f"round-trip broke at magnitude {magnitude}"


def test_sage_output_is_never_nan_or_inf_even_if_the_kernel_returns_them(monkeypatch):
    poisoned = torch.tensor([[[[float("nan"), float("inf"), float("-inf"), 1.0]]]])
    fake = _FakeSageModule(return_value=poisoned)
    _patch_sageattention(monkeypatch, fake)

    q = k = v = torch.randn(1, 1, 1, 4)
    out = att._sage(q, k, v)

    assert torch.isfinite(out).all()
    assert out[0, 0, 0, 0].item() == 0.0
    assert out[0, 0, 0, 1].item() == 0.0
    assert out[0, 0, 0, 2].item() == 0.0
    assert out[0, 0, 0, 3].item() == pytest.approx(1.0)


def _large_row_tensors(magnitude: float):
    rows = att._SAGE_SYNC_ROWS
    q = torch.randn(1, 1, rows, 8)
    k = torch.randn(1, 1, rows, 8)
    v = torch.randn(1, 1, rows, 8) * magnitude
    return q, k, v


def test_sage_large_rows_small_v_skips_the_copy_entirely(monkeypatch):
    q, k, v = _large_row_tensors(10.0)
    fake = _FakeSageModule(return_value=torch.zeros_like(v))
    _patch_sageattention(monkeypatch, fake)

    att._sage(q, k, v)

    assert fake.calls[0]["v"] is v
    assert fake.calls[0]["q"] is q
    assert fake.calls[0]["k"] is k


def test_sage_large_rows_hot_v_still_prescales_exactly(monkeypatch):
    q, k, v = _large_row_tensors(5000.0)
    kernel_out = torch.full_like(v, 3.0)
    fake = _FakeSageModule(return_value=kernel_out)
    _patch_sageattention(monkeypatch, fake)

    out = att._sage(q, k, v)

    assert torch.equal(fake.calls[0]["v"], v / 256.0)
    assert torch.allclose(out, kernel_out * 256.0)


def test_sage_below_the_row_threshold_keeps_the_sync_free_path(monkeypatch):
    v = torch.full((1, 2, 4, 8), 160.0)
    fake = _FakeSageModule(return_value=torch.zeros(1, 2, 4, 8))
    _patch_sageattention(monkeypatch, fake)

    att._sage(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8), v)

    called_v = fake.calls[0]["v"]
    assert torch.equal(called_v, v)
    assert called_v is not v


def test_absmax_matches_abs_amax_without_the_full_size_temp():
    for t in (torch.randn(3, 5) * 100, -torch.rand(4, 4) - 5.0, torch.rand(2, 8) + 3.0):
        assert torch.equal(att._absmax(t), t.abs().amax())


def test_aligned_mask_rehomes_odd_length_rows_in_16_aligned_storage():
    mask = torch.randn(1, 1, 37, 37)
    out = att._aligned_mask(mask)
    assert out.shape == mask.shape
    assert torch.equal(out, mask)
    assert out.stride(-2) % att._MASK_ALIGN == 0
    even = torch.randn(1, 1, 32, 32)
    assert att._aligned_mask(even) is even
    assert att._aligned_mask(None) is None


def test_masked_grouped_sdpa_matches_expanded_reference():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 37, 8)
    k = torch.randn(1, 2, 37, 8)
    v = torch.randn(1, 2, 37, 8)
    mask = torch.zeros(1, 1, 37, 37)
    mask[..., :5, 20:] = float("-inf")
    out = att._sdpa(q, k, v, mask, grouped=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), attn_mask=mask,
    )
    assert torch.allclose(out, ref, atol=1e-6)


def test_small_masked_grouped_sdpa_keeps_kv_unexpanded(monkeypatch):
    seen = []
    real_repeat = torch.Tensor.repeat_interleave
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", lambda self, *a, **kw: (seen.append(tuple(self.shape)), real_repeat(self, *a, **kw))[1])
    q = torch.randn(1, 4, 37, 8)
    k = torch.randn(1, 2, 37, 8)
    v = torch.randn(1, 2, 37, 8)
    mask = torch.zeros(1, 1, 37, 37)
    att._sdpa(q, k, v, mask, grouped=True)
    assert seen == []


def test_large_masked_grouped_sdpa_expands_kv(monkeypatch):
    monkeypatch.setattr(att, "_MASKED_GROUPED_EXPAND_MIN_SCORES", 0)
    seen = []
    real_repeat = torch.Tensor.repeat_interleave
    monkeypatch.setattr(torch.Tensor, "repeat_interleave", lambda self, *a, **kw: (seen.append(tuple(self.shape)), real_repeat(self, *a, **kw))[1])
    q = torch.randn(1, 4, 37, 8)
    k = torch.randn(1, 2, 37, 8)
    v = torch.randn(1, 2, 37, 8)
    mask = torch.zeros(1, 1, 37, 37)
    att._sdpa(q, k, v, mask, grouped=True)
    assert seen == [(1, 2, 37, 8), (1, 2, 37, 8)]
