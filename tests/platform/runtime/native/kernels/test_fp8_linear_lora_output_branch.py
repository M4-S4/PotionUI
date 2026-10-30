from __future__ import annotations

import torch

from src.platform.runtime.native.kernels import lora_math
from src.platform.runtime.native.kernels.fp8_linear import _scaled_mm_fast_path_reject_reason
from src.platform.runtime.native.kernels.nvfp4_linear import _nvfp4_fast_path_reject_reason
from src.platform.settings.runtime_flags import runtime_flag_values


def test_deltas_output_branch_ok_accepts_full_width_and_dim0_slice():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    full = LoraDelta(down=torch.randn(4, 32), up=torch.randn(16, 4), alpha=4.0, scale=1.0)
    sliced = LoraDelta(
        down=torch.randn(3, 32), up=torch.randn(6, 3), alpha=2.0, scale=1.0,
        target_slice=(0, 5, 6),
    )
    assert lora_math._deltas_output_branch_ok([full], out_features=16) is True
    assert lora_math._deltas_output_branch_ok([sliced], out_features=16) is True
    assert lora_math._deltas_output_branch_ok([full, sliced], out_features=16) is True


def test_deltas_output_branch_ok_rejects_lokr_only_with_the_kill_switch(monkeypatch):
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    lokr = LoraDelta(down=torch.randn(4, 4), up=torch.randn(4, 4), alpha=4.0, scale=1.0, kron=True)
    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", True)
    assert lora_math._deltas_output_branch_ok([lokr], out_features=16) is True
    assert lora_math._deltas_output_branch_ok([lokr], out_features=32) is False
    monkeypatch.setitem(runtime_flag_values(), "native_lora_fused", False)
    assert lora_math._deltas_output_branch_ok([lokr], out_features=16) is False


def test_deltas_output_branch_ok_rejects_out_of_bounds_slice():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    oob = LoraDelta(
        down=torch.randn(3, 32), up=torch.randn(6, 3), alpha=2.0, scale=1.0,
        target_slice=(0, 12, 6),
    )
    assert lora_math._deltas_output_branch_ok([oob], out_features=16) is False


def test_scaled_mm_fast_path_reason_none_for_branch_ok_full_and_sliced_deltas():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    kwargs = dict(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    )
    full = LoraDelta(down=torch.randn(4, 32), up=torch.randn(32, 4), alpha=4.0, scale=1.0)
    sliced = LoraDelta(
        down=torch.randn(3, 32), up=torch.randn(6, 3), alpha=2.0, scale=1.0,
        target_slice=(0, 5, 6),
    )
    assert _scaled_mm_fast_path_reject_reason(lora_deltas=[full], **kwargs) is None
    assert _scaled_mm_fast_path_reject_reason(lora_deltas=[sliced], **kwargs) is None


def test_scaled_mm_fast_path_reason_still_lora_deltas_for_lokr():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    kwargs = dict(
        weight_dtype=torch.float8_e4m3fn, has_weight_scale=True,
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    )
    lokr = LoraDelta(down=torch.randn(4, 4), up=torch.randn(4, 4), alpha=4.0, scale=1.0, kron=True)
    assert _scaled_mm_fast_path_reject_reason(lora_deltas=[lokr], **kwargs) == "lora_deltas"


def test_nvfp4_fast_path_reason_none_for_branch_ok_sliced_delta():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    kwargs = dict(
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    )
    sliced = LoraDelta(
        down=torch.randn(3, 32), up=torch.randn(6, 3), alpha=2.0, scale=1.0,
        target_slice=(0, 5, 6),
    )
    assert _nvfp4_fast_path_reject_reason(lora_deltas=[sliced], **kwargs) is None


def test_nvfp4_fast_path_reason_still_lora_deltas_for_lokr():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    kwargs = dict(
        input_dtype=torch.bfloat16, input_is_cuda=True, weight_is_cuda=True,
        in_features=32, out_features=32,
    )
    lokr = LoraDelta(down=torch.randn(4, 4), up=torch.randn(4, 4), alpha=4.0, scale=1.0, kron=True)
    assert _nvfp4_fast_path_reject_reason(lora_deltas=[lokr], **kwargs) == "lora_deltas"


def test_forward_scaled_mm_lora_output_branch_matches_dequant_parity():
    from src.platform.runtime.native.lora.key_mapping import LoraDelta

    torch.manual_seed(0)
    out_features, in_features, m = 16, 32, 5
    weight = torch.randn(out_features, in_features, dtype=torch.float32)
    x = torch.randn(m, in_features, dtype=torch.float32)

    full = LoraDelta(down=torch.randn(4, in_features), up=torch.randn(out_features, 4), alpha=4.0, scale=1.3)
    sliced = LoraDelta(
        down=torch.randn(3, in_features), up=torch.randn(6, 3), alpha=2.0, scale=0.7,
        target_slice=(0, 5, 6),
    )
    deltas = [full, sliced]
    assert lora_math._deltas_output_branch_ok(deltas, out_features=out_features) is True

    expected_weight = lora_math.apply_lora_deltas(weight.clone(), deltas)
    expected = x @ expected_weight.T

    branch = lora_math._lora_output_branch(x, deltas, torch.float32, out_features)
    actual = x @ weight.T + branch

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
