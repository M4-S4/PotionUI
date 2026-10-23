from __future__ import annotations

import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import src.platform.runtime.native.arch.minimax_h3.model as model_module
import vendor.gpl.comfyui.ops as wo
from src.platform.runtime.native.arch.minimax_h3.model import (
    MiniMaxH3MLP,
    default_ffn_chunk_rows,
    effective_mlp_chunk_rows,
)
from src.platform.runtime.native.lora.key_mapping import LoraDelta
from vendor.gpl.comfyui.ops import disable_weight_init, fp8_ops

from .._quant_layouts import int8_state_dict
from .test_minimax_h3_model import TINY_FULL, _build_ready, _fbcache_forward, _fbcache_inputs, _tiny_layout
from .test_minimax_h3_prepared_linear import _load_fp8

_H3_HIDDEN = 5376
_H3_FFN = 14336
_HIDDEN = 16
_FFN = 24
_ROWS = 22


class _OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func)
        return func(*args, **(kwargs or {}))


def _float_mlp(dtype: torch.dtype = torch.float32) -> MiniMaxH3MLP:
    mlp = MiniMaxH3MLP(_HIDDEN, _FFN, disable_weight_init, dtype=dtype)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for linear in (mlp.fc1, mlp.fc2):
            linear.weight.copy_((torch.randn(linear.weight.shape, generator=g) * 0.2).to(dtype))
    return mlp


def _fp8_mlp() -> MiniMaxH3MLP:
    mlp = MiniMaxH3MLP(_HIDDEN, _FFN, fp8_ops, dtype=torch.float32)
    _load_fp8(mlp.fc1, 2 * _FFN, _HIDDEN, seed=11)
    _load_fp8(mlp.fc2, _HIDDEN, _FFN, seed=12)
    return mlp


def _int8_convrot_mlp() -> MiniMaxH3MLP:
    ffn = 16
    mlp = MiniMaxH3MLP(_HIDDEN, ffn, fp8_ops, dtype=torch.float32)
    for name, out_f, in_f, seed in (("fc1", 2 * ffn, _HIDDEN, 21), ("fc2", _HIDDEN, ffn, 22)):
        sd, _w, _codes, _scale = int8_state_dict(
            out_f, in_f, prefix=f"{name}.", convrot=True, groupsize=16, seed=seed,
        )
        layer = getattr(mlp, name)
        layer._load_from_state_dict(dict(sd), f"{name}.", {}, True, [], [], [])
        assert layer.convrot_hadamard is not None
    return mlp


def _plain_delta(out_f: int, in_f: int, seed: int, rank: int = 4) -> LoraDelta:
    g = torch.Generator().manual_seed(seed)
    return LoraDelta(
        down=torch.randn(rank, in_f, generator=g) * 0.2,
        up=torch.randn(out_f, rank, generator=g) * 0.2,
        alpha=8.0, scale=0.75,
    )


def _lokr_delta(seed: int) -> LoraDelta:
    g = torch.Generator().manual_seed(seed)
    return LoraDelta(
        down=torch.randn(12, 8, generator=g) * 0.2,
        up=torch.randn(4, 2, generator=g) * 0.2,
        alpha=1.0, scale=0.5, kron=True,
    )


def _x(dtype: torch.dtype = torch.float32, hidden: int = _HIDDEN) -> torch.Tensor:
    return torch.randn(1, _ROWS, hidden, generator=torch.Generator().manual_seed(7)).to(dtype)


def _run(mlp: MiniMaxH3MLP, x: torch.Tensor, rows: int) -> torch.Tensor:
    with torch.inference_mode():
        return mlp(x, rows)


@pytest.fixture
def dequant_only(monkeypatch):
    monkeypatch.setitem(wo.RUNTIME_FLAGS, "native_fp8_matmul", False)


def test_default_rows_split_the_h3_ffn_into_six_chunks_on_long_sequences():
    for seq_len in (39_800, 103_000):
        rows = default_ffn_chunk_rows(seq_len, _H3_HIDDEN, _H3_FFN)
        assert rows == math.ceil(seq_len / 6)
        assert rows * 2 * _H3_FFN <= seq_len * _H3_HIDDEN


def test_default_rows_never_drop_below_the_floor_and_short_sequences_stay_whole():
    assert default_ffn_chunk_rows(4096, _H3_HIDDEN, _H3_FFN) == 0
    assert default_ffn_chunk_rows(10_000, _H3_HIDDEN, _H3_FFN) == 4096
    assert default_ffn_chunk_rows(24_576, _H3_HIDDEN, _H3_FFN) == 4096
    assert default_ffn_chunk_rows(24_577, _H3_HIDDEN, _H3_FFN) == 4097


def test_an_ffn_no_wider_than_the_hidden_state_is_never_chunked():
    assert default_ffn_chunk_rows(100_000, 64, 32) == 0


def test_the_mlp_takes_the_smaller_of_two_nonzero_chunk_widths():
    assert effective_mlp_chunk_rows(0, 0) == 0
    assert effective_mlp_chunk_rows(0, 7) == 7
    assert effective_mlp_chunk_rows(5, 0) == 5
    assert effective_mlp_chunk_rows(5, 7) == 5
    assert effective_mlp_chunk_rows(9, 7) == 7


@pytest.mark.parametrize("rows", [4, 5, 11])
def test_fp32_chunked_mlp_matches_the_whole_one(rows):
    mlp = _float_mlp()
    x = _x()
    torch.testing.assert_close(_run(mlp, x, rows), _run(mlp, x, 0), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("rows", [4, 5, 11])
def test_bf16_chunked_mlp_matches_the_whole_one(rows):
    mlp = _float_mlp(torch.bfloat16)
    x = _x(torch.bfloat16)
    chunked = _run(mlp, x, rows)
    assert chunked.dtype == torch.bfloat16
    torch.testing.assert_close(chunked, _run(mlp, x, 0))


def test_fp8_chunked_mlp_matches_the_whole_one(dequant_only):
    mlp = _fp8_mlp()
    x = _x()
    torch.testing.assert_close(_run(mlp, x, 5), _run(mlp, x, 0), atol=1e-6, rtol=1e-6)


def test_int8_convrot_chunked_mlp_matches_the_whole_one(dequant_only):
    mlp = _int8_convrot_mlp()
    x = _x()
    torch.testing.assert_close(_run(mlp, x, 5), _run(mlp, x, 0), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("build", [_fp8_mlp, _int8_convrot_mlp])
def test_a_quantized_mlp_dequantizes_each_weight_once_per_call_not_once_per_chunk(build, dequant_only, monkeypatch):
    mlp = build()
    calls = {"fc1": 0, "fc2": 0}
    for name in calls:
        layer = getattr(mlp, name)
        real = layer._prepare_dequant_operand

        def counting(*args, _name=name, _real=real, **kwargs):
            calls[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(layer, "_prepare_dequant_operand", counting)

    _run(mlp, _x(), 5)

    assert calls == {"fc1": 1, "fc2": 1}


def test_the_chunked_mlp_writes_into_one_output_instead_of_concatenating(dequant_only):
    mlp = _fp8_mlp()
    x = _x()
    log = _OpLog()
    with torch.inference_mode(), log:
        mlp(x, 5)
    assert torch.ops.aten.cat.default not in log.ops


def test_lora_deltas_ride_the_activation_side_of_every_chunk(dequant_only, monkeypatch):
    mlp = _fp8_mlp()
    x = _x()
    plain_only = _run(mlp, x, 0)
    mlp.fc1.lora_deltas = [_plain_delta(2 * _FFN, _HIDDEN, seed=41)]
    mlp.fc2.lora_deltas = [_plain_delta(_HIDDEN, _FFN, seed=42)]
    folded: list[int] = []
    real_apply = wo.apply_lora_deltas

    def recording_apply(weight, deltas):
        folded.append(len(deltas or []))
        return real_apply(weight, deltas)

    monkeypatch.setattr(wo, "apply_lora_deltas", recording_apply)

    whole = _run(mlp, x, 0)
    chunked = _run(mlp, x, 5)

    assert not torch.allclose(whole, plain_only, atol=1e-3)
    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-6)
    assert folded and set(folded) == {0}


def test_a_delta_with_no_activation_side_form_is_still_folded_into_the_prepared_weight(dequant_only, monkeypatch):
    monkeypatch.setitem(wo.RUNTIME_FLAGS, "native_lora_fused", False)
    mlp = _fp8_mlp()
    x = _x()
    mlp.fc1.lora_deltas = [_plain_delta(2 * _FFN, _HIDDEN, seed=43), _lokr_delta(seed=44)]
    folded: list[int] = []
    real_apply = wo.apply_lora_deltas

    def recording_apply(weight, deltas):
        folded.append(len(deltas or []))
        return real_apply(weight, deltas)

    monkeypatch.setattr(wo, "apply_lora_deltas", recording_apply)

    whole = _run(mlp, x, 0)
    folded_whole = list(folded)
    folded.clear()
    chunked = _run(mlp, x, 5)

    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-6)
    assert folded == folded_whole == [1, 0]


def test_the_model_chunks_its_mlps_by_default_and_matches_the_whole_forward(monkeypatch):
    monkeypatch.setattr(model_module, "_FFN_CHUNK_MIN_ROWS", 1)
    seen: list[int] = []
    real_forward = MiniMaxH3MLP.forward

    def recording_forward(self, x, seq_chunk_rows=0):
        seen.append(seq_chunk_rows)
        return real_forward(self, x, seq_chunk_rows)

    monkeypatch.setattr(MiniMaxH3MLP, "forward", recording_forward)
    m = _build_ready(TINY_FULL)
    layout = _tiny_layout(text_n=2, video_n=13, audio_n=7)
    torch.manual_seed(44)
    inputs = _fbcache_inputs(TINY_FULL, layout)
    ts = torch.tensor([0.2, 0.9])

    by_default = _fbcache_forward(m, layout, inputs, ts)
    default_rows = [rows for rows in seen if rows]
    seen.clear()
    whole = _fbcache_forward(m, layout, inputs, ts, ffn_chunk_rows=0)

    assert default_rows == [11] * TINY_FULL["num_layers"]
    assert not any(seen)
    for a, b in zip(by_default, whole):
        assert torch.equal(a, b)


def test_a_smaller_seq_chunk_width_also_bounds_the_mlp(monkeypatch):
    seen: list[int] = []
    real_forward = MiniMaxH3MLP.forward

    def recording_forward(self, x, seq_chunk_rows=0):
        seen.append(seq_chunk_rows)
        return real_forward(self, x, seq_chunk_rows)

    monkeypatch.setattr(MiniMaxH3MLP, "forward", recording_forward)
    m = _build_ready(TINY_FULL)
    layout = _tiny_layout(text_n=2, video_n=13, audio_n=7)
    torch.manual_seed(45)
    inputs = _fbcache_inputs(TINY_FULL, layout)
    ts = torch.tensor([0.2, 0.9])

    _fbcache_forward(m, layout, inputs, ts, seq_chunk_rows=6, ffn_chunk_rows=11)
    narrower = [rows for rows in seen if rows]
    seen.clear()
    _fbcache_forward(m, layout, inputs, ts, seq_chunk_rows=13, ffn_chunk_rows=11)

    assert narrower == [6] * TINY_FULL["num_layers"]
    assert [rows for rows in seen if rows] == [11] * TINY_FULL["num_layers"]
