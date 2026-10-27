from __future__ import annotations

import importlib
import importlib.util
import logging
import math

import pytest
import torch
import torch.nn.functional as F

from src.platform.runtime.native import kernels
from src.platform.runtime.native.kernels import FLASH, SAGE, SAGE2, SAGE3, SDPA, SPARGE, attention
from tests.fixtures.kernel_capabilities import ampere, fake_capabilities

att = importlib.import_module("src.platform.runtime.native.kernels.attention")

HEADS = 4
KV_HEADS = 2
LENGTH = 128
HEAD_DIM = 64
ACCELERATED = (FLASH, SAGE, SAGE2, SAGE3, SPARGE)
BACKENDS = (SDPA, *ACCELERATED)
ADAPTER = {FLASH: "_flash", SAGE: "_sage", SAGE2: "_sage", SAGE3: "_sage3", SPARGE: "_sparge"}
SDPA_TAKES_GROUPED_KV = "enable_gqa" in str(F.scaled_dot_product_attention.__doc__ or "")


class _CudaLike(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def _cuda_like(tensor):
    return tensor.as_subclass(_CudaLike)


def _inputs(dtype=torch.float16, *, gqa=False, q_len=LENGTH, kv_len=LENGTH):
    generator = torch.Generator().manual_seed(0)
    kv_heads = KV_HEADS if gqa else HEADS
    q = torch.randn(1, HEADS, q_len, HEAD_DIM, generator=generator)
    k = torch.randn(1, kv_heads, kv_len, HEAD_DIM, generator=generator)
    v = torch.randn(1, kv_heads, kv_len, HEAD_DIM, generator=generator)
    return _cuda_like(q.to(dtype)), k.to(dtype), v.to(dtype)


def _mask(kind, q_len=LENGTH, kv_len=LENGTH):
    if kind == "none":
        return None
    generator = torch.Generator().manual_seed(1)
    keep = torch.rand(q_len, kv_len, generator=generator) > 0.3
    keep[:, 0] = True
    if kind == "bool":
        return keep[None, None]
    return torch.zeros(q_len, kv_len).masked_fill(~keep, float("-inf"))[None, None].half()


def _reference(q, k, v, mask, causal):
    q, k, v = q.float(), k.float(), v.float()
    repeat = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask.float()
    if causal:
        keep = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _expected_adapter(backend, mask_kind, causal, q_len=LENGTH, kv_len=LENGTH):
    if backend == SDPA or mask_kind != "none":
        return "_sdpa"
    if causal and q_len != kv_len:
        return "_sdpa"
    return ADAPTER[backend]


def _fallback_notes(caplog, level=logging.INFO):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and "falls back to sdpa" in record.getMessage()
    ]


@pytest.fixture(autouse=True)
def _clean_state():
    att.reset_backend_cache()
    att.set_backend_override(None)
    yield
    att.reset_backend_cache()
    att.set_backend_override(None)


@pytest.fixture
def install_all(monkeypatch, install_capabilities):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def install(*names):
        install_capabilities(fake_capabilities(ampere(attention_backends=frozenset({SDPA, *names}))))

    install(*ACCELERATED)
    return install


@pytest.fixture
def kernel_calls(monkeypatch, install_all):
    calls = []

    def stub(adapter):
        def run(q, k, v, causal=False):
            calls.append((adapter, tuple(k.shape), causal))
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        return run

    for adapter in set(ADAPTER.values()):
        monkeypatch.setattr(att, adapter, stub(adapter))

    real_sdpa = att._sdpa

    def recording_sdpa(q, k, v, mask, grouped=False, causal=False):
        calls.append(("_sdpa", tuple(k.shape), causal))
        return real_sdpa(q, k, v, mask, grouped=grouped, causal=causal)

    monkeypatch.setattr(att, "_sdpa", recording_sdpa)
    return calls


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_dispatch_matches_the_reference_and_runs_the_expected_kernel(kernel_calls, backend, mask_kind, gqa, causal):
    q, k, v = _inputs(gqa=gqa)
    mask = _mask(mask_kind)

    out = attention(q, k, v, mask=mask, causal=causal, backend=backend)

    torch.testing.assert_close(out.float(), _reference(q, k, v, mask, causal), atol=1e-2, rtol=1e-2)
    adapter = _expected_adapter(backend, mask_kind, causal)
    assert [call[0] for call in kernel_calls] == [adapter]
    assert kernel_calls[0][2] is causal
    if adapter != "_sdpa":
        assert kernel_calls[0][1][1] == HEADS
    elif gqa:
        assert kernel_calls[0][1][1] == (KV_HEADS if SDPA_TAKES_GROUPED_KV else HEADS)


@pytest.mark.parametrize("backend", ACCELERATED)
def test_a_masked_call_falls_back_inside_the_dispatcher(kernel_calls, backend):
    q, k, v = _inputs()
    mask = _mask("bool")

    out = attention(q, k, v, mask=mask, backend=backend)

    torch.testing.assert_close(out.float(), _reference(q, k, v, mask, False), atol=1e-2, rtol=1e-2)
    assert [call[0] for call in kernel_calls] == ["_sdpa"]


def test_auto_selection_prefers_the_highest_priority_kernel_and_never_sparge(kernel_calls, install_all):
    q, k, v = _inputs()
    attention(q, k, v)
    assert [call[0] for call in kernel_calls] == ["_sage3"]

    kernel_calls.clear()
    install_all(SPARGE)
    attention(q, k, v)
    assert [call[0] for call in kernel_calls] == ["_sdpa"]


def test_a_pin_chooses_the_kernel_over_auto_selection(kernel_calls):
    q, k, v = _inputs()
    att.set_backend_override(FLASH)
    attention(q, k, v)
    assert [call[0] for call in kernel_calls] == ["_flash"]

    kernel_calls.clear()
    attention(q, k, v, backend=SAGE2)
    assert [call[0] for call in kernel_calls] == ["_sage"]


def test_an_unavailable_pin_falls_back_to_the_best_available_kernel(kernel_calls, install_all, caplog):
    install_all(SAGE2)
    q, k, v = _inputs()
    with caplog.at_level(logging.WARNING, logger=att.__name__):
        attention(q, k, v, backend=SAGE3)
    assert [call[0] for call in kernel_calls] == ["_sage"]
    assert any("sage3" in record.getMessage() and "unavailable" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_causal_with_unequal_lengths_uses_top_left_alignment_on_sdpa(kernel_calls, mask_kind):
    q, k, v = _inputs(q_len=64, kv_len=128)
    mask = _mask(mask_kind, 64, 128)

    out = attention(q, k, v, mask=mask, causal=True, backend=SDPA)

    torch.testing.assert_close(out.float(), _reference(q, k, v, mask, True), atol=1e-2, rtol=1e-2)


def test_causal_with_unequal_lengths_falls_back_from_an_accelerated_kernel(kernel_calls, caplog):
    q, k, v = _inputs(q_len=64, kv_len=128)
    with caplog.at_level(logging.DEBUG, logger=att.__name__):
        out = attention(q, k, v, causal=True, backend=FLASH)

    torch.testing.assert_close(out.float(), _reference(q, k, v, None, True), atol=1e-2, rtol=1e-2)
    assert [call[0] for call in kernel_calls] == ["_sdpa"]
    assert len(_fallback_notes(caplog)) == 1
    assert "causal-constrained" in _fallback_notes(caplog)[0]


def test_unequal_lengths_without_causal_keep_the_accelerated_kernel(kernel_calls):
    q, k, v = _inputs(q_len=64, kv_len=128)
    attention(q, k, v, backend=FLASH)
    assert [call[0] for call in kernel_calls] == ["_flash"]


def test_a_key_padding_mask_broadcasts_against_the_causal_triangle():
    q, k, v = _inputs()
    mask = torch.ones(1, 1, 1, LENGTH, dtype=torch.bool)
    mask[..., LENGTH // 2:] = False

    out = attention(q, k, v, mask=mask, causal=True, backend=SDPA)

    torch.testing.assert_close(out.float(), _reference(q, k, v, mask, True), atol=1e-2, rtol=1e-2)


def test_sdpa_without_a_mask_hands_causal_to_torch(monkeypatch):
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kwargs):
        seen.append((attn_mask, kwargs["is_causal"]))
        return real(q, k, v, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    q, k, v = _inputs()
    attention(q, k, v, causal=True, backend=SDPA)
    assert seen == [(None, True)]


def test_sdpa_with_a_mask_folds_causal_into_one_mask(monkeypatch):
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, attn_mask=None, **kwargs):
        seen.append((attn_mask.clone(), kwargs["is_causal"]))
        return real(q, k, v, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    q, k, v = _inputs()
    attention(q, k, v, mask=torch.ones(1, 1, LENGTH, LENGTH, dtype=torch.bool), causal=True, backend=SDPA)

    folded, is_causal = seen[0]
    assert is_causal is False
    assert torch.equal(folded[0, 0], torch.ones(LENGTH, LENGTH, dtype=torch.bool).tril())


def test_a_float_mask_keeps_its_values_where_causal_allows_them():
    q, k, v = _inputs(torch.float32)
    mask = torch.randn(1, 1, LENGTH, LENGTH)

    out = attention(q, k, v, mask=mask, causal=True, backend=SDPA)

    torch.testing.assert_close(out, _reference(q, k, v, mask, True), atol=1e-4, rtol=1e-4)


def test_a_fallback_is_logged_once_per_backend_and_reason(kernel_calls, install_all, caplog):
    q, k, v = _inputs()
    mask = _mask("bool")
    with caplog.at_level(logging.DEBUG, logger=att.__name__):
        for _ in range(3):
            attention(q, k, v, mask=mask, backend=SAGE2)
        first = _fallback_notes(caplog)
        assert len(first) == 1 and "sage2" in first[0] and "mask-constrained" in first[0]
        assert len(_fallback_notes(caplog, logging.DEBUG)) == 2

        attention(_cuda_like(q.float()), k.float(), v.float(), backend=SAGE2)
        attention(_cuda_like(q.float()), k.float(), v.float(), backend=SAGE2)
        attention(q, k, v, mask=mask, backend=FLASH)
        attention(q, k, v, mask=mask, backend=FLASH)

    notes = _fallback_notes(caplog)
    assert len(notes) == 3
    assert any("sage2" in note and "dtype-constrained" in note for note in notes)
    assert any("flash" in note and "mask-constrained" in note for note in notes)


def test_resetting_the_backend_cache_rearms_the_fallback_note(kernel_calls, install_all, caplog):
    q, k, v = _inputs()
    mask = _mask("bool")
    with caplog.at_level(logging.INFO, logger=att.__name__):
        attention(q, k, v, mask=mask, backend=SAGE2)
        attention(q, k, v, mask=mask, backend=SAGE2)
        assert len(_fallback_notes(caplog)) == 1

        att.reset_backend_cache()
        install_all(*ACCELERATED)
        attention(q, k, v, mask=mask, backend=SAGE2)

    assert len(_fallback_notes(caplog)) == 2


def test_the_kernel_in_use_note_is_logged_once(kernel_calls, caplog):
    q, k, v = _inputs()
    with caplog.at_level(logging.INFO, logger=att.__name__):
        for _ in range(3):
            attention(q, k, v, backend=SAGE2)
    in_use = [record for record in caplog.records if "kernels in use" in record.getMessage()]
    assert len(in_use) == 1
    assert "sage2" in in_use[0].getMessage()


def test_heads_that_disagree_with_the_tensors_raise():
    q, k, v = _inputs(gqa=True)
    with pytest.raises(ValueError, match="heads=8 disagrees with q.shape"):
        attention(q, k, v, heads=8)
    with pytest.raises(ValueError, match="kvheads=3 disagrees with k.shape"):
        attention(q, k, v, kvheads=3)


def test_kv_heads_that_do_not_divide_the_query_heads_raise():
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 3, 8, 16)
    with pytest.raises(ValueError, match="not a multiple of kv heads"):
        attention(q, k, k)


def test_inputs_that_are_not_head_split_take_the_plain_path():
    q = torch.randn(2, 8, 16)
    k = torch.randn(2, 12, 16)
    out = attention(q, k, k, backend=SDPA)
    assert torch.equal(out, F.scaled_dot_product_attention(q, k, k, is_causal=False))


def test_the_dispatcher_lives_only_in_the_kernels_package():
    assert importlib.util.find_spec("src.platform.runtime.native.attention") is None
    assert callable(kernels.attention)
    assert not hasattr(kernels, "grouped_attention")
    for name in (
        "BACKEND_PRIORITY",
        "PIN_ONLY_BACKENDS",
        "attention",
        "available_backends",
        "get_attention_backend",
        "get_backend_override",
        "known_backends",
        "reset_backend_cache",
        "set_backend_override",
        "supports_grouped_kv",
    ):
        assert name in kernels.__all__
        assert getattr(kernels, name) is getattr(att, name)
