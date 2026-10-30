from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import src.platform.runtime.native.kernels.dispatch as att
from src.platform.runtime.native.kernels import SAGE, SAGE2, SDPA

_DISPATCH = "src.platform.runtime.native.kernels.dispatch"


def _qkv(q_len: int, k_len: int, heads: int = 4, kv_heads: int = 4, dim: int = 8):
    return (
        torch.randn(1, heads, q_len, dim), torch.randn(1, kv_heads, k_len, dim), torch.randn(1, kv_heads, k_len, dim),
    )


@pytest.mark.parametrize("backend", [SAGE, SAGE2, SDPA])
def test_self_attention_over_zero_tokens_returns_an_empty_output_without_reaching_a_kernel(backend):
    q, k, v = _qkv(0, 0)

    def kernel(*args, **kwargs):
        raise AssertionError("an empty sequence must never reach an attention kernel")

    with patch(f"{_DISPATCH}._resolve_dispatch", return_value=backend), \
         patch(f"{_DISPATCH}._sage", side_effect=kernel), \
         patch(f"{_DISPATCH}._sdpa", side_effect=kernel):
        out = att.attention(q, k, v, heads=4)

    assert out.shape == (1, 4, 0, 8)
    assert out.dtype == q.dtype


def test_zero_queries_against_real_keys_is_an_empty_output_shaped_by_the_value_width():
    q = torch.randn(2, 4, 0, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 6)
    assert att.attention(q, k, v).shape == (2, 4, 0, 6)


def test_grouped_heads_with_zero_queries_keep_the_query_head_count():
    q, k, v = _qkv(0, 0, heads=8, kv_heads=2)
    assert att.attention(q, k, v, heads=8, kvheads=2).shape == (1, 8, 0, 8)


def test_queries_against_zero_keys_raise_a_clear_error_instead_of_a_kernel_crash():
    q, k, v = _qkv(3, 0)
    with patch(f"{_DISPATCH}._resolve_dispatch", return_value=SAGE):
        with pytest.raises(ValueError, match="empty key/value sequence is undefined"):
            att.attention(q, k, v)


def test_non_empty_attention_still_reaches_the_kernel():
    q, k, v = _qkv(3, 3)
    with patch(f"{_DISPATCH}._resolve_dispatch", return_value=SDPA):
        out = att.attention(q, k, v)
    assert torch.allclose(out, torch.nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-6)
