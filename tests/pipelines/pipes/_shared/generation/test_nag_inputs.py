from __future__ import annotations

import torch

from src.pipelines.pipes._shared.generation.nag_inputs import attach_nag


def _cond():
    return {"context": torch.ones(1, 4, 8)}


def _uncond():
    return {"context": torch.zeros(1, 3, 8), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_default_scale_is_noop():
    cond = _cond()
    out = attach_nag(cond, _uncond(), {})
    assert out is cond
    assert "nag_context" not in out


def test_scale_one_explicit_is_noop():
    cond = _cond()
    out = attach_nag(cond, _uncond(), {"nag_scale": 1.0})
    assert out is cond


def test_no_uncond_is_noop_even_with_scale():
    cond = _cond()
    out = attach_nag(cond, None, {"nag_scale": 1.5}, attention_mask=True)
    assert out is cond


def test_active_scale_attaches_nag_context_and_params():
    cond = _cond()
    uncond = _uncond()
    out = attach_nag(cond, uncond, {"nag_scale": 1.5, "nag_tau": 2.0, "nag_alpha": 0.25}, attention_mask=True)
    assert out is not cond
    assert torch.equal(out["nag_context"], uncond["context"])
    assert torch.equal(out["nag_attention_mask"], uncond["attention_mask"])
    assert out["nag"] == {"scale": 1.5, "tau": 2.0, "alpha": 0.25}
    assert torch.equal(out["context"], cond["context"])


def test_active_scale_default_tau_alpha():
    out = attach_nag(_cond(), _uncond(), {"nag_scale": 2.0}, attention_mask=True)
    assert out["nag"] == {"scale": 2.0, "tau": 3.5, "alpha": 0.5}


def test_missing_attention_mask_is_none_not_a_crash():
    uncond = {"context": torch.zeros(1, 3, 8)}
    out = attach_nag(_cond(), uncond, {"nag_scale": 1.2}, attention_mask=True)
    assert out["nag_attention_mask"] is None


def test_default_call_adds_no_attention_mask_key():
    uncond = _uncond()
    out = attach_nag(_cond(), uncond, {"nag_scale": 1.5})
    assert set(out) == {"context", "nag_context", "nag"}
    assert torch.equal(out["nag_context"], uncond["context"])
