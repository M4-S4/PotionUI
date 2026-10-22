from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.platform.runtime.native.engine import NativeGenerator


class _Recorder:
    def __call__(self, x, sigma, context, **kwargs):
        self.args = (x, sigma, context)
        self.kwargs = kwargs
        return x


def _forward(conditioning):
    recorder = _Recorder()
    generator = SimpleNamespace(dit=SimpleNamespace(module=recorder))
    forward = NativeGenerator._make_forward(generator, "cpu", torch.float32)
    x, sigma = torch.zeros(1), torch.ones(1)
    forward(x, sigma, {"context": torch.ones(2), **conditioning})
    return recorder


_BASE = {"y": None, "guidance": None, "attention_mask": None}
_T = torch.ones(1)
_FUSION = {"strength": 1.0, "scale": 1.5}

_CASES = [
    ({}, {}),
    ({"y": _T}, {"y": _T}),
    ({"pooled": _T}, {"y": _T}),
    ({"y": _T, "pooled": torch.zeros(1)}, {"y": _T}),
    ({"guidance": _T}, {"guidance": _T}),
    ({"attention_mask": _T}, {"attention_mask": _T}),
    ({"step_cache": _T}, {"step_cache": _T}),
    ({"ref_latents": [_T]}, {"ref_latents": [_T]}),
    ({"image_slots": (1, 2)}, {"image_slots": (1, 2)}),
    (
        {"control_context": _T, "control_context_scale": 0.5, "control_sigma_range": (0.0, 1.0)},
        {"control_context": _T, "control_context_scale": 0.5, "control_sigma_range": (0.0, 1.0)},
    ),
    (
        {"ref_boost": 1.5, "ref_boost_a": 2.0, "ref_boost_mask": _T},
        {"ref_boost": 1.5, "ref_boost_a": 2.0, "ref_boost_mask": _T},
    ),
    ({"ref_boost": 0.0}, {"ref_boost": 0.0}),
    ({"flow_map_target": _T}, {"flow_map_target": _T}),
    ({"skip_layers": [1, 2]}, {"skip_layers": [1, 2]}),
    ({"skip_layers": []}, {}),
    ({"skip_layers": None}, {}),
    ({"nag_context": _T}, {"nag_context": _T, "nag": None}),
    ({"nag_context": _T, "nag": {"scale": 2}}, {"nag_context": _T, "nag": {"scale": 2}}),
    (
        {"nag_context": _T, "nag": {"scale": 2}, "nag_attention_mask": _T},
        {"nag_context": _T, "nag": {"scale": 2}, "nag_attention_mask": _T},
    ),
    ({"nag": {"scale": 2}, "nag_attention_mask": _T}, {}),
    ({"text_fusion": _FUSION}, {"text_fusion": _FUSION}),
    ({"text_fusion": None}, {}),
]


@pytest.mark.parametrize(("conditioning", "expected"), _CASES)
def test_only_the_expected_keyword_arguments_reach_the_module(conditioning, expected):
    recorder = _forward(conditioning)

    assert set(recorder.kwargs) == set(_BASE) | set(expected)
    for key, value in {**_BASE, **expected}.items():
        assert recorder.kwargs[key] is value or recorder.kwargs[key] == value
