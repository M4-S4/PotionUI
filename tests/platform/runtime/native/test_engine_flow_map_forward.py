from __future__ import annotations

import torch

from src.platform.runtime.native.engine import NativeGenerator


def _recording_forward():
    gen = NativeGenerator.__new__(NativeGenerator)
    calls = []

    def fake_module(x, sigma, context, **kwargs):
        calls.append(kwargs)
        return x

    gen.dit = type("Fake", (), {"module": staticmethod(fake_module)})()
    return gen._make_forward("cpu", torch.float32), calls


def test_make_forward_forwards_the_flow_map_target_when_present():
    model_forward, calls = _recording_forward()
    target = torch.tensor([0.25])
    model_forward(torch.zeros(1), torch.zeros(1), {"context": torch.zeros(1), "flow_map_target": target})
    assert calls[0]["flow_map_target"] is target


def test_make_forward_omits_the_flow_map_target_when_absent():
    model_forward, calls = _recording_forward()
    model_forward(torch.zeros(1), torch.zeros(1), {"context": torch.zeros(1)})
    assert "flow_map_target" not in calls[0]
