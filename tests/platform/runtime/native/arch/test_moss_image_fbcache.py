from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.platform.runtime.native.arch.moss_image.config import MossImageConfig
from src.platform.runtime.native.arch.moss_image.model import MossImage
from src.platform.runtime.native.sampling.step_cache import FirstBlockCache
from vendor.gpl.comfyui.ops import pick_operations

from .test_moss_image_parity import CONFIG, CHANNELS, TEXT_DIM, _inputs, _sigma

_OLD = Path(__file__).parent / "fixtures" / "moss_image_forward_before_fbcache.py"


def _legacy_forward():
    spec = importlib.util.spec_from_file_location("moss_image_forward_before_fbcache", str(_OLD))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.legacy_forward


class _Cache:
    def __init__(self, skip: bool) -> None:
        self.skip = skip
        self.probes: list[torch.Tensor] = []
        self.decisions = 0
        self.recorded: tuple | None = None
        self.skipped = False

    def should_skip(self, probe):
        self.decisions += 1
        self.probes.append(probe)
        return self.skip

    def record_skip(self):
        self.skipped = True
        return torch.full((1,), 7.0)

    def record_compute(self, probe, output):
        self.recorded = (probe, output)


@pytest.fixture()
def model():
    torch.manual_seed(1)
    m = MossImage(MossImageConfig.from_detect_config(CONFIG), pick_operations(torch.float32, torch.float32))
    for p in m.parameters():
        p.data.normal_(0, 0.1)
    return m.eval()


def _count_blocks(model):
    calls = {i: 0 for i in range(len(model.layers))}
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda *_a, i=i: calls.__setitem__(i, calls[i] + 1))
    return calls


@pytest.mark.parametrize("lengths,skip", [([5], None), ([3, 7], None), ([4, 2], [1]), ([6], [0, 2])])
def test_without_a_cache_forward_is_byte_identical_to_the_previous_forward(model, lengths, skip):
    x, context, mask = _inputs(3, 4, 3, lengths)
    sigma = _sigma(0.4, len(lengths))
    with torch.no_grad():
        expected = _legacy_forward()(model, x, sigma, context, mask, skip)
        actual = model(x, sigma, context, attention_mask=mask, skip_layers=skip)
        with_none = model(x, sigma, context, attention_mask=mask, skip_layers=skip, step_cache=None)
    assert torch.equal(actual, expected)
    assert torch.equal(with_none, expected)


def test_a_skip_decision_stops_after_block_zero(model):
    x, context, mask = _inputs(4, 4, 4, [5])
    calls = _count_blocks(model)
    cache = _Cache(skip=True)
    with torch.no_grad():
        out = model(x, _sigma(0.5), context, attention_mask=mask, step_cache=cache)
    assert cache.skipped and cache.recorded is None
    assert torch.equal(out, torch.full((1,), 7.0))
    assert calls[0] == 1
    assert all(calls[i] == 0 for i in range(1, len(calls)))


def test_a_compute_decision_runs_every_block_and_records_the_video_probe(model):
    x, context, mask = _inputs(5, 4, 4, [5])
    calls = _count_blocks(model)
    cache = _Cache(skip=False)
    with torch.no_grad():
        out = model(x, _sigma(0.5), context, attention_mask=mask, step_cache=cache)
        ran = dict(calls)
        expected = _legacy_forward()(model, x, _sigma(0.5), context, mask, None)
    assert all(count == 1 for count in ran.values())
    assert torch.equal(out, expected)
    probe, recorded = cache.recorded
    assert probe.shape == (1, 16, model.cfg.dim)
    assert torch.equal(recorded, out)


def test_batch_of_two_with_different_text_lengths_makes_one_decision_over_stacked_video_tokens(model):
    x, context, mask = _inputs(6, 3, 4, [2, 9])
    cache = _Cache(skip=False)
    with torch.no_grad():
        out = model(x, _sigma(0.5, 2), context, attention_mask=mask, step_cache=cache)
        expected = _legacy_forward()(model, x, _sigma(0.5, 2), context, mask, None)
    assert cache.decisions == 1
    assert cache.probes[0].shape == (2, 12, model.cfg.dim)
    assert torch.equal(out, expected)


def test_skip_layers_containing_block_zero_still_probes_and_skips(model):
    x, context, mask = _inputs(7, 4, 4, [5])
    calls = _count_blocks(model)
    cache = _Cache(skip=True)
    with torch.no_grad():
        model(x, _sigma(0.5), context, attention_mask=mask, skip_layers=[0], step_cache=cache)
    assert cache.decisions == 1 and cache.skipped
    assert sum(calls.values()) == 0


def test_real_cache_skips_on_identical_inputs_and_returns_the_cached_output(model):
    x, context, mask = _inputs(8, 4, 4, [5, 3])
    sigma = _sigma(0.5, 2)
    cache = FirstBlockCache(rel_threshold=0.5, warmup_steps=0)
    with torch.no_grad():
        first = model(x, sigma, context, attention_mask=mask, step_cache=cache)
        second = model(x, sigma, context, attention_mask=mask, step_cache=cache)
    assert cache.stats() == {"computed": 1, "skipped": 1}
    assert torch.equal(first, second)
