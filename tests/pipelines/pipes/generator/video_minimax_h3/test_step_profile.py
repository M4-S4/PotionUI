from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import src.pipelines.pipes.generator.video_minimax_h3.main as h3_main
import src.platform.runtime.native.sol_attn as sol_attn
import tests.pipelines.pipes.generator.video_minimax_h3.test_main as h3_tests
from src.pipelines.pipes._shared.generation.generator_base import GeneratorContext
from src.pipelines.pipes._shared.generation.progress import ProgressEmitter
from src.pipelines.pipes.generator.video_minimax_h3.main import GeneratorMinimaxH3Pipe, _MiniMaxH3Ctx
from src.platform.runtime.native.errors import SamplingCancelled
from src.platform.runtime.native.kernels import stream_wait
from src.platform.runtime.native.sampling import step_profile
from src.platform.runtime.native.sampling.step_profile import STEP_EVENT


@pytest.fixture(autouse=True)
def _clean_state():
    sol_attn.reset_sol_attn_state()
    stream_wait.stop()
    yield
    sol_attn.reset_sol_attn_state()
    stream_wait.stop()


def _profiler(monkeypatch, *, active: bool = True):
    events: list = []
    fake = SimpleNamespace(active=active, mark=lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(step_profile, "get_profiler", lambda: fake)
    return events


def _steps(events):
    return [fields for event, fields in events if event == STEP_EVENT]


def _cache_driving_dit(monkeypatch):
    base = h3_tests._fake_dit_module
    probe = torch.ones(1, 4)

    def factory(video_patch_dim, audio_channels=32, seen_step_caches=None, seen_sol_attn=None):
        inner = base(video_patch_dim, audio_channels, seen_step_caches, seen_sol_attn)

        def forward(**kwargs):
            cache = kwargs.get("step_cache")
            if cache is not None and cache.should_skip(probe):
                return cache.record_skip()
            out = inner(**kwargs)
            if cache is not None:
                cache.record_compute(probe, out)
            return out

        forward.prepare_text_context = inner.prepare_text_context
        return forward

    monkeypatch.setattr(h3_tests, "_fake_dit_module", factory)


def test_each_step_carries_the_sparse_attention_plan(monkeypatch):
    events = _profiler(monkeypatch)
    h3_tests._run_generate_one({"sparse_attn": "sol", "sparse_attn_dense_last_steps": 2}, steps=4)

    steps = _steps(events)
    assert [(s["step"], s["steps"], s["attn"]) for s in steps] == [
        (1, 4, "sol-attn"), (2, 4, "sol-attn"), (3, 4, "dense"), (4, 4, "dense"),
    ]
    assert all(isinstance(s["step_s"], float) and s["step_s"] >= 0.0 for s in steps)
    assert all("cache_skip" not in s for s in steps)


def test_a_dense_warm_up_shows_on_the_leading_steps(monkeypatch):
    events = _profiler(monkeypatch)
    h3_tests._run_generate_one(
        {"sparse_attn": "sol", "sparse_attn_dense_last_steps": 1, "sparse_attn_start_percent": 0.25}, steps=4,
    )
    assert [s["attn"] for s in _steps(events)] == ["dense", "sol-attn", "sol-attn", "dense"]


def test_sparse_attention_off_reports_every_step_dense(monkeypatch):
    events = _profiler(monkeypatch)
    h3_tests._run_generate_one({}, steps=3)
    assert [s["attn"] for s in _steps(events)] == ["dense", "dense", "dense"]


def test_a_latched_off_sol_attn_reports_the_dense_path_it_fell_back_to(monkeypatch):
    events = _profiler(monkeypatch)
    monkeypatch.setattr(sol_attn, "_disabled_reason", "no kernel on this GPU")
    h3_tests._run_generate_one({"sparse_attn": "sol", "sparse_attn_dense_last_steps": 1}, steps=3)
    assert [s["attn"] for s in _steps(events)] == ["dense", "dense", "dense"]


def test_each_step_flags_whether_the_step_cache_skipped_it(monkeypatch):
    _cache_driving_dit(monkeypatch)
    events = _profiler(monkeypatch)
    h3_tests._run_generate_one(
        {"step_cache_threshold": 0.12, "step_cache_warmup_steps": 1, "step_cache_max_skips": 9}, steps=4,
    )
    assert [s["cache_skip"] for s in _steps(events)] == [False, True, True, False]


def test_profiling_off_never_consults_the_step_plan(monkeypatch):
    events = _profiler(monkeypatch, active=False)
    calls: list = []
    real = h3_main.step_attention_mode
    monkeypatch.setattr(h3_main, "step_attention_mode", lambda *a: calls.append(a) or real(*a))

    h3_tests._run_generate_one({"sparse_attn": "sol"}, steps=3)

    assert events == [] and calls == []


def test_a_cancelled_window_releases_the_stream_recorder(monkeypatch):
    _profiler(monkeypatch)
    steps = 6
    video_patch_dim = 24 * h3_tests.PATCH[0] * h3_tests.PATCH[1] * h3_tests.PATCH[2]
    bundle = h3_tests._fake_bundle_for_generate_one(video_patch_dim, seen_step_caches=[])
    ctx_extra = _MiniMaxH3Ctx(
        bundle=bundle, conditioning=[h3_tests._fake_conditioning(3)], steps=steps,
        height=32, width=32, frames=22, num_latent_frames=2, latent_height=2, latent_width=2,
        num_audio_latents=2, device="cpu", dtype=torch.float32, spec=bundle.spec,
        keyframe_images=[], keyframe_anchors=(), audio_source="generate", decode=True,
    )
    is_cancelled, _ = h3_tests._cancel_after(3)
    ctx = GeneratorContext(quantity=1, input_seeds=[7], extra=ctx_extra, is_cancelled=is_cancelled)
    pipe = GeneratorMinimaxH3Pipe({**GeneratorMinimaxH3Pipe.get_default_config(), "preview": False, "steps": steps})
    pipe._audio_results = []
    recorders: list = []
    real_start = stream_wait.start
    monkeypatch.setattr(stream_wait, "start", lambda **kw: recorders.append(real_start(**kw)) or recorders[-1])

    with patch("src.pipelines.pipes.generator.video_minimax_h3.main.encode_frames_to_mp4"):
        with pytest.raises(SamplingCancelled):
            pipe.generate_one(ctx, 0, 7, ProgressEmitter([].append, title="test"))

    assert len(recorders) == 1
    assert stream_wait.recorder is None
