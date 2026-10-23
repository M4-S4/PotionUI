from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.platform.runtime.native.engine import NativeGenerator
from src.platform.runtime.native.sampling.trajectory_cache import get_trajectory_cache

CONTEXT = torch.randn(1, 4, 8)


def _generator():
    return SimpleNamespace(
        spec=SimpleNamespace(family="krea2", variant="krea2_turbo", sampling_settings={"guidance": None}),
        dit=SimpleNamespace(module=object(), weight_revision=1),
    )


def _plan(generator, text_fusion=None):
    cond = {"context": CONTEXT}
    if text_fusion is not None:
        cond["text_fusion"] = text_fusion
    return NativeGenerator._plan_warm_start(
        generator, True, "euler", None, cond, None, 1234, (1, 4, 4, 4), 8, None, (),
        {"guidance": None, "shift": 2.02}, {}, None, None, sigmas=None,
    )


def _capture(generator, text_fusion=None):
    _plan(generator, text_fusion)
    cache = get_trajectory_cache()
    entry = cache.get(next(iter(cache._entries)))
    entry.checkpoints = {2: torch.zeros(1, 4, 4, 4), 4: torch.ones(1, 4, 4, 4), 6: torch.full((1, 4, 4, 4), 2.0)}


@pytest.fixture(autouse=True)
def _clean_cache():
    get_trajectory_cache().clear()
    yield
    get_trajectory_cache().clear()


def test_the_same_settings_resume_the_cached_trajectory():
    generator = _generator()
    fusion = {"strength": 1.0, "scale": 1.5}
    _capture(generator, fusion)

    resume, _ = _plan(generator, {"strength": 1.0, "scale": 1.5})

    assert resume is not None and resume[0] == 6


def test_a_changed_strength_never_resumes_a_trajectory_made_under_another():
    generator = _generator()
    _capture(generator, {"strength": 1.0, "scale": 1.0})

    resume, _ = _plan(generator, {"strength": 0.5, "scale": 1.0})

    assert resume is None


def test_a_changed_text_scale_never_resumes_a_trajectory_made_under_another():
    generator = _generator()
    _capture(generator, {"strength": 0.0, "scale": 1.5})

    resume, _ = _plan(generator, {"strength": 0.0, "scale": 2.0})

    assert resume is None


def test_switching_the_enhancer_on_or_off_never_resumes_across_the_switch():
    generator = _generator()
    _capture(generator)
    assert _plan(generator, {"strength": 1.0, "scale": 1.0})[0] is None

    get_trajectory_cache().clear()
    _capture(generator, {"strength": 1.0, "scale": 1.0})
    assert _plan(generator)[0] is None


def test_without_the_enhancer_an_identical_run_still_resumes():
    generator = _generator()
    _capture(generator)

    resume, _ = _plan(generator)

    assert resume is not None and resume[0] == 6


def test_a_neutral_setting_shares_the_off_trajectory():
    generator = _generator()
    _capture(generator)

    resume, _ = _plan(generator, {"strength": 0.0, "scale": 1.0})

    assert resume is not None and resume[0] == 6
