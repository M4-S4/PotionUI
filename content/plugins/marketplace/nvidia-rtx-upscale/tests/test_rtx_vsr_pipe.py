"""Behavioral tests for the `upscaler/rtx_vsr` pipe.

Every test here mocks `nvvfx` entirely (see `fake_nvvfx` in conftest.py) and
mocks `Tensor.cuda()` to a no-op (`no_real_cuda`) - the real GPU on this host
is shared and busy, and nvvfx is not installed in this venv at all.
"""

import io
import os
import subprocess
import sys
import threading

import numpy as np
import pytest
from PIL import Image

from src.plugin_api.pipes import PipeInput


@pytest.fixture
def pipe_class(pipe_module):
    return pipe_module.RtxUpscalePipe


def _rgb_image(width: int, height: int) -> Image.Image:
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8), mode="RGB")


# -- round_to_8 / target_size --------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (64, 64),
    (65, 64),   # 65/8 = 8.125 -> round -> 8 -> 64
    (100, 96),  # 100/8 = 12.5 -> banker's rounding -> 12 -> 96
    (4, 8),     # rounds to 0 -> floored up to the 8px minimum
    (0, 8),
])
def test_round_to_8(pipe_module, value, expected):
    assert pipe_module.round_to_8(value) == expected


def test_target_size_forces_input_dims_for_denoise_and_deblur(pipe_module):
    """Denoise/deblur are same-resolution enhancement - `scale` must have no
    effect on the output size, even when it's far from 1.0."""
    for intent in ("denoise", "deblur"):
        assert pipe_module.target_size(intent, 4.0, 100, 200) == (96, 200)
        assert pipe_module.target_size(intent, 1.0, 100, 200) == (96, 200)


def test_target_size_scales_for_upscale_intents(pipe_module):
    for intent in ("upscale", "upscale_highbitrate"):
        assert pipe_module.target_size(intent, 2.0, 100, 200) == (
            pipe_module.round_to_8(200), pipe_module.round_to_8(400)
        )


def test_scale_config_declares_the_4x_cap(pipe_class):
    """`scale`'s min/max drive the shared config validator's cap - the pipe
    doesn't hand-roll a range check of its own."""
    spec = next(s for s in pipe_class.configuration() if s.name == "scale")

    assert spec.min_value == 1.0
    assert spec.max_value == 4.0


# -- quality mapping -------------------------------------------------------

def test_quality_mapping_covers_every_intent_and_tier(pipe_module, fake_nvvfx):
    expected = {
        ("upscale", "low"): "LOW",
        ("upscale", "medium"): "MEDIUM",
        ("upscale", "high"): "HIGH",
        ("upscale", "ultra"): "ULTRA",
        ("upscale_highbitrate", "low"): "HIGHBITRATE_LOW",
        ("upscale_highbitrate", "ultra"): "HIGHBITRATE_ULTRA",
        ("denoise", "low"): "DENOISE_LOW",
        ("denoise", "ultra"): "DENOISE_ULTRA",
        ("deblur", "low"): "DEBLUR_LOW",
        ("deblur", "ultra"): "DEBLUR_ULTRA",
    }
    for (intent, quality), attr_name in expected.items():
        level = pipe_module._quality_level(fake_nvvfx, intent, quality)
        assert level == getattr(fake_nvvfx.VideoSuperRes.QualityLevel, attr_name)
        assert level != fake_nvvfx.VideoSuperRes.QualityLevel.BICUBIC


def test_quality_mapping_falls_back_to_effects_namespace(pipe_module):
    """The reference ComfyUI node reads `nvvfx.effects.QualityLevel` instead of
    `VideoSuperRes.QualityLevel` - both spellings must resolve."""
    from enum import IntEnum
    from types import ModuleType, SimpleNamespace

    class QL(IntEnum):
        ULTRA = 4

    module = ModuleType("nvvfx")
    module.effects = SimpleNamespace(QualityLevel=QL)
    # No `VideoSuperRes` attribute at all on this fake module.

    level = pipe_module._quality_level(module, "upscale", "ultra")
    assert level == QL.ULTRA


def test_quality_mapping_missing_nvvfx_enum_raises_actionable_error(pipe_module):
    from types import ModuleType

    module = ModuleType("nvvfx")  # neither VideoSuperRes nor effects

    with pytest.raises(pipe_module.GenerationExecutionError):
        pipe_module._quality_level(module, "upscale", "ultra")


# -- missing nvvfx -----------------------------------------------------

def test_missing_nvvfx_raises_actionable_error(pipe_module, blocking_finder_cls, monkeypatch):
    finder = blocking_finder_cls(["nvvfx"])
    monkeypatch.delitem(sys.modules, "nvvfx", raising=False)
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(pipe_module.GenerationExecutionError) as exc_info:
            pipe_module._import_nvvfx()
    finally:
        sys.meta_path.remove(finder)

    message = str(exc_info.value)
    assert 'pip install "nvidia-vfx>=0.2" --extra-index-url https://pypi.nvidia.com' in message
    assert "570.190" in message
    assert "LD_LIBRARY_PATH=/usr/lib/wsl/lib" in message


# -- session: reload-on-size-change and clone-before-next-run ------------

def test_ensure_loaded_reloads_only_on_size_change(pipe_module, fake_nvvfx):
    level = fake_nvvfx.VideoSuperRes.QualityLevel.ULTRA
    session = pipe_module._RtxSession(fake_nvvfx, level)
    effect = fake_nvvfx.VideoSuperRes.instances[-1]

    session.ensure_loaded(64, 64)
    assert effect.load_calls == 1

    session.ensure_loaded(64, 64)  # same size -> no reload
    assert effect.load_calls == 1

    session.ensure_loaded(128, 64)  # different size -> reload
    assert effect.load_calls == 2


def test_clone_dlpack_result_copies_before_the_buffer_is_reused(pipe_module):
    """Isolates the exact hazard NVIDIA's own `VideoSuperRes.run()` docstring
    warns about: "you must copy/clone it immediately before the next call or
    close()." A plain `torch.from_dlpack(result.image)` WITHOUT `.clone()`
    would return a view of the same storage nvvfx reuses for its next
    `run()` -- this simulates that reuse directly (mutating the same buffer
    object in place) rather than going through the full pipe, whose later
    dtype-converting ops happen to copy anyway and would mask a missing
    clone here."""
    import torch
    from types import SimpleNamespace

    buffer = torch.zeros(3, 4, 4)
    result = SimpleNamespace(image=buffer)

    cloned = pipe_module._clone_dlpack_result(result)
    buffer.fill_(1.0)  # simulates nvvfx reusing/overwriting this buffer on the next run()

    assert torch.all(cloned == 0.0), (
        "the cloned tensor changed when the source buffer was mutated -- "
        "_clone_dlpack_result is not actually copying the data out"
    )


def test_process_frame_produces_independent_output_per_call(pipe_module, fake_nvvfx, no_real_cuda):
    """End-to-end companion to the isolated clone test above: two frames
    processed through the same session/effect must not observe each other's
    data, even though the fake effect reuses one buffer across `run()` calls."""
    level = fake_nvvfx.VideoSuperRes.QualityLevel.ULTRA
    session = pipe_module._RtxSession(fake_nvvfx, level)
    session.ensure_loaded(8, 8)

    first = session.process_frame(_rgb_image(8, 8))
    first_pixels = np.array(first).copy()

    session.process_frame(_rgb_image(8, 8))

    assert np.all(first_pixels == 0), "the first frame's already-returned pixels must not change later"


# -- full process(): denoise/deblur ignore scale, cancellation -----------

def test_process_image_denoise_ignores_scale(pipe_class, fake_nvvfx, no_real_cuda):
    pipe = pipe_class({"intent": "denoise", "quality": "high", "scale": 4.0})
    outputs = []
    result = pipe.process(
        PipeInput(input={"image": [_rgb_image(40, 24)]}),
        outputs.append,
    )

    (out_image,) = result.output["image"]
    assert out_image.size == (40, 24)


def test_process_image_upscale_applies_scale(pipe_class, fake_nvvfx, no_real_cuda):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})
    outputs = []
    result = pipe.process(
        PipeInput(input={"image": [_rgb_image(40, 24)]}),
        outputs.append,
    )

    (out_image,) = result.output["image"]
    assert out_image.size == (80, 48)


def test_process_requires_image_or_video(pipe_module, pipe_class, fake_nvvfx):
    pipe = pipe_class(pipe_class.get_default_config())

    with pytest.raises(pipe_module.GenerationExecutionError):
        pipe.process(PipeInput(input={}), lambda o: None)


def test_cancellation_stops_before_processing_later_images(pipe_class, fake_nvvfx, no_real_cuda):
    calls = {"n": 0}

    def is_cancelled():
        calls["n"] += 1
        return calls["n"] > 1  # let the first image through, stop before the second

    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})
    images = [_rgb_image(16, 16), _rgb_image(16, 16), _rgb_image(16, 16)]
    result = pipe.process(
        PipeInput(input={"image": images}),
        lambda o: None,
        is_cancelled,
    )

    assert len(result.output["image"]) == 1


class _ChattyFfmpeg:
    def __init__(self, flood_bytes):
        read_fd, write_fd = os.pipe()
        self.stdin = io.BytesIO()
        self.stderr = os.fdopen(read_fd, "rb")
        self.returncode = None
        self._write_fd = write_fd
        self._flood = threading.Thread(target=self._chatter, args=(flood_bytes,), daemon=True)
        self._flood.start()

    def _chatter(self, flood_bytes):
        with os.fdopen(self._write_fd, "wb") as sink:
            sink.write(b"E" * flood_bytes)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self._flood.join(timeout=2)
        if self._flood.is_alive():
            raise subprocess.TimeoutExpired(cmd="chatty-ffmpeg", timeout=timeout)
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_the_pipe_writes_video_through_the_shared_writer(pipe_module, monkeypatch):
    from src.plugin_api import media

    assert pipe_module.StreamingMp4Writer is media.StreamingMp4Writer

    proc = _ChattyFfmpeg(flood_bytes=400_000)
    video_stream = sys.modules["src.platform.util.video_stream"]
    monkeypatch.setattr(video_stream, "find_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(video_stream.subprocess, "Popen", lambda *a, **k: proc)

    writer = pipe_module.StreamingMp4Writer("out.mp4", 8, 8, 24.0)
    writer.write(np.zeros((8, 8, 3), dtype=np.uint8))
    writer.close()

    assert proc.returncode == 0


@pytest.mark.parametrize("width,height,target,expected", [
    (832, 480, 1080, (1872, 1080)),
    (768, 448, 1080, (1848, 1080)),
    (480, 832, 1080, (1080, 1872)),
    (832, 480, 1440, (2496, 1440)),
    (832, 480, 4000, (3328, 1920)),
    (832, 480, 240, (832, 480)),
])
def test_target_short_side_sets_the_scale(pipe_module, width, height, target, expected):
    assert pipe_module.target_size("upscale", 2.0, width, height, target) == expected


def test_target_short_side_is_ignored_by_denoise_and_deblur(pipe_module):
    for intent in ("denoise", "deblur"):
        assert pipe_module.target_size(intent, 2.0, 832, 480, 1080) == (832, 480)


def test_zero_target_short_side_keeps_the_scale(pipe_module):
    assert pipe_module.target_size("upscale", 2.0, 832, 480, 0) == (1664, 960)


@pytest.mark.parametrize("width,height", [(100, 60), (101, 57), (833, 481), (7, 9)])
def test_an_upscale_never_lands_below_the_source(pipe_module, width, height):
    out_w, out_h = pipe_module.target_size("upscale", 1.0, width, height)
    assert out_w >= width and out_h >= height
    assert out_w % 8 == 0 and out_h % 8 == 0


def test_target_short_side_and_strength_are_declared(pipe_class):
    specs = {spec.name: spec for spec in pipe_class.configuration()}
    assert specs["target_short_side"].default == 0
    assert specs["target_short_side"].min_value == 0
    assert specs["strength"].default == 1.0
    assert (specs["strength"].min_value, specs["strength"].max_value) == (0.0, 1.0)


def test_process_image_reaches_the_target_short_side(pipe_class, fake_nvvfx, no_real_cuda):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 1.0, "target_short_side": 48})
    result = pipe.process(PipeInput(input={"image": [_rgb_image(40, 24)]}), lambda o: None)

    (out_image,) = result.output["image"]
    assert out_image.size == (80, 48)


def test_full_strength_is_not_sent_to_the_sdk(pipe_class, fake_nvvfx, no_real_cuda):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0, "strength": 1.0})
    pipe.process(PipeInput(input={"image": [_rgb_image(16, 16)]}), lambda o: None)

    assert "strength" not in fake_nvvfx.VideoSuperRes.instances[-1].kwargs


def test_a_lower_strength_reaches_the_sdk(pipe_class, fake_nvvfx, no_real_cuda):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0, "strength": 0.4})
    pipe.process(PipeInput(input={"image": [_rgb_image(16, 16)]}), lambda o: None)

    assert fake_nvvfx.VideoSuperRes.instances[-1].kwargs["strength"] == pytest.approx(0.4)


def test_strength_falls_back_on_nvvfx_01(pipe_class, fake_nvvfx_01, no_real_cuda, caplog):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0, "strength": 0.4})
    with caplog.at_level("WARNING"):
        result = pipe.process(PipeInput(input={"image": [_rgb_image(16, 16)]}), lambda o: None)

    assert result.output["image"][0].size == (32, 32)
    assert "strength" not in fake_nvvfx_01.VideoSuperRes.instances[-1].kwargs
    assert "nvidia-vfx < 0.2" in caplog.text


def test_the_quality_levels_resolve_on_nvvfx_01(pipe_module, fake_nvvfx_01):
    level = pipe_module._quality_level(fake_nvvfx_01, "upscale_highbitrate", "ultra")
    assert level == fake_nvvfx_01.VideoSuperRes.QualityLevel.HIGHBITRATE_ULTRA


def test_a_size_error_from_load_is_explained(pipe_module, pipe_class, fake_nvvfx, no_real_cuda):
    fake_nvvfx.VideoSuperRes.fail_load = RuntimeError("NvVFX_Load failed: invalid parameter (code -7)")
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})

    with pytest.raises(pipe_module.GenerationExecutionError) as exc_info:
        pipe.process(PipeInput(input={"image": [_rgb_image(40, 24)]}), lambda o: None)

    message = str(exc_info.value)
    assert "refused the frame size" in message
    assert "40x24 to 80x48" in message
    assert "-7" in message


def test_another_sdk_error_names_the_sizes(pipe_module, pipe_class, fake_nvvfx, no_real_cuda):
    fake_nvvfx.VideoSuperRes.fail_run = RuntimeError("NvVFX_Run failed: inference error")
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})

    with pytest.raises(pipe_module.GenerationExecutionError) as exc_info:
        pipe.process(PipeInput(input={"image": [_rgb_image(40, 24)]}), lambda o: None)

    message = str(exc_info.value)
    assert "failed (40x24 to 80x48)" in message
    assert "refused" not in message


def test_the_effect_is_closed_after_a_failure(pipe_module, pipe_class, fake_nvvfx, no_real_cuda):
    fake_nvvfx.VideoSuperRes.fail_run = RuntimeError("boom")
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})

    with pytest.raises(pipe_module.GenerationExecutionError):
        pipe.process(PipeInput(input={"image": [_rgb_image(16, 16)]}), lambda o: None)

    assert fake_nvvfx.VideoSuperRes.instances[-1].closed


def test_nvvfx_02_gets_the_source_size_before_load(pipe_class, fake_nvvfx, no_real_cuda):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})
    pipe.process(PipeInput(input={"image": [_rgb_image(40, 24), _rgb_image(48, 24)]}), lambda o: None)
    effect = fake_nvvfx.VideoSuperRes.instances[-1]

    assert (effect.input_width, effect.input_height) == (48, 24)
    assert effect.load_calls == 2


def test_nvvfx_01_loads_without_a_source_size(pipe_class, fake_nvvfx_01, no_real_cuda):
    pipe = pipe_class({"intent": "upscale", "quality": "ultra", "scale": 2.0})
    pipe.process(PipeInput(input={"image": [_rgb_image(40, 24)]}), lambda o: None)

    assert not hasattr(fake_nvvfx_01.VideoSuperRes.instances[-1], "input_width")


def test_a_new_source_size_reloads_even_at_the_same_output(pipe_module, fake_nvvfx):
    session = pipe_module._RtxSession(fake_nvvfx, fake_nvvfx.VideoSuperRes.QualityLevel.ULTRA)
    effect = fake_nvvfx.VideoSuperRes.instances[-1]

    session.ensure_loaded(64, 64, (32, 32))
    session.ensure_loaded(64, 64, (32, 32))
    session.ensure_loaded(64, 64, (16, 16))

    assert effect.load_calls == 2


def test_frames_run_without_a_cuda_stream_when_cuda_is_absent(pipe_module, no_real_cuda):
    import contextlib
    import torch

    assert isinstance(pipe_module.default_stream(torch), contextlib.nullcontext)
