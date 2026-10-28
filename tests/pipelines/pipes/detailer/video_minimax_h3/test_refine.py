from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.pipelines.pipes._shared.media.video_encode import AudioTrack
from src.pipelines.pipes._shared.video.tubes.detailer import TubeClip
from src.pipelines.pipes.detailer.video_minimax_h3 import refine as R
from src.pipelines.pipes.generator.video_minimax_h3.geometry import align_num_frames
from src.pipelines.pipes.generator.video_minimax_h3.main import GeneratorMinimaxH3Pipe

_MOD = "src.pipelines.pipes.detailer.video_minimax_h3.refine"


def test_max_window_frames_is_the_longest_aligned_clip_within_fifteen_seconds():
    assert R.MAX_WINDOW_FRAMES == 345
    assert align_num_frames(R.MAX_WINDOW_FRAMES) == R.MAX_WINDOW_FRAMES
    assert align_num_frames(R.MAX_WINDOW_FRAMES + 1) > 15 * 24


@pytest.mark.parametrize("w,h", [(512, 512), (100, 70), (512, 96), (96, 512), (1000, 40), (33, 33)])
def test_fit_h3_canvas_snaps_to_32_inside_supported_aspect(w, h):
    fw, fh = R.fit_h3_canvas(w, h)
    assert fw % 32 == 0 and fh % 32 == 0
    assert fw >= w and fh >= h
    assert 0.25 <= fw / fh <= 4.0


def test_fit_h3_canvas_keeps_an_already_valid_canvas():
    assert R.fit_h3_canvas(512, 768) == (512, 768)


def test_slice_audio_takes_the_tube_time_span():
    sr = 100
    wave = np.arange(2 * 1000, dtype=np.float32).reshape(2, 1000)
    track = R.slice_audio(wave, sr, 25.0, start_frame=50, num_frames=25)
    assert isinstance(track, AudioTrack)
    assert track.sample_rate == sr
    assert track.waveform.shape == (2, 100)
    assert track.waveform[0, 0] == wave[0, 200]
    assert track.waveform[1, -1] == wave[1, 299]


def test_slice_audio_pads_silence_past_the_end_and_for_silent_sources():
    wave = np.ones((2, 120), dtype=np.float32)
    track = R.slice_audio(wave, 100, 25.0, start_frame=25, num_frames=25)
    assert track.waveform.shape == (2, 100)
    assert np.all(track.waveform[:, :20] == 1.0)
    assert np.all(track.waveform[:, 20:] == 0.0)
    silent = R.slice_audio(np.zeros((2, 0), np.float32), 32000, 24.0, 0, 24)
    assert silent.waveform.shape == (2, 32000)
    assert not silent.waveform.any()


class _FakeVae:
    def __init__(self):
        self.moved = []
        self.offloaded = 0
        self.seen = None
        self.module = SimpleNamespace(parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]),
                                      encode=self._encode)

    def _encode(self, pixels):
        self.seen = pixels
        return torch.zeros(1, 4, 2, pixels.shape[3] // 16, pixels.shape[4] // 16)

    def move_to(self, device):
        self.moved.append(device)

    def offload(self):
        self.offloaded += 1


def test_encode_tube_pads_to_the_vae_lattice_and_imagenet_normalizes():
    vae = _FakeVae()
    frames = torch.full((7, 64, 32, 3), 0.5)
    latent, n0 = R.encode_tube(SimpleNamespace(video_vae=vae), frames, "cpu")
    assert n0 == 7
    assert tuple(vae.seen.shape) == (1, 3, align_num_frames(7), 64, 32)
    expected = (0.5 - 0.485) / 0.229
    assert torch.allclose(vae.seen[0, 0], torch.full_like(vae.seen[0, 0], expected), atol=1e-5)
    assert vae.moved == ["cpu"] and vae.offloaded == 1
    assert latent.shape[0] == 1


def test_encode_tube_offloads_the_vae_when_encode_raises():
    vae = _FakeVae()

    def boom(pixels):
        raise RuntimeError("oom")

    vae.module.encode = boom
    with pytest.raises(RuntimeError):
        R.encode_tube(SimpleNamespace(video_vae=vae), torch.zeros((5, 32, 32, 3)), "cpu")
    assert vae.offloaded == 1


def test_sample_tube_runs_the_h3_generator_as_a_locked_audio_refine(monkeypatch):
    seen = {}

    def fake_process(self, pipe_input, generation_outputs):
        seen["config"] = dict(self.config)
        seen["input"] = pipe_input.input
        return SimpleNamespace(output={"latent": ["frames"]})

    monkeypatch.setattr(R.TubeRefineGenerator, "process", fake_process)
    latent = torch.zeros(1, 4, 2, 4, 4)
    audio = AudioTrack(waveform=np.zeros((2, 10), np.float32), sample_rate=32000)
    out = R.sample_tube(
        object(), "cond", latent, audio, denoise=0.55, steps=6, video_sigma_shift=1.0,
        sampler="euler", scheduler="simple", device="cpu", seed=7, generation_outputs=lambda o: None)
    assert out == "frames"
    cfg = seen["config"]
    assert cfg["decode"] is False
    assert cfg["audio_source"] == "passthrough" and cfg["audio_refine"] == "lock"
    assert cfg["denoise"] == 0.55 and cfg["steps"] == 6 and cfg["video_sigma_shift"] == 1.0
    assert cfg["preview"] is False and cfg["quantity"] == 1
    assert seen["input"]["initial_latent"] == [latent]
    assert seen["input"]["audio"] == [audio]
    assert seen["input"]["conditioning"] == ["cond"] and seen["input"]["seed"] == [7]


def test_tube_refine_generator_decodes_the_refined_latent(monkeypatch):
    monkeypatch.setattr(GeneratorMinimaxH3Pipe, "generate_one", lambda self, ctx, i, s, p: "latent")
    monkeypatch.setattr(GeneratorMinimaxH3Pipe, "_decode_video", lambda self, c, latent: ("decoded", c, latent))
    gen = R.TubeRefineGenerator(config=R.TubeRefineGenerator.get_default_config())
    ctx = SimpleNamespace(extra="h3ctx")
    assert gen.generate_one(ctx, 0, 1, None) == ("decoded", "h3ctx", "latent")


def _patch_pipeline(monkeypatch, calls):
    monkeypatch.setattr(f"{_MOD}.load_clip_audio", lambda clip: (np.zeros((2, 32000 * 30), np.float32), 32000))

    def fake_encode(bundle, chunk, device):
        calls.setdefault("chunks", []).append(int(chunk.shape[0]))
        return torch.zeros(1), int(chunk.shape[0])

    def fake_sample(bundle, cond, latent, audio, **kw):
        calls.setdefault("audio", []).append(audio.waveform.shape[1])
        calls.setdefault("denoise", []).append(kw["denoise"])
        n = calls["chunks"][-1]
        return np.full((align_num_frames(n), 8, 8, 3), 9, np.uint8)

    monkeypatch.setattr(f"{_MOD}.encode_tube", fake_encode)
    monkeypatch.setattr(f"{_MOD}.sample_tube", fake_sample)


def test_refine_tube_pixels_splits_long_tubes_into_fifteen_second_windows(monkeypatch):
    calls = {}
    _patch_pipeline(monkeypatch, calls)
    pixels = torch.zeros(1, 3, 400, 8, 8)
    clip = TubeClip(source="/x.mp4", start_frame=0, num_frames=400, fps=24.0)
    out = R.refine_tube_pixels(object(), "c", pixels, clip, strength="balanced", device="cpu", seed=1,
                               generation_outputs=lambda o: None)
    assert calls["chunks"] == [345, 55]
    assert out.shape == (400, 8, 8, 3)
    assert calls["audio"] == [int(round(345 / 24 * 32000)), int(round(55 / 24 * 32000))]


@pytest.mark.parametrize("strength,denoise", [("light", 0.40), ("balanced", 0.55), ("strong", 0.70)])
def test_refine_tube_pixels_maps_strength_to_denoise(monkeypatch, strength, denoise):
    calls = {}
    _patch_pipeline(monkeypatch, calls)
    clip = TubeClip(source="/x.mp4", start_frame=10, num_frames=12, fps=24.0)
    out = R.refine_tube_pixels(object(), "c", torch.zeros(1, 3, 12, 8, 8), clip, strength=strength,
                               device="cpu", seed=1, generation_outputs=lambda o: None)
    assert calls["denoise"] == [denoise]
    assert out.shape[0] == 12
