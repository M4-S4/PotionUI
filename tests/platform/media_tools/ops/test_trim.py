from __future__ import annotations

import sys

import numpy as np
import pytest
import soundfile

from src.platform.media_tools import (
    InvalidToolParams,
    ToolCancelled,
    ToolContext,
    ToolFailed,
    probe,
    run_chain,
    validate_params,
)
from src.platform.media_tools.ops import trim
from src.platform.media_tools.ops.trim import head_tail_window, validated_window
from tests.platform.media_tools.media_factory import make_audio, make_video, run_ffmpeg
from tests.platform.media_tools.ops.time_helpers import (
    count_video_frames,
    fake_video,
    files_under,
    slow_context,
)

pytestmark = pytest.mark.usefixtures("need_ffmpeg")


@pytest.fixture
def video(tmp_path):
    return probe(make_video(tmp_path / "clip.mp4", seconds=2.0))


@pytest.fixture
def audio(tmp_path):
    return probe(make_audio(tmp_path / "tone.wav", seconds=2.0))


def run_trim(media, ctx, start, end):
    return run_chain([(trim.spec(), {"start": start, "end": end})], [media], ctx)


class TestParams:
    def test_end_is_required(self):
        with pytest.raises(InvalidToolParams) as caught:
            validate_params(trim.spec(), {"start": 0})
        assert caught.value.field == "end"

    def test_end_must_follow_start(self):
        with pytest.raises(InvalidToolParams) as caught:
            validate_params(trim.spec(), {"start": 1.5, "end": 1.0})
        assert caught.value.field == "end"

    def test_negative_start_is_refused(self):
        with pytest.raises(InvalidToolParams) as caught:
            validate_params(trim.spec(), {"start": -1, "end": 1.0})
        assert caught.value.field == "start"

    def test_start_defaults_to_zero(self):
        assert validate_params(trim.spec(), {"end": 1.0}) == {"start": 0.0, "end": 1.0}


class TestWindow:
    def test_end_within_tolerance_is_clamped_to_the_duration(self):
        assert validated_window(0.5, 2.03, 2.0) == (0.5, 2.0)

    def test_end_past_tolerance_is_refused(self):
        with pytest.raises(InvalidToolParams) as caught:
            validated_window(0.5, 2.5, 2.0)
        assert caught.value.field == "end"

    def test_start_past_the_end_is_refused(self):
        with pytest.raises(InvalidToolParams) as caught:
            validated_window(2.0, 2.01, 2.0)
        assert caught.value.field == "start"

    def test_unknown_duration_fails(self):
        with pytest.raises(Exception) as caught:
            validated_window(0.0, 1.0, None)
        assert "duration" in str(caught.value)


class TestRun:
    def test_trims_a_video_and_keeps_its_sound(self, video, ctx):
        (result,) = run_trim(video, ctx, 0.5, 1.5)
        assert result.kind == "video"
        assert (result.width, result.height) == (64, 48)
        assert result.duration == pytest.approx(1.0, abs=0.15)
        assert result.has_audio is True
        assert result.path.parent == ctx.scratch_dir

    def test_trims_audio(self, audio, ctx):
        (result,) = run_trim(audio, ctx, 0.5, 1.5)
        assert result.kind == "audio"
        assert result.duration == pytest.approx(1.0, abs=0.15)

    def test_a_video_without_sound_stays_silent(self, tmp_path, ctx):
        silent = probe(make_video(tmp_path / "mute.mp4", seconds=2.0, audio=False))
        (result,) = run_trim(silent, ctx, 0.0, 1.0)
        assert result.has_audio is False
        assert result.duration == pytest.approx(1.0, abs=0.15)

    def test_odd_sized_video_comes_out_even(self, tmp_path, ctx):
        odd = tmp_path / "odd.mp4"
        run_ffmpeg(
            "-f", "lavfi", "-i", "testsrc=size=63x47:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv444p", str(odd),
        )
        media = probe(odd)
        assert (media.width, media.height) == (63, 47)
        (result,) = run_trim(media, ctx, 0.0, 0.5)
        assert (result.width, result.height) == (62, 46)

    def test_end_past_the_clip_is_refused_before_encoding(self, video, ctx):
        with pytest.raises(InvalidToolParams) as caught:
            run_trim(video, ctx, 0.0, 5.0)
        assert caught.value.field == "end"
        assert ctx.commands == []

    def test_two_trims_are_one_pass_and_the_second_is_relative(self, video, ctx):
        steps = [
            (trim.spec(), {"start": 0.5, "end": 1.5}),
            (trim.spec(), {"start": 0.25, "end": 0.75}),
        ]
        (result,) = run_chain(steps, [video], ctx)
        assert len(ctx.commands) == 1
        command = ctx.commands[0]
        assert command[command.index("-ss") + 1] == "0.750000"
        assert command[command.index("-t") + 1] == "0.500000"
        assert result.duration == pytest.approx(0.5, abs=0.15)

    def test_second_trim_is_validated_against_the_first(self, video, ctx):
        steps = [
            (trim.spec(), {"start": 0.5, "end": 1.0}),
            (trim.spec(), {"start": 0.0, "end": 1.5}),
        ]
        with pytest.raises(InvalidToolParams) as caught:
            run_chain(steps, [video], ctx)
        assert caught.value.field == "end"


class TestHeadTailWindow:
    def test_head_keeps_exactly_the_requested_frames(self, video, ctx):
        start, end = head_tail_window(video, "head", frames=5)
        (result,) = run_trim(video, ctx, start, end)
        assert start == 0.0
        assert count_video_frames(result.path) == 5

    def test_tail_keeps_exactly_the_requested_frames(self, video, ctx):
        start, end = head_tail_window(video, "tail", frames=5)
        (result,) = run_trim(video, ctx, start, end)
        assert end == pytest.approx(2.0)
        assert count_video_frames(result.path) == 5

    @pytest.mark.parametrize("rate", ["24", "30000/1001", "25"])
    @pytest.mark.parametrize("keep", ["head", "tail"])
    def test_frame_counts_are_exact_at_awkward_frame_rates(self, tmp_path, ctx, rate, keep):
        clip = tmp_path / "rate.mp4"
        run_ffmpeg(
            "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={rate}:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(clip),
        )
        media = probe(clip)
        start, end = head_tail_window(media, "head" if keep == "head" else "tail", frames=7)
        (result,) = run_trim(media, ctx, start, end)
        assert count_video_frames(result.path) == 7

    def test_tail_trim_keeps_the_sound_when_fps_is_zero(self, video, ctx):
        start, end = head_tail_window(video.with_changes(fps=None), "tail", frames=5, fps=0.0)
        (result,) = run_trim(video, ctx, start, end)
        assert result.has_audio is True
        assert result.kind == "video"

    def test_frame_step_rounds_down_to_one_plus_a_multiple(self, video):
        _, end = head_tail_window(video, "head", frames=10, frame_step=4)
        assert end == pytest.approx(9 / 10 - 0.025)

    def test_seconds_override_frames(self, video):
        _, end = head_tail_window(video, "head", frames=99, seconds=0.5)
        assert end == pytest.approx(0.5 - 0.025)

    def test_explicit_fps_sets_the_window(self, video):
        _, end = head_tail_window(video, "head", frames=5, fps=20.0)
        assert end == pytest.approx(5 / 20 - 0.25 / 20)

    def test_unknown_fps_falls_back_to_24(self, video):
        _, end = head_tail_window(video.with_changes(fps=None), "head", frames=5)
        assert end == pytest.approx(5 / 24 - 0.25 / 24)

    def test_a_window_longer_than_the_clip_keeps_everything(self, video):
        assert head_tail_window(video, "head", frames=500) == (0.0, 2.0)
        assert head_tail_window(video, "tail", frames=500) == (0.0, 2.0)

    def test_unknown_edge_is_refused(self, video):
        with pytest.raises(InvalidToolParams) as caught:
            head_tail_window(video, "middle", frames=5)
        assert caught.value.field == "keep"


class TestRunnerBehaviour:
    def test_progress_rises_to_completion(self, video, tmp_path):
        seen = []
        ctx = ToolContext(tmp_path / "scratch", progress=lambda fraction, label: seen.append((fraction, label)))
        run_trim(video, ctx, 0.0, 1.5)
        fractions = [fraction for fraction, _ in seen]
        assert fractions == sorted(fractions)
        assert fractions[0] == 0.0
        assert fractions[-1] == 1.0
        assert any(0.0 < fraction < 1.0 for fraction in fractions)
        assert seen[0][1] == "Trim"

    def test_a_cancelled_context_stops_before_encoding(self, video, tmp_path):
        ctx = ToolContext(tmp_path / "scratch", is_cancelled=lambda: True)
        with pytest.raises(ToolCancelled):
            run_trim(video, ctx, 0.0, 1.0)
        assert ctx.commands == []
        assert files_under(ctx.scratch_dir) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as a stand-in encoder")
    def test_cancelling_a_running_encode_kills_it(self, tmp_path):
        ctx = slow_context(tmp_path)
        with pytest.raises(ToolCancelled):
            run_chain(
                [(trim.spec(), {"start": 0.0, "end": 1.0})],
                [fake_video(tmp_path / "x.mp4")],
                ctx,
            )
        assert len(ctx.commands) == 1


def test_trim_spec_is_registered_with_the_contract_shape():
    spec = trim.spec()
    assert spec.accepts == ("video", "audio")
    assert spec.cardinality == "one_to_one"
    assert [p["name"] for p in spec.params] == ["start", "end"]
    assert isinstance(spec.impl, trim.Trim)
    assert spec.impl.has_image_path() is False


class TestAudioFormat:
    @pytest.mark.parametrize("rate, channels", [(22050, 1), (44100, 2), (16000, 1), (48000, 2)])
    def test_sample_rate_channel_count_and_wav_container_survive_a_trim(self, tmp_path, ctx, rate, channels):
        source = tmp_path / "tone.wav"
        run_ffmpeg(
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration=2",
            "-ac", str(channels), str(source),
        )

        (result,) = run_trim(probe(source), ctx, 0.5, 1.5)

        info = soundfile.info(result.path)
        assert result.path.suffix == ".wav"
        assert (info.samplerate, info.channels) == (rate, channels)
        assert info.frames / info.samplerate == pytest.approx(1.0, abs=0.02)

    def test_the_window_starts_where_asked(self, tmp_path, ctx):
        source = tmp_path / "ramp.wav"
        run_ffmpeg(
            "-f", "lavfi", "-i", "aevalsrc=if(lt(t\\,1)\\,0.0\\,0.5):s=8000:d=2", "-ac", "1", str(source),
        )

        (result,) = run_trim(probe(source), ctx, 1.0, 2.0)

        samples, _ = soundfile.read(result.path)
        assert abs(samples[: len(samples) // 2]).mean() > 0.4

    def test_a_window_cut_from_the_start_keeps_the_sources_beginning(self, tmp_path, ctx):
        source = tmp_path / "ramp.wav"
        run_ffmpeg(
            "-f", "lavfi", "-i", "aevalsrc=if(lt(t\\,1)\\,0.5\\,0.0):s=8000:d=2", "-ac", "1", str(source),
        )

        (result,) = run_trim(probe(source), ctx, 0.0, 1.0)

        samples, _ = soundfile.read(result.path)
        assert abs(samples[: len(samples) // 2]).mean() > 0.4
        assert abs(samples[-800:]).mean() > 0.4


class TestLosslessAudio:
    @staticmethod
    def source(tmp_path, subtype, fmt="WAV", suffix=".wav"):
        samples = np.stack([
            0.3 * np.sin(2 * np.pi * 440 * np.arange(88200) / 44100),
            0.2 * np.sin(2 * np.pi * 660 * np.arange(88200) / 44100),
        ], axis=1)
        path = tmp_path / f"{subtype}{suffix}"
        soundfile.write(path, samples, 44100, subtype=subtype, format=fmt)
        return path

    @pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "PCM_U8"])
    def test_a_wav_keeps_its_sample_format(self, tmp_path, ctx, subtype):
        path = self.source(tmp_path, subtype)

        (result,) = run_trim(probe(path), ctx, 0.5, 1.5)

        assert soundfile.info(result.path).subtype == subtype

    @pytest.mark.parametrize("subtype", ["PCM_24", "PCM_32", "FLOAT"])
    def test_a_trimmed_wav_holds_the_sources_samples_unquantised(self, tmp_path, ctx, subtype):
        path = self.source(tmp_path, subtype)
        whole, _ = soundfile.read(path, dtype="float64")

        (result,) = run_trim(probe(path), ctx, 0.5, 1.5)

        kept, _ = soundfile.read(result.path, dtype="float64")
        assert len(kept) == 44100
        assert np.allclose(kept, whole[22050:66150], atol=1e-6)

    @pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
    def test_a_flac_keeps_its_bit_depth(self, tmp_path, ctx, subtype):
        path = self.source(tmp_path, subtype, "FLAC", ".flac")

        (result,) = run_trim(probe(path), ctx, 0.5, 1.5)

        assert result.path.suffix == ".flac"
        assert soundfile.info(result.path).subtype == subtype

    @pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
    def test_an_aiff_keeps_its_bit_depth(self, tmp_path, ctx, subtype):
        path = self.source(tmp_path, subtype, "AIFF", ".aiff")

        (result,) = run_trim(probe(path), ctx, 0.5, 1.5)

        assert soundfile.info(result.path).subtype == subtype

    def test_the_probe_reports_the_codec_and_depth(self, tmp_path):
        media = probe(self.source(tmp_path, "PCM_24"))

        assert (media.audio_codec, media.bit_depth) == ("pcm_s24le", 24)

    def test_a_lossy_source_is_not_widened(self, tmp_path, ctx):
        path = tmp_path / "tone.mp3"
        run_ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2", str(path))

        (result,) = run_trim(probe(path), ctx, 0.5, 1.5)

        assert result.path.suffix == ".mp3"


class QuietRunner:
    def __init__(self):
        self.commands = []

    def run(self, args, *, duration=None, on_progress=None, is_cancelled=None):
        self.commands.append(list(args))


def test_an_encode_that_leaves_no_file_fails_instead_of_returning_a_missing_clip(video, tmp_path):
    ctx = ToolContext(tmp_path / "scratch", runner=QuietRunner())

    with pytest.raises(ToolFailed, match="empty file"):
        run_trim(video, ctx, 0.0, 1.0)
