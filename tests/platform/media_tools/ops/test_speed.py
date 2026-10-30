from __future__ import annotations

import math

import pytest

from src.platform.media_tools import InvalidToolParams, ToolMedia, probe, run_chain, validate_params
from src.platform.media_tools.ops import speed, trim
from src.platform.media_tools.ops.speed import audio_filters, tempo_factors
from tests.platform.media_tools.media_factory import make_audio, make_video
from tests.platform.media_tools.ops.time_helpers import count_video_frames

pytestmark = pytest.mark.usefixtures("need_ffmpeg")


@pytest.fixture
def video(tmp_path):
    return probe(make_video(tmp_path / "clip.mp4", seconds=2.0))


@pytest.fixture
def audio(tmp_path):
    return probe(make_audio(tmp_path / "tone.wav", seconds=2.0))


def run_speed(media, ctx, **params):
    return run_chain([(speed.spec(), params)], [media], ctx)


def command_text(ctx, index=0):
    return " ".join(ctx.commands[index])


class TestTempoFactors:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.25, [0.5, 0.5]), (4.0, [2.0, 2.0]), (1.5, [1.5]), (2.0, [2.0]), (0.5, [0.5])],
    )
    def test_known_chains(self, value, expected):
        assert tempo_factors(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", [0.25, 0.3, 0.7, 1.0, 1.9, 2.5, 3.3, 4.0])
    def test_every_factor_is_in_range_and_multiplies_to_the_speed(self, value):
        factors = tempo_factors(value)
        assert all(0.5 <= factor <= 2.0 for factor in factors)
        assert math.prod(factors) == pytest.approx(value)

    def test_keep_pitch_uses_atempo_only(self):
        filters = audio_filters(4.0, True)
        assert filters == ["atempo=2", "atempo=2"]

    def test_pitch_shift_resamples_instead_of_stretching(self):
        filters = audio_filters(2.0, False)
        assert filters == ["aresample=48000", "asetrate=48000*2", "aresample=48000"]


class TestParams:
    def test_speed_below_range_is_refused(self):
        with pytest.raises(InvalidToolParams) as caught:
            validate_params(speed.spec(), {"speed": 0.1})
        assert caught.value.field == "speed"

    def test_speed_above_range_is_refused(self):
        with pytest.raises(InvalidToolParams) as caught:
            validate_params(speed.spec(), {"speed": 5})
        assert caught.value.field == "speed"

    def test_defaults(self):
        assert validate_params(speed.spec(), {}) == {"speed": 2.0, "keep_pitch": True}


class TestVideo:
    def test_double_speed_halves_the_length_and_keeps_the_frame_rate(self, video, ctx):
        (result,) = run_speed(video, ctx, speed=2)
        assert result.kind == "video"
        assert (result.width, result.height) == (64, 48)
        assert result.duration == pytest.approx(1.0, abs=0.15)
        assert result.fps == pytest.approx(10.0)
        assert result.has_audio is True
        assert count_video_frames(result.path) == pytest.approx(10, abs=1)

    def test_half_speed_doubles_the_length(self, video, ctx):
        (result,) = run_speed(video, ctx, speed=0.5)
        assert result.duration == pytest.approx(4.0, abs=0.2)
        assert result.has_audio is True

    def test_keep_pitch_stretches_the_audio_with_atempo(self, video, ctx):
        run_speed(video, ctx, speed=2, keep_pitch=True)
        text = command_text(ctx)
        assert "atempo=2" in text
        assert "asetrate" not in text
        assert "setpts=PTS/2" in text

    def test_dropping_pitch_resamples_the_audio(self, video, ctx):
        (result,) = run_speed(video, ctx, speed=2, keep_pitch=False)
        text = command_text(ctx)
        assert "asetrate=48000*2" in text
        assert "atempo" not in text
        assert result.duration == pytest.approx(1.0, abs=0.15)
        assert result.has_audio is True

    def test_four_times_speed_chains_two_tempo_filters(self, video, ctx):
        run_speed(video, ctx, speed=4)
        assert "atempo=2,atempo=2" in command_text(ctx)

    def test_a_video_without_sound_gets_no_audio_filter(self, tmp_path, ctx):
        silent = probe(make_video(tmp_path / "mute.mp4", seconds=2.0, audio=False))
        (result,) = run_speed(silent, ctx, speed=2)
        assert "atempo" not in command_text(ctx)
        assert result.has_audio is False
        assert result.duration == pytest.approx(1.0, abs=0.15)


class TestAudio:
    def test_audio_speed_changes_length_without_a_video_filter(self, audio, ctx):
        (result,) = run_speed(audio, ctx, speed=2)
        assert result.kind == "audio"
        assert result.duration == pytest.approx(1.0, abs=0.15)
        text = command_text(ctx)
        assert "-vn" in ctx.commands[0]
        assert "setpts" not in text
        assert "atempo=2" in text

    def test_slow_audio_keeps_pitch_by_default(self, audio, ctx):
        (result,) = run_speed(audio, ctx, speed=0.25)
        assert result.duration == pytest.approx(8.0, abs=0.3)
        assert "atempo=0.5,atempo=0.5" in command_text(ctx)


class TestFusion:
    def test_trim_then_speed_on_a_video_is_one_ffmpeg_pass(self, video, ctx):
        steps = [
            (trim.spec(), {"start": 0.5, "end": 1.5}),
            (speed.spec(), {"speed": 2}),
        ]
        (result,) = run_chain(steps, [video], ctx)
        assert len(ctx.commands) == 1
        command = ctx.commands[0]
        assert command[command.index("-ss") + 1] == "0.500000"
        assert command[command.index("-t") + 1] == "1.000000"
        assert "atempo=2" in command_text(ctx)
        assert "setpts=PTS/2" in command_text(ctx)
        assert result.duration == pytest.approx(0.5, abs=0.15)
        assert result.has_audio is True

    def test_trim_then_speed_on_audio_is_one_ffmpeg_pass(self, audio, ctx):
        steps = [
            (trim.spec(), {"start": 0.5, "end": 1.5}),
            (speed.spec(), {"speed": 2}),
        ]
        (result,) = run_chain(steps, [audio], ctx)
        assert len(ctx.commands) == 1
        command = ctx.commands[0]
        assert "-ss" in command and "-t" in command
        assert result.kind == "audio"
        assert result.duration == pytest.approx(0.5, abs=0.15)

    def test_speed_then_trim_cuts_in_the_sped_up_timeline(self, video, ctx):
        steps = [
            (speed.spec(), {"speed": 2}),
            (trim.spec(), {"start": 0.0, "end": 0.5}),
        ]
        (result,) = run_chain(steps, [video], ctx)
        assert len(ctx.commands) == 1
        text = command_text(ctx)
        assert "-ss" not in ctx.commands[0]
        assert "trim=start=0.000000:end=0.500000" in text
        assert "atrim=start=0.000000:end=0.500000" in text
        assert result.duration == pytest.approx(0.5, abs=0.15)

    def test_speed_fragment_reports_the_new_duration(self):
        media = ToolMedia(path=None, kind="video", width=64, height=48, duration=4.0, fps=10.0, has_audio=True)
        fragment = speed.spec().impl.video_filter(media, {"speed": 4, "keep_pitch": True})
        assert fragment.duration == pytest.approx(1.0)
        assert fragment.vf == ["setpts=PTS/4", "fps=10"]
