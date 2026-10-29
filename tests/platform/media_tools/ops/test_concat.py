from __future__ import annotations

import sys

import pytest

from src.platform.media_tools import (
    InvalidSelection,
    InvalidToolParams,
    ToolCancelled,
    ToolContext,
    UnsupportedMedia,
    probe,
    run_chain,
    validate_params,
)
from src.platform.media_tools.ops import concat
from src.platform.media_tools.ops.concat import blend_chain, check_layout, fit_filter
from tests.platform.media_tools.media_factory import make_audio, make_video, run_ffmpeg
from tests.platform.media_tools.ops.time_helpers import fake_video, files_under, slow_context

pytestmark = pytest.mark.usefixtures("need_ffmpeg")


@pytest.fixture
def clips(tmp_path):
    return {
        "a": probe(make_video(tmp_path / "a.mp4", seconds=2.0, size=(64, 48))),
        "wide": probe(make_video(tmp_path / "wide.mp4", seconds=1.0, size=(96, 64), freq=660)),
        "small": probe(make_video(tmp_path / "small.mp4", seconds=1.0, size=(32, 32), freq=880)),
        "mute": probe(make_video(tmp_path / "mute.mp4", seconds=1.0, size=(64, 48), audio=False)),
    }


def run_concat(media, ctx, **params):
    return run_chain([(concat.spec(), params)], list(media), ctx)


def command_text(ctx):
    return " ".join(ctx.commands[0])


class TestJoin:
    def test_joins_two_clips_end_to_end(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"]], ctx)
        assert result.kind == "video"
        assert (result.width, result.height) == (64, 48)
        assert result.duration == pytest.approx(3.0, abs=0.2)
        assert result.has_audio is True
        assert result.path.suffix == ".mp4"
        assert result.path.parent == ctx.scratch_dir
        assert len(ctx.commands) == 1

    def test_three_clips_of_different_sizes_take_the_first_clips_canvas_when_padded(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"], clips["small"]], ctx, fit="pad")
        assert (result.width, result.height) == (64, 48)
        assert result.duration == pytest.approx(4.0, abs=0.25)
        text = command_text(ctx)
        assert "pad=64:48" in text
        assert "crop=64:48" not in text

    def test_three_clips_of_different_sizes_take_the_first_clips_canvas_when_covered(self, clips, ctx):
        (result,) = run_concat([clips["wide"], clips["a"], clips["small"]], ctx, fit="cover")
        assert (result.width, result.height) == (96, 64)
        text = command_text(ctx)
        assert "crop=96:64" in text
        assert "pad=" not in text

    def test_the_frame_rate_of_the_first_clip_is_used(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"]], ctx)
        assert result.fps == pytest.approx(10.0)
        assert "fps=10" in command_text(ctx)

    def test_an_odd_first_clip_gives_an_even_canvas(self, tmp_path, clips, ctx):
        odd = tmp_path / "odd.mp4"
        run_ffmpeg(
            "-f", "lavfi", "-i", "testsrc=size=63x47:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv444p", str(odd),
        )
        (result,) = run_concat([probe(odd), clips["a"]], ctx)
        assert (result.width, result.height) == (62, 46)

    def test_reports_progress_up_to_completion(self, clips, tmp_path):
        seen = []
        ctx = ToolContext(tmp_path / "scratch", progress=lambda fraction, label: seen.append(fraction))
        run_concat([clips["a"], clips["wide"]], ctx)
        assert seen == sorted(seen)
        assert seen[-1] == 1.0
        assert any(0.0 < fraction < 1.0 for fraction in seen)


class TestAudio:
    def test_a_silent_clip_among_clips_with_sound_still_gives_sound(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["mute"], clips["wide"]], ctx)
        assert result.has_audio is True
        assert "anullsrc" in command_text(ctx)
        assert result.duration == pytest.approx(4.0, abs=0.25)

    def test_all_silent_clips_give_a_silent_video(self, clips, tmp_path, ctx):
        other = probe(make_video(tmp_path / "other.mp4", seconds=1.0, audio=False))
        (result,) = run_concat([clips["mute"], other], ctx)
        assert result.has_audio is False
        assert "anullsrc" not in command_text(ctx)

    def test_keep_audio_off_drops_the_sound(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"]], ctx, keep_audio=False)
        assert result.has_audio is False
        assert "[a]" not in command_text(ctx)


class TestCrossfade:
    def test_crossfade_shortens_the_result_by_the_overlap(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"]], ctx, overlap="crossfade", overlap_frames=5)
        assert result.duration == pytest.approx(2.5, abs=0.2)
        text = command_text(ctx)
        assert "xfade=transition=fade:duration=0.500000:offset=1.500000" in text
        assert "acrossfade=d=0.500000" in text
        assert result.has_audio is True

    def test_crossfade_over_three_clips_chains_the_offsets(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"], clips["small"]], ctx, overlap="crossfade", overlap_frames=3)
        text = command_text(ctx)
        assert "offset=1.700000" in text
        assert "offset=2.400000" in text
        assert result.duration == pytest.approx(3.4, abs=0.25)

    def test_crossfade_with_zero_frames_is_a_plain_join(self, clips, ctx):
        (result,) = run_concat([clips["a"], clips["wide"]], ctx, overlap="crossfade", overlap_frames=0)
        assert "xfade" not in command_text(ctx)
        assert result.duration == pytest.approx(3.0, abs=0.2)

    def test_cut_ignores_the_overlap_frames(self, clips, ctx):
        run_concat([clips["a"], clips["wide"]], ctx, overlap="cut", overlap_frames=5)
        assert "xfade" not in command_text(ctx)

    def test_overlap_frames_are_converted_with_the_first_clips_frame_rate(self, clips, ctx):
        run_concat([clips["a"], clips["wide"]], ctx, overlap="crossfade", overlap_frames=4)
        assert "duration=0.400000" in command_text(ctx)

    def test_a_crossfade_longer_than_a_clip_is_refused(self, clips, ctx):
        with pytest.raises(InvalidToolParams) as caught:
            run_concat([clips["a"], clips["wide"]], ctx, overlap="crossfade", overlap_frames=10)
        assert caught.value.field == "overlap_frames"
        assert ctx.commands == []

    def test_overlap_frames_is_ignored_unless_crossfading(self):
        resolved = validate_params(concat.spec(), {"overlap": "cut", "overlap_frames": 5})
        assert "overlap_frames" not in resolved


class TestSelection:
    def test_one_clip_is_not_enough(self, clips, ctx):
        with pytest.raises(InvalidSelection):
            run_concat([clips["a"]], ctx)

    def test_nine_clips_are_too_many(self, clips, ctx):
        with pytest.raises(InvalidSelection):
            run_concat([clips["a"]] * 9, ctx)

    def test_eight_clips_are_allowed_by_the_selection_check(self, clips, ctx):
        (result,) = run_concat([clips["mute"]] * 8, ctx, keep_audio=False)
        assert result.duration == pytest.approx(8.0, abs=0.4)

    def test_audio_is_not_accepted(self, clips, tmp_path, ctx):
        tone = probe(make_audio(tmp_path / "tone.wav"))
        with pytest.raises(UnsupportedMedia):
            run_concat([clips["a"], tone], ctx)

    def test_a_clip_with_unknown_length_is_refused(self, clips, tmp_path, ctx):
        unknown = clips["a"].with_changes(duration=None)
        with pytest.raises(UnsupportedMedia):
            run_concat([clips["a"], unknown], ctx)


class TestHelpers:
    def test_fit_filters(self):
        assert fit_filter(64, 48, "pad") == (
            "scale=64:48:force_original_aspect_ratio=decrease,pad=64:48:(ow-iw)/2:(oh-ih)/2"
        )
        assert fit_filter(64, 48, "cover") == (
            "scale=64:48:force_original_aspect_ratio=increase,crop=64:48"
        )

    def test_layout_check_accepts_a_fade_shorter_than_both_sides(self):
        check_layout([2.0, 1.0], [0.5])

    def test_layout_check_refuses_a_fade_as_long_as_a_clip(self):
        with pytest.raises(InvalidToolParams):
            check_layout([2.0, 1.0], [1.0])

    def test_blend_chain_offsets_follow_the_joined_length(self):
        filters = blend_chain([2.0, 1.0, 1.0], [0.5, 0.0], False)
        assert filters == [
            "[v0][v1]xfade=transition=fade:duration=0.500000:offset=1.500000[vj1]",
            "[vj1][v2]concat=n=2:v=1:a=0[v]",
        ]


class TestCancel:
    def test_a_cancelled_context_stops_before_encoding(self, clips, tmp_path):
        ctx = ToolContext(tmp_path / "scratch", is_cancelled=lambda: True)
        with pytest.raises(ToolCancelled):
            run_concat([clips["a"], clips["wide"]], ctx)
        assert ctx.commands == []
        assert files_under(ctx.scratch_dir) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as a stand-in encoder")
    def test_cancelling_a_running_join_leaves_nothing_behind(self, tmp_path):
        ctx = slow_context(tmp_path)
        media = [fake_video(tmp_path / "x.mp4"), fake_video(tmp_path / "y.mp4")]
        with pytest.raises(ToolCancelled):
            run_concat(media, ctx)
        assert files_under(ctx.scratch_dir) == []


def test_spec_is_many_to_one_with_a_selection():
    spec = concat.spec()
    assert spec.cardinality == "many_to_one"
    assert spec.selection == {"min": 2, "max": 8}
    assert spec.accepts == ("video",)
