from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.platform.media_tools import (
    InvalidSelection,
    InvalidToolParams,
    ToolCancelled,
    ToolContext,
    ToolMedia,
    UnsupportedMedia,
    probe,
    run_chain,
    validate_params,
)
from src.platform.media_tools.ops import concat
from src.platform.media_tools.ops.concat import (
    blend_chain,
    check_layout,
    concat_args,
    fit_filter,
    join_clips,
    kept_lengths,
    trim_filter,
)
from tests.platform.media_tools.media_factory import make_audio, make_video, run_ffmpeg
from tests.platform.media_tools.ops.time_helpers import count_video_frames, fake_video, files_under, slow_context

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


def listed(path, width, height, duration, fps, audio):
    return ToolMedia(
        path=Path(path), kind="video", width=width, height=height, duration=duration, fps=fps, has_audio=audio,
    )


def native(path):
    return str(Path(path))


PLAIN_ARGS = [
    "-i", native("/c/a.mp4"), "-i", native("/c/w.mp4"), "-filter_complex",
    "[0:v]fps=10,scale=64:48:force_original_aspect_ratio=decrease,pad=64:48:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v0];"
    "[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];"
    "[1:v]fps=10,scale=64:48:force_original_aspect_ratio=decrease,pad=64:48:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v1];"
    "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
    "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]",
    "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "192k",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", native("/c/out.mp4"),
]

COVER_SILENT_ARGS = [
    "-i", native("/c/w.mp4"), "-i", native("/c/a.mp4"), "-i", native("/c/m.mp4"), "-f", "lavfi", "-t", "1.000", "-i", "anullsrc=r=48000:cl=stereo",
    "-filter_complex",
    "[0:v]fps=10,scale=96:64:force_original_aspect_ratio=increase,crop=96:64,setsar=1,format=yuv420p[v0];"
    "[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];"
    "[1:v]fps=10,scale=96:64:force_original_aspect_ratio=increase,crop=96:64,setsar=1,format=yuv420p[v1];"
    "[1:a]aresample=48000,aformat=channel_layouts=stereo[a1];"
    "[2:v]fps=10,scale=96:64:force_original_aspect_ratio=increase,crop=96:64,setsar=1,format=yuv420p[v2];"
    "[3:a]aresample=48000,aformat=channel_layouts=stereo[a2];"
    "[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]",
    "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "192k",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", native("/c/out.mp4"),
]

FADE_GRAPH_TAIL = (
    "[v0][v1]xfade=transition=fade:duration=0.500000:offset=1.500000[vj1];"
    "[a0][a1]acrossfade=d=0.500000[aj1];"
    "[vj1][v2]concat=n=2:v=1:a=0[v];"
    "[aj1][a2]concat=n=2:v=0:a=1[a]"
)


class TestUntrimmedCommandsAreUnchanged:
    @pytest.fixture
    def listing(self):
        return {
            "a": listed("/c/a.mp4", 64, 48, 2.0, 10.0, True),
            "w": listed("/c/w.mp4", 96, 64, 1.0, 10.0, True),
            "m": listed("/c/m.mp4", 63, 47, 1.0, 10.0, False),
        }

    def test_a_plain_join(self, listing):
        args = concat_args(
            [listing["a"], listing["w"]], Path("/c/out.mp4"), crossfades=[0.0], audio=True, fit="pad",
        )
        assert args == PLAIN_ARGS

    def test_a_covered_join_with_a_silent_clip(self, listing):
        args = concat_args(
            [listing["w"], listing["a"], listing["m"]], Path("/c/out.mp4"), crossfades=[0.0, 0.0], audio=True, fit="cover",
        )
        assert args == COVER_SILENT_ARGS

    def test_a_crossfade_keeps_its_offsets_and_timebase(self, listing):
        args = concat_args(
            [listing["a"], listing["w"], listing["m"]], Path("/c/out.mp4"), crossfades=[0.5, 0.0], audio=True, fit="pad",
        )
        graph = args[args.index("-filter_complex") + 1]
        assert graph.endswith(FADE_GRAPH_TAIL)
        assert graph.count(",settb=AVTB[v") == 3

    def test_a_silent_join(self, listing):
        args = concat_args(
            [listing["a"], listing["w"]], Path("/c/out.mp4"), crossfades=[0.0], audio=False, fit="pad",
        )
        graph = args[args.index("-filter_complex") + 1]
        assert graph.endswith("[v0][v1]concat=n=2:v=1:a=0[v]")
        assert "-c:a" not in args and "[a]" not in args


class TestTrims:
    def test_trim_filter_nudges_both_edges(self):
        assert trim_filter("trim", 2.0, 0.5, 0.25) == "trim=start=0.499000:end=1.749000,setpts=PTS-STARTPTS,"
        assert trim_filter("atrim", 2.0, 0.5, 0.0) == "atrim=start=0.499000,asetpts=PTS-STARTPTS,"
        assert trim_filter("trim", 2.0, 0.0, 0.25) == "trim=end=1.749000,setpts=PTS-STARTPTS,"
        assert trim_filter("trim", 2.0, 0.0, 0.0) == ""

    def test_kept_lengths_subtract_both_edges(self):
        clips = [listed("/c/a.mp4", 64, 48, 2.0, 10.0, True), listed("/c/w.mp4", 64, 48, 1.0, 10.0, True)]
        assert kept_lengths(clips, [(0.0, 0.0), (0.25, 0.25)]) == [2.0, 0.5]

    @pytest.mark.parametrize("trims", [[(0.0, 0.0), (1.0, 0.0)], [(0.0, 0.0), (0.6, 0.6)], [(-0.1, 0.0), (0.0, 0.0)]])
    def test_a_clip_with_nothing_left_is_refused(self, trims, ctx):
        clips = [listed("/c/a.mp4", 64, 48, 2.0, 10.0, True), listed("/c/w.mp4", 64, 48, 1.0, 10.0, True)]
        with pytest.raises(InvalidToolParams) as caught:
            join_clips(clips, ctx, trims=trims)
        assert caught.value.field == "trims"
        assert ctx.commands == []

    def test_trimming_the_start_of_the_second_clip_shortens_the_join(self, clips, ctx):
        result = join_clips([clips["a"], clips["wide"]], ctx, trims=[(0.0, 0.0), (0.3, 0.0)])
        assert result.duration == pytest.approx(2.7, abs=0.2)
        assert result.has_audio is True
        text = command_text(ctx)
        assert "trim=start=0.299000" in text
        assert "atrim=start=0.299000" in text

    def test_trimming_the_end_of_the_first_clip_shortens_the_join(self, clips, ctx):
        result = join_clips([clips["a"], clips["wide"]], ctx, trims=[(0.0, 0.5), (0.0, 0.0)])
        assert result.duration == pytest.approx(2.5, abs=0.2)
        assert "trim=end=1.499000" in command_text(ctx)

    def test_trims_feed_the_crossfade_offset_with_the_kept_length(self, clips, ctx):
        result = join_clips([clips["a"], clips["wide"]], ctx, trims=[(0.0, 0.5), (0.0, 0.0)], crossfades=[0.3])
        assert "offset=1.200000" in command_text(ctx)
        assert result.duration == pytest.approx(2.2, abs=0.2)

    def test_a_trim_count_that_does_not_match_the_clips_is_refused(self, clips, ctx):
        with pytest.raises(InvalidToolParams) as caught:
            join_clips([clips["a"], clips["wide"]], ctx, trims=[(0.0, 0.0)])
        assert caught.value.field == "trims"
        assert ctx.commands == []


class TestCanvasClip:
    def test_a_backward_extend_takes_the_canvas_from_the_source_clip(self, clips, ctx):
        extension, source = clips["small"], clips["a"]
        result = join_clips([extension, source], ctx, canvas=1, fit="cover")
        assert (result.width, result.height) == (64, 48)
        text = command_text(ctx)
        assert "crop=64:48" in text
        assert "crop=32:32" not in text

    def test_without_a_canvas_index_the_first_clip_still_decides(self, clips, ctx):
        result = join_clips([clips["small"], clips["a"]], ctx, fit="cover")
        assert (result.width, result.height) == (32, 32)

    def test_the_frame_rate_comes_from_the_canvas_clip(self, tmp_path, clips, ctx):
        quick = probe(make_video(tmp_path / "quick.mp4", seconds=1.0, size=(64, 48), fps=20))
        result = join_clips([quick, clips["a"]], ctx, canvas=1)
        assert "fps=10" in command_text(ctx)
        assert result.fps == pytest.approx(10.0)

    def test_the_canvas_size_is_made_even(self, tmp_path, clips, ctx):
        odd = tmp_path / "odd.mp4"
        run_ffmpeg(
            "-f", "lavfi", "-i", "testsrc=size=63x47:rate=10:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv444p", str(odd),
        )
        result = join_clips([clips["a"], probe(odd)], ctx, canvas=1)
        assert (result.width, result.height) == (62, 46)

    @pytest.mark.parametrize("canvas", [-1, 2])
    def test_a_canvas_that_is_not_a_clip_is_refused(self, clips, ctx, canvas):
        with pytest.raises(InvalidToolParams) as caught:
            join_clips([clips["a"], clips["wide"]], ctx, canvas=canvas)
        assert caught.value.field == "canvas"
        assert ctx.commands == []


class TestJoinClips:
    def test_the_tool_and_the_function_encode_the_same_command(self, clips, tmp_path):
        tool_ctx = ToolContext(tmp_path / "tool")
        direct_ctx = ToolContext(tmp_path / "direct")
        run_concat([clips["a"], clips["wide"]], tool_ctx, fit="cover", keep_audio=False)
        join_clips([clips["a"], clips["wide"]], direct_ctx, fit="cover", audio=False)

        def normalised(context):
            return [part.replace(str(context.scratch_dir), "<out>") for part in context.commands[0]]

        assert normalised(tool_ctx) == normalised(direct_ctx)

    def test_a_wrong_crossfade_count_is_refused(self, clips, ctx):
        with pytest.raises(InvalidToolParams) as caught:
            join_clips([clips["a"], clips["wide"]], ctx, crossfades=[0.1, 0.1])
        assert caught.value.field == "crossfades"

    def test_no_clips_is_refused(self, ctx):
        with pytest.raises(InvalidToolParams):
            join_clips([], ctx)

    def test_a_cancelled_join_leaves_no_file_behind(self, clips, tmp_path):
        ctx = ToolContext(tmp_path / "scratch", is_cancelled=lambda: True)
        with pytest.raises(ToolCancelled):
            join_clips([clips["a"], clips["wide"]], ctx)
        assert files_under(ctx.scratch_dir) == []


class TestExtendFrameCounts:
    def test_a_trimmed_head_and_a_crossfade_give_the_expected_frame_count(self, tmp_path, ctx):
        first = probe(make_video(tmp_path / "first.mp4", seconds=2.0, size=(32, 24), fps=8, audio=False))
        second = probe(make_video(tmp_path / "second.mp4", seconds=2.0, size=(32, 24), fps=8, audio=False))

        result = join_clips([first, second], ctx, trims=[(0.0, 0.0), (0.25, 0.0)], crossfades=[0.375])

        assert count_video_frames(result.path) == 16 + 14 - 3

    def test_a_plain_join_has_the_sum_of_the_frames(self, tmp_path, ctx):
        first = probe(make_video(tmp_path / "first.mp4", seconds=1.0, size=(32, 24), fps=8, audio=False))
        second = probe(make_video(tmp_path / "second.mp4", seconds=2.0, size=(32, 24), fps=8, audio=False))

        result = join_clips([first, second], ctx)

        assert count_video_frames(result.path) == 8 + 16

    def test_a_trimmed_tail_and_a_crossfade_give_the_expected_frame_count(self, tmp_path, ctx):
        extension = probe(make_video(tmp_path / "extension.mp4", seconds=2.0, size=(32, 24), fps=8, audio=False))
        source = probe(make_video(tmp_path / "source.mp4", seconds=2.0, size=(32, 24), fps=8, audio=False))

        result = join_clips([extension, source], ctx, trims=[(0.0, 0.25), (0.0, 0.0)], crossfades=[0.25], canvas=1)

        assert count_video_frames(result.path) == 14 + 16 - 2


class TestNarrowerClipFit:
    @pytest.fixture
    def white(self, tmp_path):
        def make_white(name, width):
            path = tmp_path / f"{name}.mp4"
            run_ffmpeg(
                "-f", "lavfi", "-i", f"color=c=white:size={width}x128:rate=10:duration=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
            )
            return probe(path)

        return make_white

    def last_frame(self, path):
        capture = cv2.VideoCapture(str(path))
        last = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last = frame
        capture.release()
        return last

    @pytest.mark.parametrize("fit, edges_are_lit", [("pad", False), ("cover", True)])
    def test_black_bars_only_show_when_padding(self, white, ctx, fit, edges_are_lit):
        source, extension = white("source", 72), white("extension", 56)

        result = join_clips([source, extension], ctx, fit=fit)

        frame = self.last_frame(result.path)
        assert frame.shape[:2] == (128, 72)
        edges = float(np.concatenate([frame[:, :4], frame[:, -4:]]).mean())
        assert (edges > 224) is edges_are_lit
        assert (edges < 32) is (not edges_are_lit)

    def test_the_source_fills_the_frame_when_it_comes_second_and_sets_the_canvas(self, white, ctx):
        source, extension = white("source", 72), white("extension", 56)

        result = join_clips([extension, source], ctx, fit="cover", canvas=1)

        assert (result.width, result.height) == (72, 128)
        first_frame = cv2.VideoCapture(str(result.path)).read()[1]
        assert float(np.concatenate([first_frame[:, :4], first_frame[:, -4:]]).mean()) > 224
