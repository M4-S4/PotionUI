from __future__ import annotations

import sys

import pytest

from src.platform.media_tools import (
    InvalidToolParams,
    ToolCancelled,
    ToolContext,
    UnsupportedMedia,
    probe,
    run_chain,
    validate_params,
)
from src.platform.media_tools.ops import frame
from tests.platform.media_tools.media_factory import make_audio, make_video
from tests.platform.media_tools.ops.time_helpers import fake_video, files_under, slow_context

pytestmark = pytest.mark.usefixtures("need_ffmpeg")


@pytest.fixture
def video(tmp_path):
    return probe(make_video(tmp_path / "clip.mp4", seconds=2.0, size=(64, 48)))


def run_frame(media, ctx, time):
    return run_chain([(frame.spec(), {"time": time})], [media], ctx)


class TestRun:
    def test_extracts_a_png_with_the_video_size(self, video, ctx):
        (result,) = run_frame(video, ctx, 1.0)
        assert result.kind == "image"
        assert (result.width, result.height) == (64, 48)
        assert result.path.suffix == ".png"
        assert result.path.parent == ctx.scratch_dir

    def test_time_zero_is_the_first_frame(self, video, ctx):
        (result,) = run_frame(video, ctx, 0)
        assert result.kind == "image"
        assert len(ctx.commands) == 1

    def test_different_times_give_different_pictures(self, video, ctx):
        (early,) = run_frame(video, ctx, 0.0)
        (late,) = run_frame(video, ctx, 1.5)
        assert early.path.read_bytes() != late.path.read_bytes()

    def test_a_time_after_the_last_frame_still_gives_the_last_frame(self, video, ctx):
        (result,) = run_frame(video, ctx, 1.99)
        assert result.kind == "image"
        assert result.path.stat().st_size > 0
        assert len(ctx.commands) == 2

    def test_seek_is_before_the_input(self, video, ctx):
        run_frame(video, ctx, 1.25)
        command = ctx.commands[0]
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-ss") + 1] == "1.250000"
        assert "-frames:v" in command


class TestRefusals:
    def test_time_at_the_end_is_refused(self, video, ctx):
        with pytest.raises(InvalidToolParams) as caught:
            run_frame(video, ctx, video.duration)
        assert caught.value.field == "time"
        assert ctx.commands == []

    def test_time_past_the_end_is_refused(self, video, ctx):
        with pytest.raises(InvalidToolParams) as caught:
            run_frame(video, ctx, 10)
        assert caught.value.field == "time"

    def test_negative_time_is_refused_by_validation(self):
        with pytest.raises(InvalidToolParams) as caught:
            validate_params(frame.spec(), {"time": -1})
        assert caught.value.field == "time"

    def test_only_video_is_accepted(self, tmp_path, ctx):
        audio = probe(make_audio(tmp_path / "tone.wav"))
        with pytest.raises(UnsupportedMedia):
            run_frame(audio, ctx, 0.5)


class TestCancel:
    def test_a_cancelled_context_stops_before_encoding(self, video, tmp_path):
        ctx = ToolContext(tmp_path / "scratch", is_cancelled=lambda: True)
        with pytest.raises(ToolCancelled):
            run_frame(video, ctx, 0.5)
        assert files_under(ctx.scratch_dir) == []

    @pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as a stand-in encoder")
    def test_cancelling_a_running_extract_leaves_nothing_behind(self, tmp_path):
        ctx = slow_context(tmp_path)
        with pytest.raises(ToolCancelled):
            run_frame(fake_video(tmp_path / "x.mp4"), ctx, 1.0)
        assert files_under(ctx.scratch_dir) == []


def test_spec_declares_video_to_image():
    spec = frame.spec()
    assert spec.accepts == ("video",)
    assert spec.produces == "image"
    assert spec.editor == "frame"
