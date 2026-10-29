import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.features.generation.engine import validate_pipe_configuration
from src.pipelines.contracts import IOType, PipeInput
from src.pipelines.outputs import GenerationExecutionError, ProgressGenerationOutput
from src.pipelines.pipes._shared.media import tools as pipe_tools
from src.pipelines.pipes.video_trim.main import HEAD_TAIL, KEPT_CRF, VideoTrimPipe
from src.platform.media_tools import ToolMedia, probe, validate_params
from src.platform.util.video_transcode import find_ffmpeg
from tests.platform.media_tools.media_factory import ffmpeg_available, make_video
from tests.platform.media_tools.ops.time_helpers import count_video_frames


def make(**config):
    values = VideoTrimPipe.get_default_config()
    values.update(config)
    return VideoTrimPipe(values)


def run(pipe, cancelled=None, **inputs):
    emitted = []
    output = pipe.process(PipeInput(input=inputs), emitted.append, cancelled).output["video"]
    return output, emitted


def frames_of(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def audio_seconds(path):
    result = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    hours, minutes, seconds = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", result.stderr)[-1]
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


@pytest.fixture
def scratch_root(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe_tools.tempfile, "tempdir", str(tmp_path))
    return tmp_path


def leftovers(root):
    return sorted(path.name for path in root.iterdir() if path.name.startswith(("potionui-media-tool-", "potionui-trim-")))


class TestContract:
    def test_the_pipe_keeps_its_name_inputs_and_outputs(self):
        assert VideoTrimPipe.name == "video_trim"
        (video_in,) = VideoTrimPipe.inputs()
        (video_out,) = VideoTrimPipe.outputs()
        assert (video_in.name, video_in.io_type, video_in.required, video_in.is_array) == ("video", IOType.VIDEO, True, True)
        assert (video_out.name, video_out.io_type, video_out.is_array) == ("video", IOType.VIDEO, True)

    def test_the_configuration_keeps_its_names_defaults_and_bounds(self):
        specs = {spec.name: spec for spec in VideoTrimPipe.configuration()}
        assert {name: spec.default for name, spec in specs.items()} == {
            "keep": "tail", "frames": 25, "seconds": 0.0, "frame_step": 1, "fps": 0.0, "keep_audio": True,
        }
        assert specs["keep"].choices == ["head", "tail"]
        assert specs["frames"].min_value == 1 and specs["frame_step"].min_value == 1

    def test_a_keep_other_than_head_or_tail_is_rejected(self):
        with pytest.raises(ValueError, match="keep"):
            validate_pipe_configuration(VideoTrimPipe, {"keep": "middle"})

    def test_the_tool_spec_accepts_the_pipes_defaults(self):
        resolved = validate_params(HEAD_TAIL, VideoTrimPipe.get_default_config())
        assert resolved["keep"] == "tail" and resolved["keep_audio"] is True

    def test_no_video_is_an_error(self):
        with pytest.raises(ValueError, match="at least one video"):
            run(make(), video=None)


class TestFilters:
    def media(self, **fields):
        values = dict(path=Path("/m/a.mp4"), kind="video", width=64, height=48, duration=3.0, fps=10.0, has_audio=True)
        values.update(fields)
        return ToolMedia(**values)

    def fragment(self, **params):
        base = {"keep": "tail", "frames": 9, "seconds": 0.0, "frame_step": 1, "fps": 0.0, "keep_audio": True}
        base.update(params)
        return HEAD_TAIL.impl.video_filter(self.media(), base)

    def test_a_tail_selects_its_last_frames_by_reversing_twice(self):
        fragment = self.fragment(keep="tail")
        assert fragment.vf[:3] == ["reverse", "trim=end_frame=9", "reverse"]
        assert fragment.af[:3] == ["areverse", "atrim=end=0.900000", "areverse"]

    def test_a_head_selects_its_first_frames(self):
        fragment = self.fragment(keep="head")
        assert fragment.vf[0] == "trim=end_frame=9"
        assert fragment.af[0] == "atrim=end=0.900000"

    def test_the_picture_always_ends_with_zeroed_timestamps_and_a_pinned_rate(self):
        assert self.fragment().vf[-2:] == ["setpts=PTS-STARTPTS", "fps=10"]
        assert self.fragment(fps=24.0).vf[-2:] == ["setpts=PTS-STARTPTS", "fps=24"]

    def test_a_resample_comes_before_the_frames_are_counted(self):
        assert self.fragment(keep="head", fps=24.0).vf[:2] == ["fps=24", "trim=end_frame=9"]

    def test_the_clip_is_encoded_at_the_high_quality_it_always_was(self):
        assert self.fragment().crf == KEPT_CRF == 12

    def test_the_sound_is_dropped_when_asked(self):
        assert self.fragment(keep_audio=False).drop_audio is True
        assert self.fragment(keep_audio=True).drop_audio is False

    @pytest.mark.parametrize("seconds, fps, source_fps, step, expected", [
        (1.0, 24.0, 10.0, 8, 17),
        (1.0, 0.0, 30.0, 8, 25),
        (0.25, 8.0, 10.0, 8, 1),
        (0.5, 0.0, 24.0, 1, 12),
    ])
    def test_a_length_in_seconds_becomes_a_stepped_frame_count(self, seconds, fps, source_fps, step, expected):
        fragment = HEAD_TAIL.impl.video_filter(self.media(fps=source_fps), {
            "keep": "head", "frames": 25, "seconds": seconds, "frame_step": step, "fps": fps, "keep_audio": True,
        })
        assert f"trim=end_frame={expected}" in fragment.vf


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not available")
class TestFrames:
    @pytest.mark.parametrize("source_fps, seconds", [(10, 3.0), (24, 2.0), (30, 2.0)])
    @pytest.mark.parametrize("keep", ["head", "tail"])
    @pytest.mark.parametrize("target", [0.0, 24.0])
    @pytest.mark.parametrize("frames", [1, 9, 17])
    def test_exactly_the_requested_frames_come_out_at_the_expected_rate(
        self, tmp_path, scratch_root, source_fps, seconds, keep, target, frames,
    ):
        clip = make_video(tmp_path / "in.mp4", seconds=seconds, size=(64, 48), fps=source_fps)
        (result,), _ = run(make(keep=keep, frames=frames, fps=target), video=[str(clip)])

        assert count_video_frames(Path(result)) == frames
        assert probe(result).fps == pytest.approx(target or source_fps, abs=0.01)
        Path(result).unlink()

    @pytest.mark.parametrize("keep", ["head", "tail"])
    def test_the_frames_kept_are_the_first_or_last_ones_of_the_source(self, tmp_path, scratch_root, keep):
        clip = make_video(tmp_path / "in.mp4", seconds=3.0, size=(64, 48), fps=10)
        source = frames_of(clip)
        (result,), _ = run(make(keep=keep, frames=9), video=[str(clip)])

        kept = frames_of(result)
        expected = source[:9] if keep == "head" else source[-9:]
        assert len(kept) == 9
        for got, want in zip(kept, expected):
            assert np.abs(got.astype(int) - want.astype(int)).mean() < 6
        other = source[-9:] if keep == "head" else source[:9]
        assert np.abs(kept[0].astype(int) - other[0].astype(int)).mean() > 6
        Path(result).unlink()

    @pytest.mark.parametrize("seconds, fps, source_fps, step, expected", [
        (1.0, 24.0, 10, 8, 17),
        (1.0, 0.0, 30, 8, 25),
        (0.25, 8.0, 10, 8, 1),
        (0.5, 0.0, 24, 1, 12),
    ])
    def test_a_length_in_seconds_gives_the_stepped_frame_count(
        self, tmp_path, scratch_root, seconds, fps, source_fps, step, expected,
    ):
        clip = make_video(tmp_path / "in.mp4", seconds=3.0, size=(64, 48), fps=source_fps)
        (result,), _ = run(make(seconds=seconds, fps=fps, frame_step=step, keep="head"), video=[str(clip)])

        assert count_video_frames(Path(result)) == expected
        Path(result).unlink()

    def test_a_clip_shorter_than_the_request_is_kept_whole(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48), fps=10)
        (result,), _ = run(make(frames=50), video=[str(clip)])

        assert count_video_frames(Path(result)) == 10
        Path(result).unlink()

    def test_an_odd_sized_clip_comes_out_even(self, tmp_path, scratch_root):
        odd = tmp_path / "odd.mp4"
        subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
             "-i", "testsrc=size=63x47:rate=10:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv444p", str(odd)],
            check=True,
        )
        (result,), _ = run(make(frames=5), video=[str(odd)])

        assert (probe(result).width, probe(result).height) == (62, 46)
        Path(result).unlink()


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not available")
class TestSound:
    @pytest.mark.parametrize("fps", [0.0, 10.0])
    @pytest.mark.parametrize("keep", ["head", "tail"])
    def test_the_sound_of_the_kept_frames_stays_with_them_whatever_the_rate_setting(
        self, tmp_path, scratch_root, fps, keep,
    ):
        clip = make_video(tmp_path / "in.mp4", seconds=3.0, size=(64, 48), fps=10, audio=True)
        (result,), _ = run(make(keep=keep, frames=17, fps=fps), video=[str(clip)])

        assert probe(result).has_audio is True
        assert audio_seconds(result) == pytest.approx(1.7, abs=0.1)
        Path(result).unlink()

    def test_the_sound_is_dropped_when_asked(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0, size=(64, 48), fps=10, audio=True)
        (result,), _ = run(make(keep_audio=False), video=[str(clip)])

        assert probe(result).has_audio is False
        Path(result).unlink()

    def test_a_clip_without_sound_stays_silent(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0, size=(64, 48), fps=10, audio=False)
        (result,), _ = run(make(keep_audio=True), video=[str(clip)])

        assert probe(result).has_audio is False
        Path(result).unlink()


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is not available")
class TestRun:
    def test_every_video_gets_its_own_trimmed_file_that_outlives_the_run(self, tmp_path, scratch_root):
        clips = [make_video(tmp_path / f"{name}.mp4", seconds=2.0, size=(64, 48), fps=10) for name in "ab"]
        out, _ = run(make(frames=5), video=[str(clip) for clip in clips])

        assert len(out) == 2 and len(set(out)) == 2
        assert all(Path(path).is_file() and count_video_frames(Path(path)) == 5 for path in out)
        assert leftovers(scratch_root) == sorted(Path(path).name for path in out)
        for path in out:
            Path(path).unlink()

    def test_a_single_path_string_is_accepted(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0, size=(64, 48), fps=10)
        out, _ = run(make(frames=5), video=str(clip))

        assert len(out) == 1
        Path(out[0]).unlink()

    def test_progress_names_the_end_or_start_and_reaches_the_finish(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0, size=(64, 48), fps=10)
        out, emitted = run(make(keep="tail", frames=5), video=[str(clip)])
        out_head, emitted_head = run(make(keep="head", frames=5), video=[str(clip)])

        for items, label in ((emitted, "Taking the end of the video"), (emitted_head, "Taking the start of the video")):
            progress = [item for item in items if isinstance(item, ProgressGenerationOutput)]
            assert progress[0].state == label
            assert progress[-1].progress.current == 100
        for path in (*out, *out_head):
            Path(path).unlink()

    def test_cancelling_returns_nothing_and_leaves_no_file(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0, size=(64, 48), fps=10)
        out, _ = run(make(frames=5), cancelled=lambda: True, video=[str(clip)])

        assert out == []
        assert leftovers(scratch_root) == []

    def test_a_missing_file_is_a_clean_generation_error(self, tmp_path, scratch_root):
        with pytest.raises(GenerationExecutionError, match="missing"):
            run(make(), video=[str(tmp_path / "gone.mp4")])
        assert leftovers(scratch_root) == []

    def test_a_failure_on_a_later_video_removes_the_earlier_results(self, tmp_path, scratch_root):
        clip = make_video(tmp_path / "in.mp4", seconds=2.0, size=(64, 48), fps=10)
        with pytest.raises(GenerationExecutionError):
            run(make(frames=5), video=[str(clip), str(tmp_path / "gone.mp4")])
        assert leftovers(scratch_root) == []
