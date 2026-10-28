from types import SimpleNamespace

import sys
import time

import pytest

from src.pipelines.pipes._shared.media import video_concat as concat_module
from src.pipelines.pipes._shared.media.video_concat import (
    NO_SOUND_NOTICE,
    ClipInfo,
    ClipTrim,
    VideoConcatCancelled,
    VideoConcatError,
    concat_command,
    probe_clip,
    run_cancellable,
    concat_videos,
    trim_command,
    trim_video,
)
from src.platform.util.video_transcode import VideoProbe

FFMPEG = "/opt/ffmpeg/bin/ffmpeg"


def finished(returncode=0, stdout=b"", stderr=b""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def ffmpeg_found():
    return FFMPEG


def no_ffmpeg():
    return None


def probed(**fields):
    values = dict(duration=4.0, width=1280, height=720, fps=24.0, video_codec="h264", has_audio=True, size_bytes=10)
    values.update(fields)
    return VideoProbe(**values)


def test_the_join_scales_every_shot_to_the_first_and_fills_missing_sound_with_silence(tmp_path):
    infos = [
        ClipInfo(width=1281, height=721, fps=24.0, duration=4.0, has_audio=True),
        ClipInfo(width=640, height=360, fps=30.0, duration=6.0, has_audio=False),
    ]

    command = concat_command(FFMPEG, ["a.mp4", "b.mp4"], infos, tmp_path / "film.mp4")

    graph = command[command.index("-filter_complex") + 1]
    assert command[:7] == [FFMPEG, "-y", "-loglevel", "error", "-i", "a.mp4", "-i"]
    assert ["-f", "lavfi", "-t", "6.000", "-i", "anullsrc=r=48000:cl=stereo"] == command[8:14]
    assert "[0:v]scale=1280:720" in graph and "[1:v]scale=1280:720" in graph and "fps=24" in graph
    assert "[0:a]aresample=48000" in graph and "[2:a]aresample=48000" in graph
    assert graph.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    assert command[-1] == str(tmp_path / "film.mp4") and "-map" in command and "[a]" in command


def test_a_film_without_any_sound_is_joined_without_an_audio_track(tmp_path):
    infos = [ClipInfo(640, 360, 24.0, 2.0, False), ClipInfo(640, 360, 24.0, 2.0, False)]

    command = concat_command(FFMPEG, ["a.mp4", "b.mp4"], infos, tmp_path / "film.mp4")

    graph = command[command.index("-filter_complex") + 1]
    assert "anullsrc" not in " ".join(command) and graph.endswith("[v0][v1]concat=n=2:v=1:a=0[v]")
    assert "[a]" not in command and "-c:a" not in command


def test_ffmpeg_joins_the_shots_when_it_is_available(tmp_path):
    calls = []
    out = tmp_path / "film.mp4"

    def run(command, **kwargs):
        calls.append(command)
        out.write_bytes(b"film")
        return finished()

    info = ClipInfo(64, 48, 24.0, 2.0, True)
    concat_videos(["a.mp4", "b.mp4"], out, run=run, find=ffmpeg_found, probe=lambda clip: info)

    (command,) = calls
    assert command[0] == FFMPEG and out.read_bytes() == b"film"


def test_a_shot_that_cannot_be_probed_skips_ffmpeg(tmp_path):
    pytest.importorskip("cv2", reason="the fallback join reads clips with cv2", exc_type=ImportError)
    from src.features.cloud.testing.fake import fake_video_bytes

    clips = []
    for index in range(2):
        clip = tmp_path / f"{index}.mp4"
        clip.write_bytes(fake_video_bytes(1, start=(0, 0, 0), end=(255, 255, 255)))
        clips.append(clip)

    def run(command, **kwargs):
        raise AssertionError("ffmpeg must not run")

    out = concat_videos(clips, tmp_path / "film.mp4", run=run, find=ffmpeg_found, probe=lambda clip: None)

    assert out.stat().st_size > 0


def test_without_ffmpeg_the_shots_are_joined_frame_by_frame_at_the_first_shots_size(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="the fallback join reads clips with cv2", exc_type=ImportError)
    from src.features.cloud.testing.fake import fake_video_bytes

    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(fake_video_bytes(1, start=(255, 0, 0), end=(255, 0, 0), size=(32, 24)))
    second.write_bytes(fake_video_bytes(2, start=(0, 0, 255), end=(0, 0, 255), size=(48, 36)))

    out = concat_videos([first, second], tmp_path / "film.mp4", find=no_ffmpeg)

    capture = cv2.VideoCapture(str(out))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    assert (count, size) == (8 + 16, (32, 24))


def test_a_failing_ffmpeg_falls_back_to_the_frame_join(tmp_path):
    pytest.importorskip("cv2", reason="the fallback join reads clips with cv2", exc_type=ImportError)
    from src.features.cloud.testing.fake import fake_video_bytes

    clip = tmp_path / "only.mp4"
    clip.write_bytes(fake_video_bytes(1, start=(0, 0, 0), end=(9, 9, 9)))

    notices = []
    out = concat_videos(
        [clip, clip], tmp_path / "film.mp4",
        run=lambda command, **kwargs: finished(returncode=1, stderr=b"boom"),
        find=ffmpeg_found, probe=lambda path: ClipInfo(32, 24, 8.0, 1.0, False),
        notify=notices.append,
    )

    assert out.stat().st_size > 0
    assert notices == [NO_SOUND_NOTICE]


def test_a_cancel_during_the_ffmpeg_join_stops_without_falling_back(tmp_path):
    def run(command, **kwargs):
        assert kwargs["is_cancelled"]() is True
        raise VideoConcatCancelled()

    with pytest.raises(VideoConcatCancelled):
        concat_videos(
            ["a.mp4", "b.mp4"], tmp_path / "film.mp4", run=run, find=ffmpeg_found,
            probe=lambda clip: ClipInfo(64, 48, 24.0, 2.0, True), is_cancelled=lambda: True,
        )


def test_a_running_process_is_terminated_when_the_cancel_flag_rises():
    polls = []

    def is_cancelled():
        polls.append(1)
        return len(polls) >= 2

    started = time.monotonic()
    with pytest.raises(VideoConcatCancelled):
        run_cancellable([sys.executable, "-c", "import time; time.sleep(120)"], timeout=60, is_cancelled=is_cancelled)

    assert time.monotonic() - started < 30


def test_a_process_that_outlives_the_timeout_is_killed_and_reported():
    with pytest.raises(concat_module.subprocess.TimeoutExpired):
        run_cancellable([sys.executable, "-c", "import time; time.sleep(120)"], timeout=0.5)


def test_a_finished_process_reports_its_status_and_the_end_of_its_complaint():
    code = "import sys; sys.stderr.write('x' * 1000 + 'END'); sys.exit(3)"

    result = run_cancellable([sys.executable, "-c", code], timeout=60)

    assert result.returncode == 3 and result.stderr.endswith(b"END") and len(result.stderr) <= 400


def test_nothing_to_join_is_an_error(tmp_path):
    with pytest.raises(VideoConcatError):
        concat_videos([], tmp_path / "film.mp4")


def test_the_shared_probe_becomes_the_clip_size_rate_length_and_sound():
    info = probe_clip("x.mp4", probe=lambda path: probed(fps=29.97, duration=5.005))

    assert (info.width, info.height, info.fps, info.duration, info.has_audio) == (1280, 720, 29.97, 5.005, True)


def test_a_clip_without_a_usable_probe_is_not_joined_by_ffmpeg():
    assert probe_clip("x.mp4", probe=lambda path: None) is None
    assert probe_clip("x.mp4", probe=lambda path: probed(width=0)) is None
    assert probe_clip("x.mp4", probe=lambda path: probed(duration=0.0)) is None
    assert probe_clip("x.mp4", probe=lambda path: probed(fps=0.0)).fps == 24.0


def graph_of(command):
    return command[command.index("-filter_complex") + 1]


def test_trims_cut_the_start_and_end_of_a_clip_in_picture_and_sound(tmp_path):
    infos = [ClipInfo(640, 360, 24.0, 4.0, True), ClipInfo(640, 360, 24.0, 2.0, True)]

    graph = graph_of(concat_command(
        FFMPEG, ["a.mp4", "b.mp4"], infos, tmp_path / "out.mp4",
        trims=[ClipTrim(end=0.5), ClipTrim(start=0.375)],
    ))

    assert "[0:v]trim=end=3.499000,setpts=PTS-STARTPTS,scale=640:360" in graph
    assert "[1:v]trim=start=0.374000,setpts=PTS-STARTPTS,scale=640:360" in graph
    assert "[0:a]atrim=end=3.499000,asetpts=PTS-STARTPTS,aresample" in graph
    assert "[1:a]atrim=start=0.374000,asetpts=PTS-STARTPTS,aresample" in graph
    assert graph.endswith("concat=n=2:v=1:a=1[v][a]")


def test_a_crossfade_blends_the_seam_at_the_end_of_what_came_before(tmp_path):
    infos = [ClipInfo(640, 360, 24.0, 4.0, True), ClipInfo(640, 360, 24.0, 2.0, False), ClipInfo(640, 360, 24.0, 3.0, True)]

    graph = graph_of(concat_command(
        FFMPEG, ["a.mp4", "b.mp4", "c.mp4"], infos, tmp_path / "out.mp4",
        trims=[ClipTrim(), ClipTrim(start=0.5), ClipTrim()], crossfades=[0.25, 0.0],
    ))

    assert "settb=AVTB[v0]" in graph
    assert "[v0][v1]xfade=transition=fade:duration=0.250000:offset=3.750000[vj1]" in graph
    assert "[a0][a1]acrossfade=d=0.250000[aj1]" in graph
    assert graph.endswith("[vj1][v2]concat=n=2:v=1:a=0[v];[aj1][a2]concat=n=2:v=0:a=1[a]")


def test_a_join_without_sound_drops_every_audio_track(tmp_path):
    infos = [ClipInfo(640, 360, 24.0, 4.0, True), ClipInfo(640, 360, 24.0, 2.0, True)]

    command = concat_command(FFMPEG, ["a.mp4", "b.mp4"], infos, tmp_path / "out.mp4", audio=False)

    assert "[a]" not in command and "atrim" not in graph_of(command) and "aresample" not in graph_of(command)


@pytest.mark.parametrize("trims, crossfades", [
    ([ClipTrim(start=3.0), ClipTrim(end=1.0)], [0.0]),
    ([ClipTrim(), ClipTrim()], [2.0]),
    ([ClipTrim(start=-1.0), ClipTrim()], [0.0]),
    ([ClipTrim()], [0.0]),
    ([ClipTrim(), ClipTrim()], []),
])
def test_a_layout_that_leaves_nothing_to_show_is_refused(tmp_path, trims, crossfades):
    infos = [ClipInfo(640, 360, 24.0, 3.0, False), ClipInfo(640, 360, 24.0, 2.0, False)]

    with pytest.raises(VideoConcatError):
        concat_command(FFMPEG, ["a.mp4", "b.mp4"], infos, tmp_path / "out.mp4", trims=trims, crossfades=crossfades)


def test_the_tail_trim_seeks_near_the_end_and_counts_frames_backwards(tmp_path):
    command = trim_command(FFMPEG, "in.mp4", tmp_path / "tail.mp4", keep="tail", frames=17, fps=25)

    assert command[command.index("-sseof") + 1] == "-1.760"
    assert command[command.index("-vf") + 1].startswith("fps=25,reverse,trim=end_frame=17,reverse,setpts=PTS-STARTPTS")
    assert command[command.index("-af") + 1] == "areverse,atrim=end=0.680000,areverse,asetpts=PTS-STARTPTS"
    assert command[command.index("-frames:v") + 1] == "17" and "0:a:0?" in command


def test_the_head_trim_keeps_the_first_frames_at_the_source_rate_without_sound(tmp_path):
    command = trim_command(FFMPEG, "in.mp4", tmp_path / "head.mp4", keep="head", frames=9, audio=False)

    assert "-sseof" not in command and "-an" in command and "-af" not in command
    assert command[command.index("-vf") + 1].startswith("trim=end_frame=9,setpts=PTS-STARTPTS")


@pytest.mark.parametrize("keep, frames", [("middle", 5), ("tail", 0)])
def test_a_trim_that_cannot_be_made_is_refused(tmp_path, keep, frames):
    with pytest.raises(VideoConcatError):
        trim_command(FFMPEG, "in.mp4", tmp_path / "out.mp4", keep=keep, frames=frames)


def test_trimming_without_ffmpeg_is_an_error(tmp_path):
    with pytest.raises(VideoConcatError):
        trim_video("in.mp4", tmp_path / "out.mp4", keep="tail", frames=5, find=no_ffmpeg)


def test_a_failed_trim_reports_ffmpegs_complaint(tmp_path):
    with pytest.raises(VideoConcatError, match="boom"):
        trim_video(
            "in.mp4", tmp_path / "out.mp4", keep="tail", frames=5, find=ffmpeg_found,
            run=lambda command, **kwargs: finished(returncode=1, stderr=b"boom"),
        )


def test_the_frame_join_fallback_applies_trims_and_cuts_crossfades(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="the fallback join reads clips with cv2", exc_type=ImportError)
    from src.features.cloud.testing.fake import fake_video_bytes

    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(fake_video_bytes(1, start=(255, 0, 0), end=(255, 0, 0), size=(32, 24)))
    second.write_bytes(fake_video_bytes(1, start=(0, 0, 255), end=(0, 0, 255), size=(32, 24)))

    out = concat_videos(
        [first, second], tmp_path / "film.mp4", find=no_ffmpeg,
        trims=[ClipTrim(end=2 / 8), ClipTrim(start=1 / 8)], crossfades=[3 / 8],
    )

    capture = cv2.VideoCapture(str(out))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert count == (8 - 2) + (8 - 1 - 3)


@pytest.mark.skipif(concat_module.find_ffmpeg() is None, reason="needs a real ffmpeg")
def test_a_real_ffmpeg_trims_and_crossfades_to_the_expected_frame_count(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="frames are counted with cv2", exc_type=ImportError)
    from src.features.cloud.testing.fake import fake_video_bytes

    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(fake_video_bytes(2, start=(255, 0, 0), end=(255, 0, 0), size=(32, 24)))
    second.write_bytes(fake_video_bytes(2, start=(0, 0, 255), end=(0, 0, 255), size=(32, 24)))
    infos = {str(first): ClipInfo(32, 24, 8.0, 2.0, False), str(second): ClipInfo(32, 24, 8.0, 2.0, False)}
    tail = trim_video(second, tmp_path / "tail.mp4", keep="tail", frames=5, fps=8)
    joined = concat_videos(
        [first, second], tmp_path / "film.mp4", probe=lambda clip: infos[str(clip)],
        trims=[ClipTrim(), ClipTrim(start=2 / 8)], crossfades=[3 / 8],
    )

    counts = []
    for path in (tail, joined):
        capture = cv2.VideoCapture(str(path))
        counts.append(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
    assert counts == [5, 16 + 14 - 3]
