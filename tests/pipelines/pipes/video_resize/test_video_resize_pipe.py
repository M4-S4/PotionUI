import subprocess

import pytest

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.video_resize import main as pipe_module
from src.pipelines.pipes.video_resize.main import VideoResizePipe, fit_size, parse_size

FFMPEG = pipe_module.find_ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="needs a real ffmpeg")


def make(**config):
    values = VideoResizePipe.get_default_config()
    values.update(config)
    return VideoResizePipe(values)


def synth(path, size="320x180", fps=24, audio=True):
    command = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration=1"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", "-shortest"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return str(path)


def streams(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    probe = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return width, height, fps, "Audio:" in probe.stderr


def run(pipe, video):
    notes = []
    out = pipe.process(PipeInput(input={"video": [video]}), notes.append).output["video"]
    return out, [note.state for note in notes]


@pytest.mark.parametrize("source, requested, expected", [
    ((1920, 1080), (1280, 720), (1280, 720, False)),
    ((1920, 1080), (720, 720), (720, 404, False)),
    ((1080, 1920), (1280, 720), (404, 720, False)),
    ((641, 361), (320, 320), (320, 180, False)),
    ((640, 360), (1920, 1080), (640, 360, True)),
    ((641, 361), (4000, 4000), (640, 360, True)),
])
def test_the_size_fits_inside_the_box_with_even_sides_and_never_grows(source, requested, expected):
    assert fit_size(source, requested) == expected


@pytest.mark.parametrize("value", ["", "1280", "axb", "1x720", None])
def test_a_malformed_size_is_a_clear_error(value):
    with pytest.raises(ValueError):
        parse_size(value)


def test_no_video_is_an_error():
    with pytest.raises(ValueError):
        make(size="640x360").process(PipeInput(input={"video": None}), lambda output: None)


def test_an_unknown_quality_is_an_error(monkeypatch):
    monkeypatch.setattr(pipe_module, "probe_video_stream", lambda path: (640, 360, 24.0, 24))
    with pytest.raises(ValueError):
        make(size="320x180", quality="tiny").process(PipeInput(input={"video": ["a.mp4"]}), lambda output: None)


def test_quality_picks_the_crf_and_audio_falls_back_to_aac_when_copy_fails(monkeypatch, tmp_path):
    commands = []

    class Done:
        def __init__(self, code):
            self.returncode, self.stderr = code, b"boom"

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[command.index("-c:a") + 1] == "copy":
            return Done(1)
        open(command[-1], "wb").write(b"x")
        return Done(0)

    monkeypatch.setattr(pipe_module, "probe_video_stream", lambda path: (640, 360, 24.0, 24))
    monkeypatch.setattr(pipe_module, "run_cancellable", fake_run)
    monkeypatch.setattr(pipe_module, "find_ffmpeg", lambda: "ffmpeg")

    out, _ = run(make(size="320x180", quality="small"), "a.mp4")

    assert len(out) == 1 and len(commands) == 2
    assert commands[0][commands[0].index("-crf") + 1] == "28"
    assert commands[1][commands[1].index("-c:a") + 1] == "aac"
    assert "scale=320:180:flags=lanczos,setsar=1" in commands[1]


@needs_ffmpeg
def test_a_real_video_is_downscaled_keeping_fps_and_sound(tmp_path):
    video = synth(tmp_path / "in.mp4", size="320x180", fps=24, audio=True)

    out, _ = run(make(size="160x90", quality="high"), video)

    width, height, fps, has_audio = streams(out[0])
    assert (width, height) == (160, 90)
    assert fps == pytest.approx(24.0, abs=0.01)
    assert has_audio


@needs_ffmpeg
def test_a_silent_video_stays_silent_and_a_mismatched_box_keeps_the_aspect(tmp_path):
    video = synth(tmp_path / "in.mp4", size="320x180", fps=30, audio=False)

    out, _ = run(make(size="100x100"), video)

    width, height, fps, has_audio = streams(out[0])
    assert (width, height) == (100, 56)
    assert width % 2 == 0 and height % 2 == 0
    assert fps == pytest.approx(30.0, abs=0.01)
    assert not has_audio


@needs_ffmpeg
def test_a_box_larger_than_the_source_keeps_the_source_size_and_says_so(tmp_path):
    video = synth(tmp_path / "in.mp4", size="320x180", audio=False)

    out, notes = run(make(size="1920x1080"), video)

    assert streams(out[0])[:2] == (320, 180)
    assert any("keeps its size" in note for note in notes)
