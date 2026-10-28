import pytest

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes.video_trim import main as pipe_module
from src.pipelines.pipes.video_trim.main import VideoTrimPipe


def make(**config):
    values = VideoTrimPipe.get_default_config()
    values.update(config)
    return VideoTrimPipe(values)


def test_every_video_is_trimmed_with_the_configured_window(monkeypatch):
    calls = []

    def fake(clip, out_path, **kwargs):
        calls.append((clip, kwargs))
        return out_path

    monkeypatch.setattr(pipe_module, "trim_video", fake)

    out = make(keep="head", frames="17", fps="25", keep_audio="false").process(
        PipeInput(input={"video": ["a.mp4", "b.mp4"]}), lambda output: None,
    ).output["video"]

    assert len(out) == 2 and [clip for clip, _ in calls] == ["a.mp4", "b.mp4"]
    assert {key: calls[0][1][key] for key in ("keep", "frames", "fps", "audio")} == {
        "keep": "head", "frames": 17, "fps": 25.0, "audio": False,
    }


def test_no_video_is_an_error():
    with pytest.raises(ValueError):
        make().process(PipeInput(input={"video": None}), lambda output: None)
