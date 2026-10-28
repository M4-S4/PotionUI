import pytest

from src.pipelines.contracts import PipeInput
from src.pipelines.pipes._shared.media.video_concat import ClipTrim, VideoConcatCancelled
from src.pipelines.pipes.video_concat import main as pipe_module
from src.pipelines.pipes.video_concat.main import VideoConcatPipe


def make(**config):
    values = VideoConcatPipe.get_default_config()
    values.update(config)
    return VideoConcatPipe(values)


@pytest.fixture
def joins(monkeypatch):
    calls = []

    def fake(clips, out_path, **kwargs):
        calls.append({"clips": list(clips), **kwargs})
        return out_path

    monkeypatch.setattr(pipe_module, "concat_videos", fake)
    return calls


def run(pipe, **inputs):
    return pipe.process(PipeInput(input=inputs), lambda output: None).output["video"]


def test_without_extensions_every_video_is_joined_once_in_order(joins):
    out = run(make(), video=["a.mp4", "b.mp4", "c.mp4"])

    assert len(out) == 1 and [call["clips"] for call in joins] == [["a.mp4", "b.mp4", "c.mp4"]]
    assert joins[0]["trims"] is None and joins[0]["crossfades"] is None and joins[0]["audio"] is True


def test_each_extension_makes_its_own_joined_video_after_the_source(joins):
    out = run(make(overlap_frames=12, fps=24), video="src.mp4", extension=["e1.mp4", "e2.mp4"])

    assert len(out) == 2
    assert [call["clips"] for call in joins] == [["src.mp4", "e1.mp4"], ["src.mp4", "e2.mp4"]]
    assert joins[0]["crossfades"] == [0.5] and joins[0]["trims"] == [ClipTrim(), ClipTrim()]


def test_an_extension_before_the_source_cuts_its_repeated_tail(joins):
    run(make(extension_position="before", overlap_frames=6, fps=24, overlap="cut", keep_audio="false"),
        video=["src.mp4"], extension=["e.mp4"])

    (call,) = joins
    assert call["clips"] == ["e.mp4", "src.mp4"]
    assert call["trims"] == [ClipTrim(end=0.25), ClipTrim()] and call["crossfades"] == [0.0]
    assert call["audio"] is False


def test_an_extension_after_the_source_cuts_its_repeated_head(joins):
    run(make(overlap_frames=6, fps=24, overlap="cut"), video=["src.mp4"], extension=["e.mp4"])

    assert joins[0]["trims"] == [ClipTrim(), ClipTrim(start=0.25)]


def test_no_video_is_an_error(joins):
    with pytest.raises(ValueError):
        run(make(), video=[])


def test_a_cancel_stops_without_a_joined_video(monkeypatch, tmp_path):
    def cancelled(clips, out_path, **kwargs):
        raise VideoConcatCancelled()

    monkeypatch.setattr(pipe_module, "concat_videos", cancelled)

    assert run(make(), video=["src.mp4"], extension=["e1.mp4", "e2.mp4"]) == []
