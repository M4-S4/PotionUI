from __future__ import annotations

import pytest

from src.platform.media_tools import InvalidToolParams
from tests.platform.media_tools.media_factory import make_image
from tests.platform.media_tools.ops.geometry_support import (
    first_frame,
    is_blue,
    is_red,
    make_marker_video,
    marker_image,
    resolved,
    run_one,
    tool_of,
)


def test_image_horizontal_flip_mirrors_left_to_right():
    out = tool_of("flip").image(marker_image((4, 2), (0, 0)), resolved("flip", axis="horizontal"))
    assert out.size == (4, 2)
    assert out.getpixel((3, 0)) == (255, 0, 0)
    assert out.getpixel((0, 0)) == (0, 0, 0)


def test_image_vertical_flip_mirrors_top_to_bottom():
    out = tool_of("flip").image(marker_image((4, 2), (0, 0)), resolved("flip", axis="vertical"))
    assert out.getpixel((0, 1)) == (255, 0, 0)
    assert out.getpixel((0, 0)) == (0, 0, 0)


def test_default_axis_is_horizontal():
    out = tool_of("flip").image(marker_image((4, 2), (0, 0)), resolved("flip"))
    assert out.getpixel((3, 0)) == (255, 0, 0)


def test_unknown_axis_is_refused():
    with pytest.raises(InvalidToolParams) as caught:
        resolved("flip", axis="diagonal")
    assert caught.value.field == "axis"


def test_flip_keeps_alpha_of_a_png(tmp_path, ctx):
    path = tmp_path / "rgba.png"
    make_image(path, (20, 10), "RGBA", (10, 20, 30, 128))
    out = run_one("flip", {"axis": "vertical"}, path, ctx)
    assert (out.width, out.height) == (20, 10)
    assert out.has_alpha is True


def test_video_horizontal_flip_moves_the_marker_right(need_ffmpeg, tmp_path, ctx):
    source = make_marker_video(tmp_path / "in.mp4", size=(64, 48), box=(16, 12))
    out = run_one("flip", {"axis": "horizontal"}, source, ctx)
    assert (out.width, out.height) == (64, 48)
    assert out.has_audio is True
    assert out.duration == pytest.approx(1.0, abs=0.15)
    frame = first_frame(out.path, tmp_path / "frame.png")
    assert is_red(frame.getpixel((59, 4)))
    assert is_blue(frame.getpixel((4, 4)))
    assert "hflip" in " ".join(ctx.commands[0])


def test_video_vertical_flip_moves_the_marker_down(need_ffmpeg, tmp_path, ctx):
    source = make_marker_video(tmp_path / "in.mp4", size=(64, 48), box=(16, 12))
    out = run_one("flip", {"axis": "vertical"}, source, ctx)
    frame = first_frame(out.path, tmp_path / "frame.png")
    assert is_red(frame.getpixel((4, 43)))
    assert is_blue(frame.getpixel((4, 4)))
    assert "vflip" in " ".join(ctx.commands[0])
