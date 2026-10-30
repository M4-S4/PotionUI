from __future__ import annotations

import pytest
from PIL import Image

from src.platform.media_tools import InvalidToolParams
from tests.platform.media_tools.media_factory import make_image, make_video
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


@pytest.mark.parametrize(
    "degrees, size, position",
    [(90, (2, 4), (1, 0)), (180, (4, 2), (3, 1)), (270, (2, 4), (0, 3))],
)
def test_image_rotation_is_clockwise(degrees, size, position):
    out = tool_of("rotate").image(marker_image((4, 2), (0, 0)), resolved("rotate", degrees=degrees))
    assert out.size == size
    assert out.getpixel(position) == (255, 0, 0)
    assert sum(1 for pixel in out.getdata() if pixel == (255, 0, 0)) == 1


def test_default_rotation_is_a_quarter_turn():
    out = tool_of("rotate").image(marker_image((4, 2), (0, 0)), resolved("rotate"))
    assert out.size == (2, 4)


def test_degrees_accepts_a_string_and_rejects_others():
    assert resolved("rotate", degrees="270")["degrees"] == 270
    with pytest.raises(InvalidToolParams) as caught:
        resolved("rotate", degrees=45)
    assert caught.value.field == "degrees"


def test_rotation_keeps_alpha_of_a_png(tmp_path, ctx):
    path = tmp_path / "rgba.png"
    make_image(path, (20, 10), "RGBA", (10, 20, 30, 128))
    out = run_one("rotate", {"degrees": 90}, path, ctx)
    assert (out.width, out.height) == (10, 20)
    assert out.has_alpha is True


def test_video_rotation_swaps_the_frame_and_moves_the_marker_clockwise(need_ffmpeg, tmp_path, ctx):
    source = make_marker_video(tmp_path / "in.mp4", size=(64, 48), box=(16, 12))
    out = run_one("rotate", {"degrees": 90}, source, ctx)
    assert (out.width, out.height) == (48, 64)
    assert out.has_audio is True
    assert out.duration == pytest.approx(1.0, abs=0.15)
    frame = first_frame(out.path, tmp_path / "frame.png")
    assert is_red(frame.getpixel((43, 4)))
    assert is_blue(frame.getpixel((4, 4)))


def test_video_rotation_by_270_moves_the_marker_counter_clockwise(need_ffmpeg, tmp_path, ctx):
    source = make_marker_video(tmp_path / "in.mp4", size=(64, 48), box=(16, 12))
    out = run_one("rotate", {"degrees": 270}, source, ctx)
    assert (out.width, out.height) == (48, 64)
    frame = first_frame(out.path, tmp_path / "frame.png")
    assert is_red(frame.getpixel((4, 59)))
    assert is_blue(frame.getpixel((4, 4)))


def test_video_rotation_by_180_keeps_the_frame_size(need_ffmpeg, tmp_path, ctx):
    source = make_marker_video(tmp_path / "in.mp4", size=(64, 48), box=(16, 12))
    out = run_one("rotate", {"degrees": 180}, source, ctx)
    assert (out.width, out.height) == (64, 48)
    frame = first_frame(out.path, tmp_path / "frame.png")
    assert is_red(frame.getpixel((59, 43)))
    assert is_blue(frame.getpixel((4, 4)))


def test_video_filter_is_declared_for_video_only(tmp_path):
    from src.platform.media_tools import probe

    path = make_image(tmp_path / "a.png")
    assert tool_of("rotate").video_filter(probe(path), {"degrees": 90}) is None


def test_video_rotation_updates_the_virtual_size(need_ffmpeg, tmp_path):
    from src.platform.media_tools import probe

    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    fragment = tool_of("rotate").video_filter(probe(path), {"degrees": 90})
    assert (fragment.width, fragment.height) == (48, 64)
    assert fragment.vf == ["transpose=1"]
