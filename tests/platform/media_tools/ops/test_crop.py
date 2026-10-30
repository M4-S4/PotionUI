from __future__ import annotations

import pytest
from PIL import Image

from src.platform.media_tools import InvalidToolParams, probe
from tests.platform.media_tools.media_factory import make_image, make_video
from tests.platform.media_tools.ops.geometry_support import (
    resolved,
    run_one,
    spec_of,
    tool_of,
)


def numbered_image(size=(8, 6)) -> Image.Image:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), (x * 10, y * 10, 7))
    return image


def test_image_crop_returns_exact_pixels():
    source = numbered_image()
    out = tool_of("crop").image(source, resolved("crop", x=2, y=1, width=4, height=3))
    assert out.size == (4, 3)
    assert out.getpixel((0, 0)) == source.getpixel((2, 1))
    assert out.getpixel((3, 2)) == source.getpixel((5, 3))


def test_image_crop_defaults_origin_to_zero():
    out = tool_of("crop").image(numbered_image(), resolved("crop", width=3, height=2))
    assert out.size == (3, 2)
    assert out.getpixel((0, 0)) == (0, 0, 7)


@pytest.mark.parametrize(
    "params, field",
    [
        ({"x": 6, "y": 0, "width": 4, "height": 2}, "width"),
        ({"x": 0, "y": 5, "width": 2, "height": 4}, "height"),
        ({"x": 8, "y": 0, "width": 1, "height": 1}, "x"),
        ({"x": 0, "y": 6, "width": 1, "height": 1}, "y"),
    ],
)
def test_image_crop_outside_the_image_names_the_field(params, field):
    with pytest.raises(InvalidToolParams) as caught:
        tool_of("crop").image(numbered_image(), resolved("crop", **params))
    assert caught.value.field == field


def test_crop_rejects_non_positive_size_at_validation():
    with pytest.raises(InvalidToolParams) as caught:
        resolved("crop", width=0, height=5)
    assert caught.value.field == "width"


def test_crop_requires_a_size():
    with pytest.raises(InvalidToolParams) as caught:
        resolved("crop", width=5)
    assert caught.value.field == "height"


def test_crop_keeps_alpha_of_a_png(tmp_path, ctx):
    path = tmp_path / "rgba.png"
    make_image(path, (20, 10), "RGBA", (10, 20, 30, 128))
    out = run_one("crop", {"x": 2, "y": 2, "width": 10, "height": 5}, path, ctx)
    assert (out.width, out.height) == (10, 5)
    assert out.has_alpha is True
    with Image.open(out.path) as saved:
        assert saved.mode == "RGBA"
        assert saved.getpixel((0, 0)) == (10, 20, 30, 128)


def test_crop_of_a_jpeg_stays_a_jpeg(tmp_path, ctx):
    path = tmp_path / "photo.jpg"
    make_image(path, (20, 10))
    out = run_one("crop", {"width": 10, "height": 10}, path, ctx)
    assert out.path.suffix == ".jpg"
    assert (out.width, out.height) == (10, 10)


def test_video_crop_changes_size_and_keeps_length_and_sound(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
    out = run_one("crop", {"x": 8, "y": 6, "width": 32, "height": 24}, path, ctx)
    assert (out.width, out.height) == (32, 24)
    assert out.has_audio is True
    assert out.duration == pytest.approx(1.0, abs=0.15)
    assert len(ctx.commands) == 1


def test_video_crop_is_even_aligned(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    out = run_one("crop", {"x": 1, "y": 1, "width": 21, "height": 15}, path, ctx)
    assert (out.width, out.height) == (20, 14)
    command = " ".join(ctx.commands[0])
    assert "crop=20:14:0:0" in command


def test_video_crop_outside_the_frame_fails_before_encoding(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    with pytest.raises(InvalidToolParams) as caught:
        run_one("crop", {"x": 40, "width": 40, "height": 10}, path, ctx)
    assert caught.value.field == "width"
    assert ctx.commands == []


def test_video_crop_too_small_after_even_alignment(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    with pytest.raises(InvalidToolParams) as caught:
        run_one("crop", {"width": 1, "height": 10}, path, ctx)
    assert caught.value.field == "width"


def test_video_filter_ignores_images(tmp_path):
    path = make_image(tmp_path / "a.png")
    assert tool_of("crop").video_filter(probe(path), {"x": 0, "y": 0, "width": 4, "height": 4}) is None


def test_spec_accepts_images_and_videos():
    spec = spec_of("crop")
    assert spec.accepts == ("image", "video")
    assert spec.editor == "crop"
