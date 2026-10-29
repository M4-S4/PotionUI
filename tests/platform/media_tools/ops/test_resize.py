from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.platform.media_tools import InvalidToolParams, ToolMedia
from tests.platform.media_tools.media_factory import make_image, make_video
from tests.platform.media_tools.ops.geometry_support import resolved, run_one, spec_of, tool_of


def test_resize_needs_a_width_or_a_height():
    with pytest.raises(InvalidToolParams) as caught:
        resolved("resize")
    assert caught.value.field == "width"


def test_image_resize_with_width_only_keeps_aspect():
    out = tool_of("resize").image(Image.new("RGB", (40, 30)), resolved("resize", width=20))
    assert out.size == (20, 15)


def test_image_resize_with_height_only_keeps_aspect():
    out = tool_of("resize").image(Image.new("RGB", (40, 30)), resolved("resize", height=60))
    assert out.size == (80, 60)


def test_image_resize_with_both_sides_stretches():
    out = tool_of("resize").image(Image.new("RGB", (40, 30)), resolved("resize", width=10, height=10))
    assert out.size == (10, 10)


def test_image_resize_over_the_limit_names_the_field():
    with pytest.raises(InvalidToolParams) as caught:
        tool_of("resize").image(Image.new("RGB", (40, 30)), resolved("resize", width=9000))
    assert caught.value.field == "width"
    with pytest.raises(InvalidToolParams) as caught:
        tool_of("resize").image(Image.new("RGB", (40, 30)), resolved("resize", width=100, height=9000))
    assert caught.value.field == "height"


def test_image_resize_derived_side_over_the_limit_is_refused():
    with pytest.raises(InvalidToolParams) as caught:
        tool_of("resize").image(Image.new("RGB", (100, 10)), resolved("resize", height=1000))
    assert caught.value.field == "width"


def test_palette_image_with_transparency_keeps_it():
    source = Image.new("P", (10, 10), 0)
    source.putpalette([255, 0, 0] * 256)
    source.info["transparency"] = 0
    out = tool_of("resize").image(source, resolved("resize", width=20))
    assert out.size == (20, 20)
    assert out.mode == "RGBA"
    assert out.getpixel((5, 5))[3] == 0


def test_resize_keeps_alpha_of_a_png(tmp_path, ctx):
    path = tmp_path / "rgba.png"
    make_image(path, (20, 10), "RGBA", (10, 20, 30, 128))
    out = run_one("resize", {"width": 10}, path, ctx)
    assert (out.width, out.height) == (10, 5)
    assert out.has_alpha is True


def test_resize_quality_is_validated():
    with pytest.raises(InvalidToolParams) as caught:
        resolved("resize", width=10, quality="ultra")
    assert caught.value.field == "quality"


def test_video_resize_with_width_only_keeps_aspect_and_sound(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
    out = run_one("resize", {"width": 32}, path, ctx)
    assert (out.width, out.height) == (32, 24)
    assert out.has_audio is True
    assert out.duration == pytest.approx(1.0, abs=0.15)


def test_video_resize_forces_even_dimensions(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
    out = run_one("resize", {"width": 31}, path, ctx)
    assert out.width % 2 == 0 and out.height % 2 == 0
    assert (out.width, out.height) == (30, 22)


def test_video_resize_over_the_limit_fails_before_encoding(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    with pytest.raises(InvalidToolParams) as caught:
        run_one("resize", {"width": 5000}, path, ctx)
    assert caught.value.field == "width"
    assert ctx.commands == []


def test_video_resize_too_small_after_even_alignment(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    with pytest.raises(InvalidToolParams) as caught:
        run_one("resize", {"width": 1}, path, ctx)
    assert caught.value.field == "width"


@pytest.mark.parametrize("quality, crf", [("high", "18"), ("balanced", "23"), ("small", "28")])
def test_video_resize_quality_maps_to_crf(need_ffmpeg, tmp_path, ctx, quality, crf):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    run_one("resize", {"width": 32, "quality": quality}, path, ctx)
    command = ctx.commands[0]
    assert command[command.index("-crf") + 1] == crf


def test_video_resize_without_an_encoder_choice_uses_the_balanced_crf(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    run_one("resize", {"height": 24}, path, ctx)
    command = ctx.commands[0]
    assert command[command.index("-crf") + 1] == "23"


@pytest.mark.parametrize("field", ["width", "height"])
def test_resize_over_the_image_limit_is_refused_by_validation_before_the_tool_runs(field):
    with pytest.raises(InvalidToolParams) as caught:
        resolved("resize", **{field: 8193})
    assert caught.value.field == field
    assert "at most 8192" in caught.value.message


def test_resize_limit_boundary_is_accepted_by_validation():
    assert resolved("resize", width=8192)["width"] == 8192
    assert resolved("resize", height=8192)["height"] == 8192


def test_spec_exposes_the_resize_limit_to_the_catalog():
    declared = {p["name"]: p for p in spec_of("resize").params}
    assert declared["width"]["max"] == 8192
    assert declared["height"]["max"] == 8192


def test_video_limit_is_still_tighter_than_the_validated_maximum():
    media = ToolMedia(path=Path("/m/clip.mp4"), kind="video", width=64, height=48, duration=1.0)
    params = resolved("resize", width=5000)
    with pytest.raises(InvalidToolParams) as caught:
        tool_of("resize").video_filter(media, params)
    assert caught.value.field == "width"
