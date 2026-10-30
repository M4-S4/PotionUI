from __future__ import annotations

import pytest
from PIL import Image

from src.platform.media_tools import (
    InvalidToolParams,
    MediaTool,
    ToolContext,
    ToolSpec,
    UnsupportedMedia,
    probe,
    run_chain,
)
from tests.platform.media_tools.media_factory import make_image, make_video
from tests.platform.media_tools.ops.geometry_support import (
    SPECS,
    first_frame,
    is_blue,
    is_red,
    make_marker_video,
    make_odd_video,
    run_steps,
)


def video_filter_of(command: list[str]) -> str:
    return command[command.index("-vf") + 1]


class CopyThrough(MediaTool):
    def run(self, inputs, params, ctx):
        outputs = []
        for media in inputs:
            target = ctx.new_path(media.path.suffix)
            ctx.ffmpeg(["-i", str(media.path), "-c", "copy", str(target)])
            outputs.append(ctx.probe(target, media.kind))
        return outputs


def copy_through_spec() -> ToolSpec:
    return ToolSpec(
        id="copythrough",
        label="Copy through",
        accepts=("video",),
        impl=CopyThrough(),
    )


def test_video_crop_resize_rotate_is_one_ffmpeg_invocation(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
    out = run_steps(
        [
            ("crop", {"x": 8, "y": 6, "width": 40, "height": 30}),
            ("resize", {"width": 20, "height": 16}),
            ("rotate", {"degrees": 90}),
        ],
        path,
        ctx,
    )
    assert len(ctx.commands) == 1
    chain = video_filter_of(ctx.commands[0])
    assert chain.index("crop=40:30:8:6") < chain.index("scale=20:16") < chain.index("transpose=1")
    assert (out.width, out.height) == (16, 20)
    assert out.has_audio is True
    assert out.duration == pytest.approx(1.0, abs=0.15)


def test_fused_video_chain_applies_the_filters_in_order(need_ffmpeg, tmp_path, ctx):
    path = make_marker_video(tmp_path / "in.mp4", size=(64, 48), box=(16, 12))
    out = run_steps(
        [("flip", {"axis": "horizontal"}), ("rotate", {"degrees": 90})],
        path,
        ctx,
    )
    assert len(ctx.commands) == 1
    assert (out.width, out.height) == (48, 64)
    frame = first_frame(out.path, tmp_path / "frame.png")
    assert is_red(frame.getpixel((43, 59)))
    assert is_blue(frame.getpixel((4, 4)))


def test_last_resize_quality_wins_in_a_fused_chain(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    run_steps(
        [
            ("resize", {"width": 32, "quality": "small"}),
            ("resize", {"width": 16, "quality": "high"}),
        ],
        path,
        ctx,
    )
    assert len(ctx.commands) == 1
    command = ctx.commands[0]
    assert command[command.index("-crf") + 1] == "18"


@pytest.mark.parametrize("quality, crf", [("high", "18"), ("balanced", "23"), ("small", "28")])
def test_quality_presets_reach_the_encoder(need_ffmpeg, tmp_path, ctx, quality, crf):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    run_steps([("crop", {"width": 32, "height": 24}), ("resize", {"width": 16, "quality": quality})], path, ctx)
    command = ctx.commands[0]
    assert command[command.index("-crf") + 1] == crf


def test_crop_after_resize_is_checked_against_the_resized_video_frame(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    with pytest.raises(InvalidToolParams) as caught:
        run_steps(
            [("resize", {"width": 32, "height": 24}), ("crop", {"width": 40, "height": 10})],
            path,
            ctx,
        )
    assert caught.value.field == "width"
    assert ctx.commands == []


def test_crop_after_rotate_is_checked_against_the_rotated_video_frame(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", size=(64, 48))
    run_steps([("rotate", {"degrees": 90}), ("crop", {"width": 40, "height": 60})], path, ctx)
    assert len(ctx.commands) == 1
    with pytest.raises(InvalidToolParams):
        run_steps([("rotate", {"degrees": 90}), ("crop", {"width": 60, "height": 40})], path, ToolContext(tmp_path / "other"))


def test_image_chain_runs_in_memory_and_saves_once(tmp_path, ctx):
    path = tmp_path / "in.png"
    make_image(path, (40, 30), "RGB", (10, 20, 30))
    out = run_steps(
        [
            ("crop", {"x": 0, "y": 0, "width": 40, "height": 20}),
            ("resize", {"width": 20, "height": 10}),
            ("rotate", {"degrees": 90}),
        ],
        path,
        ctx,
    )
    assert (out.width, out.height) == (10, 20)
    assert ctx.commands == []
    assert len(list(ctx.scratch_dir.iterdir())) == 1


def test_image_chain_orders_the_pixels_like_the_steps(tmp_path, ctx):
    path = tmp_path / "in.png"
    image = Image.new("RGB", (4, 2), (0, 0, 0))
    image.putpixel((0, 0), (255, 0, 0))
    image.save(path)
    out = run_steps([("flip", {"axis": "horizontal"}), ("rotate", {"degrees": 90})], path, ctx)
    with Image.open(out.path) as saved:
        assert saved.size == (2, 4)
        assert saved.getpixel((1, 3)) == (255, 0, 0)


def test_crop_after_resize_is_checked_against_the_resized_image(tmp_path, ctx):
    path = tmp_path / "in.png"
    make_image(path, (40, 30))
    with pytest.raises(InvalidToolParams) as caught:
        run_steps(
            [("resize", {"width": 20, "height": 15}), ("crop", {"width": 30, "height": 10})],
            path,
            ctx,
        )
    assert caught.value.field == "width"
    assert list(ctx.scratch_dir.glob("*")) == []


def test_crop_that_fits_the_original_but_not_the_resized_image_is_refused(tmp_path, ctx):
    path = tmp_path / "in.png"
    make_image(path, (40, 30))
    with pytest.raises(InvalidToolParams):
        run_steps([("resize", {"width": 20}), ("crop", {"width": 30, "height": 10})], path, ctx)


def test_invalid_params_in_a_chain_carry_the_step_index(tmp_path, ctx):
    path = tmp_path / "in.png"
    make_image(path, (40, 30))
    with pytest.raises(InvalidToolParams) as caught:
        run_steps(
            [("flip", {}), ("rotate", {}), ("crop", {"width": 0, "height": 5})],
            path,
            ctx,
        )
    assert caught.value.field == "width"
    assert caught.value.step_index == 2


def test_video_filters_then_a_tool_without_a_fast_path_still_chains(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
    outputs = run_chain(
        [(SPECS["crop"], {"x": 0, "y": 0, "width": 32, "height": 24}), (copy_through_spec(), {})],
        [probe(path)],
        ctx,
    )
    assert len(ctx.commands) == 2
    assert (outputs[0].width, outputs[0].height) == (32, 24)
    assert outputs[0].has_audio is True
    assert outputs[0].path.is_file()


def test_a_tool_without_a_fast_path_then_video_filters_chains(need_ffmpeg, tmp_path, ctx):
    path = make_video(tmp_path / "in.mp4", seconds=1.0, size=(64, 48))
    outputs = run_chain(
        [(copy_through_spec(), {}), (SPECS["rotate"], {"degrees": 90}), (SPECS["flip"], {"axis": "vertical"})],
        [probe(path)],
        ctx,
    )
    assert len(ctx.commands) == 2
    assert (outputs[0].width, outputs[0].height) == (48, 64)


def test_odd_sized_video_comes_out_even(need_ffmpeg, tmp_path, ctx):
    path = make_odd_video(tmp_path / "odd.mp4", size=(65, 49))
    source = probe(path)
    assert (source.width, source.height) == (65, 49)
    out = run_steps([("flip", {"axis": "horizontal"})], path, ctx)
    assert out.width % 2 == 0 and out.height % 2 == 0
    assert (out.width, out.height) == (64, 48)
    assert out.has_audio is True


def test_odd_sized_video_rotated_comes_out_even(need_ffmpeg, tmp_path, ctx):
    path = make_odd_video(tmp_path / "odd.mp4", size=(65, 49))
    out = run_steps([("rotate", {"degrees": 90})], path, ctx)
    assert (out.width, out.height) == (48, 64)


def test_crop_subject_after_color_key_on_a_jpeg_uses_the_alpha_the_key_made(tmp_path, ctx):
    path = tmp_path / "screen.jpg"
    screen = Image.new("RGB", (60, 60), (0, 177, 64))
    screen.paste((200, 30, 30), (20, 20, 40, 40))
    screen.save(path, quality=95)
    out = run_steps(
        [("color_key", {}), ("crop_subject", {"margin": 0})],
        path,
        ctx,
    )
    assert out.path.suffix == ".png"
    assert out.has_alpha is True
    assert ctx.commands == []
    assert 20 <= out.width < 60 and 20 <= out.height < 60
    assert len(list(ctx.scratch_dir.iterdir())) == 1
    with Image.open(out.path) as saved:
        assert saved.mode == "RGBA"
        assert saved.getpixel((out.width // 2, out.height // 2))[3] == 255


def test_crop_subject_on_an_opaque_image_is_refused(tmp_path, ctx):
    path = tmp_path / "opaque.jpg"
    make_image(path, (40, 30))
    with pytest.raises(UnsupportedMedia):
        run_steps([("crop_subject", {})], path, ctx)


def test_crop_subject_after_a_tool_that_adds_no_alpha_is_refused(tmp_path, ctx):
    path = tmp_path / "opaque.png"
    make_image(path, (40, 30))
    with pytest.raises(UnsupportedMedia):
        run_steps([("rotate", {}), ("crop_subject", {})], path, ctx)


def test_fit_canvas_then_crop_subject_sees_the_transparent_canvas(tmp_path, ctx):
    path = tmp_path / "in.png"
    make_image(path, (20, 20), "RGB", (200, 30, 30))
    out = run_steps(
        [("fit_canvas", {"width": 60, "height": 60, "scale_percent": 50}), ("crop_subject", {})],
        path,
        ctx,
    )
    assert (out.width, out.height) == (30, 30)
    assert out.has_alpha is True


@pytest.mark.parametrize("tool_id, params", [
    ("crop", {"x": 1, "y": 1, "width": 10, "height": 6}),
    ("resize", {"width": 12}),
    ("rotate", {"degrees": 270}),
    ("flip", {"axis": "horizontal"}),
])
def test_png_alpha_survives_every_image_tool(tmp_path, ctx, tool_id, params):
    path = tmp_path / "in.png"
    image = Image.new("RGBA", (20, 10), (10, 20, 30, 0))
    image.paste((10, 20, 30, 200), (5, 2, 15, 8))
    image.save(path)
    out = run_steps([(tool_id, params)], path, ctx)
    assert out.has_alpha is True
    with Image.open(out.path) as saved:
        assert saved.mode == "RGBA"
        alpha = list(saved.getchannel("A").getdata())
        assert min(alpha) <= 5
        assert max(alpha) >= 190
