from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.platform.media_tools import FilterFragment, ToolContext, ToolMedia, UnsupportedMedia, run_chain
from src.platform.media_tools.fastpath import (
    EVEN_CROP,
    advance_media,
    audio_output_suffix,
    build_timed_args,
    merge_fragments,
    prepare_image,
    run_single,
    save_image,
    video_encoder_args,
    video_output_suffix,
)
from tests.platform.media_tools import fake_tools as fakes
from tests.platform.media_tools.media_factory import make_audio, make_image, make_video, run_ffmpeg


def video(path="/m/clip.mp4", **kw):
    values = dict(path=Path(path), kind="video", width=64, height=48, duration=10.0, fps=10.0, has_audio=True)
    values.update(kw)
    return ToolMedia(**values)


def test_advance_media_applies_reported_changes():
    media = video()

    changed = advance_media(media, FilterFragment(width=32, height=24, duration=5.0, drop_audio=True))

    assert (changed.width, changed.height, changed.duration, changed.has_audio) == (32, 24, 5.0, False)
    assert advance_media(media, FilterFragment()) is media


def test_advance_media_derives_duration_from_a_time_window():
    media = video(duration=10.0)

    assert advance_media(media, FilterFragment(start=2.0, end=5.0)).duration == 3.0
    assert advance_media(media, FilterFragment(start=2.0)).duration == 8.0
    assert advance_media(media, FilterFragment(start=2.0, end=50.0)).duration == 8.0


def test_leading_windows_merge_into_a_single_seek():
    plan = merge_fragments(video(), [FilterFragment(start=1.0, end=6.0), FilterFragment(start=1.0, end=3.0)])

    assert (plan.seek_start, plan.seek_end) == (2.0, 4.0)
    assert plan.vf == [] and plan.af == []
    assert plan.duration == 2.0


def test_inner_window_cannot_extend_past_the_outer_one():
    plan = merge_fragments(video(), [FilterFragment(start=1.0, end=3.0), FilterFragment(start=0.0, end=9.0)])

    assert (plan.seek_start, plan.seek_end) == (1.0, 3.0)


def test_open_ended_window_keeps_the_outer_end():
    plan = merge_fragments(video(), [FilterFragment(start=1.0, end=6.0), FilterFragment(start=2.0)])

    assert (plan.seek_start, plan.seek_end) == (3.0, 6.0)


def test_window_after_a_filter_becomes_trim_and_atrim():
    plan = merge_fragments(video(), [FilterFragment(vf=["hflip"]), FilterFragment(start=1.0, end=2.0)])

    assert plan.seek_start is None and plan.seek_end is None
    assert plan.vf == ["hflip", "trim=start=1.000000:end=2.000000", "setpts=PTS-STARTPTS"]
    assert plan.af == ["atrim=start=1.000000:end=2.000000", "asetpts=PTS-STARTPTS"]


def test_open_ended_inline_window_has_no_end():
    plan = merge_fragments(video(), [FilterFragment(af=["volume=1"]), FilterFragment(start=1.5)])

    assert plan.af[1] == "atrim=start=1.500000"


def test_merge_collects_filters_quality_and_drop_audio_in_order():
    plan = merge_fragments(
        video(),
        [
            FilterFragment(vf=["a"], af=["x"], crf=18, width=32, height=24),
            FilterFragment(vf=["b"], af=["y"], drop_audio=True, crf=28),
        ],
    )

    assert plan.vf == ["a", "b"] and plan.af == ["x", "y"]
    assert plan.crf == 28 and plan.drop_audio
    assert (plan.width, plan.height) == (32, 24)


def test_video_args_for_mp4_have_h264_faststart_and_optional_audio_map(tmp_path):
    media = video()
    plan = merge_fragments(media, [FilterFragment(vf=["hflip"])])

    args = build_timed_args(media, plan, tmp_path / "o.mp4")

    assert args[args.index("-i") + 1] == str(media.path)
    assert args[args.index("-map") + 1] == "0:v:0"
    assert "0:a:0?" in args
    assert args[args.index("-vf") + 1] == "hflip"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[args.index("-movflags") + 1] == "+faststart"
    assert args[args.index("-c:a") + 1] == "copy"
    assert args[-1] == str(tmp_path / "o.mp4")


def test_audio_is_reencoded_when_filtered_or_windowed(tmp_path):
    media = video()

    filtered = build_timed_args(media, merge_fragments(media, [FilterFragment(af=["volume=2"])]), tmp_path / "o.mp4")
    windowed = build_timed_args(media, merge_fragments(media, [FilterFragment(start=1.0, end=2.0)]), tmp_path / "o.mp4")

    for args in (filtered, windowed):
        assert args[args.index("-c:a") + 1] == "aac"
    assert filtered[filtered.index("-af") + 1] == "volume=2"
    assert "-vf" not in windowed
    assert windowed[windowed.index("-ss") + 1] == "1.000000"
    assert windowed[windowed.index("-t") + 1] == "1.000000"


@pytest.mark.parametrize("source", ["/m/clip.avi", "/m/clip.wmv", "/m/clip.ogv", "/m/clip.flv"])
def test_audio_is_reencoded_when_the_output_container_differs_from_the_source(tmp_path, source):
    media = video(path=source)
    out = tmp_path / ("o" + video_output_suffix(media))

    args = build_timed_args(media, merge_fragments(media, [FilterFragment(vf=["hflip"])]), out)

    assert out.suffix == ".mp4"
    assert args[args.index("-c:a") + 1] == "aac"


@pytest.mark.parametrize("source", ["/m/clip.mp4", "/m/CLIP.MP4", "/m/clip.mkv", "/m/clip.mov"])
def test_audio_is_copied_when_the_container_is_kept_and_nothing_touches_it(tmp_path, source):
    media = video(path=source)
    out = tmp_path / ("o" + video_output_suffix(media))

    args = build_timed_args(media, merge_fragments(media, [FilterFragment(vf=["hflip"])]), out)

    assert args[args.index("-c:a") + 1] == "copy"


def test_audio_is_reencoded_for_a_suffix_change_even_in_a_webm_source(tmp_path):
    media = video(path="/m/clip.webm")

    args = build_timed_args(media, merge_fragments(media, [FilterFragment(vf=["hflip"])]), tmp_path / "o.mp4")

    assert args[args.index("-c:a") + 1] == "aac"


@pytest.mark.usefixtures("need_ffmpeg")
def test_wma_audio_from_an_avi_survives_the_fallback_mp4_container(tmp_path):
    source = tmp_path / "wma.avi"
    run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=22050:duration=1",
        "-c:v", "mpeg4", "-c:a", "wmav2", "-shortest", str(source),
    )
    context = ToolContext(tmp_path / "scratch")
    media = context.probe(source)
    assert media.has_audio is True
    mirror = fakes.make_spec("mirror", fakes.Mirror(), accepts=("image", "video"))

    outputs = run_chain([(mirror, {})], [media], context)

    assert outputs[0].path.suffix == ".mp4"
    assert outputs[0].has_audio is True
    assert outputs[0].duration == pytest.approx(1.0, abs=0.2)


def test_audio_filters_are_skipped_when_the_source_has_no_sound(tmp_path):
    media = video(has_audio=False)
    plan = merge_fragments(media, [FilterFragment(vf=["hflip"], af=["atempo=2"])])

    args = build_timed_args(media, plan, tmp_path / "o.mp4")

    assert "-af" not in args


def test_drop_audio_disables_audio_and_skips_the_audio_map(tmp_path):
    media = video()
    plan = merge_fragments(media, [FilterFragment(drop_audio=True)])

    args = build_timed_args(media, plan, tmp_path / "o.mp4")

    assert "-an" in args and "0:a:0?" not in args


def test_odd_or_unknown_sizes_get_the_even_crop_and_even_sizes_do_not(tmp_path):
    for media, expected in (
        (video(width=63, height=48), True),
        (video(width=64, height=47), True),
        (video(width=None, height=None), True),
        (video(width=64, height=48), False),
    ):
        plan = merge_fragments(media, [FilterFragment(vf=["hflip"])])
        args = build_timed_args(media, plan, tmp_path / "o.mp4")
        assert (EVEN_CROP in args[args.index("-vf") + 1]) is expected


def test_vf_is_omitted_when_there_is_nothing_to_filter(tmp_path):
    media = video()

    args = build_timed_args(media, merge_fragments(media, [FilterFragment(start=1.0, end=2.0)]), tmp_path / "o.mp4")

    assert "-vf" not in args


def test_audio_only_args_drop_video_and_pick_a_codec_by_suffix(tmp_path):
    media = ToolMedia(path=Path("/m/tone.wav"), kind="audio", duration=5.0, has_audio=True)
    plan = merge_fragments(media, [FilterFragment(start=1.0, end=3.0, af=["atempo=2"])])

    wav = build_timed_args(media, plan, tmp_path / "o.wav")
    mp3 = build_timed_args(media, plan, tmp_path / "o.mp3")
    m4a = build_timed_args(media, plan, tmp_path / "o.m4a")
    opus = build_timed_args(media, plan, tmp_path / "o.opus")

    assert "-vn" in wav and "-map" not in wav
    assert wav[wav.index("-af") + 1] == "atempo=2"
    assert "-c:a" not in mp3 and mp3[mp3.index("-b:a") + 1] == "192k"
    assert m4a[m4a.index("-c:a") + 1] == "aac"
    assert opus[opus.index("-c:a") + 1] == "libopus"


def test_encoder_args_for_webm_and_quality_offsets():
    webm = video_encoder_args(".webm", 23)
    mp4 = video_encoder_args(".mp4", None)
    mkv = video_encoder_args(".mkv", 18)

    assert webm[webm.index("-c:v") + 1] == "libvpx-vp9"
    assert webm[webm.index("-crf") + 1] == "32"
    assert mp4[mp4.index("-crf") + 1] == "23"
    assert "-movflags" in mp4 and "-movflags" not in mkv
    assert mkv[mkv.index("-crf") + 1] == "18"
    assert all("yuv420p" in args for args in (webm, mp4, mkv))


@pytest.mark.parametrize(
    "name,expected",
    [("a.mp4", ".mp4"), ("a.WEBM", ".webm"), ("a.mov", ".mov"), ("a.avi", ".mp4"), ("a.wmv", ".mp4"), ("a", ".mp4")],
)
def test_video_output_suffix(name, expected):
    assert video_output_suffix(video(path=f"/m/{name}")) == expected


@pytest.mark.parametrize(
    "name,expected",
    [("a.wav", ".wav"), ("a.MP3", ".mp3"), ("a.m4a", ".m4a"), ("a.wma", ".wav"), ("a.xyz", ".wav")],
)
def test_audio_output_suffix(name, expected):
    assert audio_output_suffix(ToolMedia(path=Path(f"/m/{name}"), kind="audio")) == expected


def test_save_image_promotes_alpha_images_from_opaque_only_formats_to_png(tmp_path):
    rgba = Image.new("RGBA", (4, 4), (1, 2, 3, 100))

    for suffix in (".jpg", ".jpeg", ".bmp"):
        written = save_image(rgba, tmp_path / f"a{suffix}")
        assert written.suffix == ".png"
        with Image.open(written) as saved:
            assert saved.mode == "RGBA" and saved.getpixel((0, 0))[3] == 100


def test_save_image_keeps_alpha_for_png_and_webp(tmp_path):
    rgba = Image.new("RGBA", (4, 4), (1, 2, 3, 100))

    png = save_image(rgba, tmp_path / "a.png")
    webp = save_image(rgba, tmp_path / "a.webp")

    assert png.suffix == ".png" and webp.suffix == ".webp"
    with Image.open(webp) as saved:
        assert saved.mode == "RGBA"


def test_save_image_flattens_to_rgb_for_jpeg_without_alpha(tmp_path):
    palette = Image.new("P", (4, 4))
    cmyk = Image.new("CMYK", (4, 4))
    rgba_opaque = Image.new("RGBA", (4, 4), (9, 9, 9, 255))

    for image in (palette, cmyk):
        written = save_image(image, tmp_path / "a.jpg")
        with Image.open(written) as saved:
            assert saved.mode == "RGB"
    assert save_image(rgba_opaque, tmp_path / "b.jpg").suffix == ".png"


def test_save_image_carries_the_icc_profile(tmp_path):
    image = Image.new("RGB", (4, 4))
    image.info["icc_profile"] = b"profile-bytes"

    written = save_image(image, tmp_path / "a.png")

    with Image.open(written) as saved:
        assert saved.info.get("icc_profile") == b"profile-bytes"


def test_prepare_image_bakes_in_the_exif_orientation(tmp_path):
    path = tmp_path / "rot.jpg"
    image = Image.new("RGB", (40, 30), (9, 9, 9))
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(path, exif=exif)

    prepared = prepare_image(path)

    assert prepared.size == (30, 40)


def test_run_single_image_uses_the_in_memory_path(tmp_path):
    context = ToolContext(tmp_path / "scratch")
    media = ToolMedia(path=make_image(tmp_path / "a.png", (40, 30)), kind="image", width=40, height=30)

    result = run_single(fakes.Mirror(), media, {}, context)

    assert (result.width, result.height) == (40, 30)
    assert context.commands == []


def test_run_single_refuses_a_tool_without_the_matching_fast_path(tmp_path):
    context = ToolContext(tmp_path / "scratch")
    image = ToolMedia(path=make_image(tmp_path / "a.png"), kind="image")
    clip = ToolMedia(path=Path("/m/clip.mp4"), kind="video")

    with pytest.raises(UnsupportedMedia):
        run_single(fakes.Window(), image, {}, context)
    with pytest.raises(UnsupportedMedia):
        run_single(fakes.Tint(), clip, {}, context)


@pytest.mark.usefixtures("need_ffmpeg")
def test_default_run_of_a_video_tool_goes_through_one_pass(tmp_path):
    context = ToolContext(tmp_path / "scratch")
    source = make_video(tmp_path / "v.mp4", seconds=1.0, size=(64, 48))
    media = context.probe(source)

    outputs = fakes.Mirror().run([media], {}, context)

    assert len(outputs) == 1 and len(context.commands) == 1
    assert (outputs[0].width, outputs[0].height) == (64, 48)


@pytest.mark.usefixtures("need_ffmpeg")
def test_audio_run_keeps_the_source_suffix_when_encodable(tmp_path):
    context = ToolContext(tmp_path / "scratch")
    media = context.probe(make_audio(tmp_path / "tone.wav", 1.0))

    outputs = fakes.Faster().run([media], {"factor": 2.0}, context)

    assert outputs[0].path.suffix == ".wav"
    assert outputs[0].duration == pytest.approx(0.5, abs=0.15)


def test_a_frame_cap_is_a_hard_limit_with_timestamps_passed_through():
    plan = merge_fragments(video(), [FilterFragment(frames=9)])

    args = build_timed_args(video(), plan, Path("/m/out.mp4"))

    assert args[args.index("-frames:v") + 1] == "9"
    assert args[args.index("-fps_mode") + 1] == "passthrough"


def test_the_smallest_frame_cap_wins_when_fragments_merge():
    assert merge_fragments(video(), [FilterFragment(frames=9), FilterFragment(frames=5)]).frames == 5
    assert merge_fragments(video(), [FilterFragment(), FilterFragment(frames=7)]).frames == 7


def test_no_frame_cap_means_no_frames_flag():
    args = build_timed_args(video(), merge_fragments(video(), [FilterFragment(vf=["hflip"])]), Path("/m/out.mp4"))

    assert "-frames:v" not in args and "-fps_mode" not in args


def test_cropping_to_even_sides_also_resets_the_pixel_aspect():
    odd = video(width=63, height=47)

    args = build_timed_args(odd, merge_fragments(odd, [FilterFragment(vf=["hflip"])]), Path("/m/out.mp4"))

    assert args[args.index("-vf") + 1] == f"hflip,{EVEN_CROP},setsar=1"
