from __future__ import annotations

import importlib
import json
import subprocess

import pytest
from PIL import Image

from src.platform.media_tools import UnsupportedMedia, probe
from src.platform.media_tools.probe import kind_for_suffix, parse_ffmpeg_banner
from tests.platform.media_tools.media_factory import make_audio, make_image, make_video

probe_module = importlib.import_module("src.platform.media_tools.probe")

VIDEO_BANNER = "\n".join(
    [
        "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':",
        "  Duration: 00:00:02.04, start: 0.000000, bitrate: 123 kb/s",
        "  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 320x240 [SAR 1:1 DAR 4:3], 25 fps, 25 tbr, 12800 tbn (default)",
        "  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706d), 44100 Hz, stereo, fltp, 64 kb/s (default)",
    ]
)

ROTATED_BANNER = "\n".join(
    [
        "  Duration: 00:00:05.00, start: 0.000000, bitrate: 900 kb/s",
        "  Stream #0:0(und): Video: h264 (High), yuv420p, 1920x1080, 29.97 fps, 29.97 tbr, 90k tbn",
        "    Side data:",
        "      displaymatrix: rotation of -90.00 degrees",
    ]
)

COVER_ART_BANNER = "\n".join(
    [
        "  Duration: 00:03:00.00, start: 0.000000, bitrate: 192 kb/s",
        "  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 192 kb/s",
        "  Stream #0:1: Video: mjpeg (Baseline), yuvj420p(pc, bt470bg/unknown/unknown), 500x500 [SAR 1:1 DAR 1:1], 90k tbr, 90k tbn (attached pic)",
    ]
)

AUDIO_ONLY_BANNER = "\n".join(
    [
        "  Duration: 00:00:02.00, bitrate: 352 kb/s",
        "  Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 22050 Hz, mono, s16, 352 kb/s",
    ]
)


def test_banner_parse_of_a_normal_video():
    info = parse_ffmpeg_banner(VIDEO_BANNER)

    assert info == {"duration": 2.04, "has_audio": True, "width": 320, "height": 240, "fps": 25.0}


def test_banner_parse_ignores_the_hex_codec_tag_when_reading_the_size():
    info = parse_ffmpeg_banner(VIDEO_BANNER)

    assert (info["width"], info["height"]) == (320, 240)


def test_banner_parse_swaps_sides_for_a_rotated_video():
    info = parse_ffmpeg_banner(ROTATED_BANNER)

    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["fps"] == 29.97


def test_banner_parse_skips_attached_cover_art():
    info = parse_ffmpeg_banner(COVER_ART_BANNER)

    assert info["has_audio"] is True
    assert "width" not in info
    assert info["duration"] == 180.0


def test_banner_parse_of_audio_only_and_of_garbage():
    assert parse_ffmpeg_banner(AUDIO_ONLY_BANNER) == {"duration": 2.0, "has_audio": True}
    assert parse_ffmpeg_banner("nothing useful here") is None


@pytest.mark.parametrize(
    "name,kind",
    [("a.PNG", "image"), ("a.jpeg", "image"), ("a.webm", "video"), ("a.MKV", "video"), ("a.flac", "audio"), ("a.m4a", "audio"), ("a.xyz", None), ("noext", None)],
)
def test_kind_for_suffix(name, kind, tmp_path):
    assert kind_for_suffix(tmp_path / name) == kind


def test_image_probe_reads_size_and_alpha(tmp_path):
    opaque = probe(make_image(tmp_path / "a.png", (40, 30)))
    clear = probe(make_image(tmp_path / "b.png", (10, 20), mode="RGBA", color=(1, 2, 3, 0)))
    palette = tmp_path / "c.png"
    Image.new("P", (5, 5)).save(palette, transparency=0)

    assert (opaque.kind, opaque.width, opaque.height, opaque.has_alpha) == ("image", 40, 30, False)
    assert (clear.width, clear.height, clear.has_alpha) == (10, 20, True)
    assert probe(palette).has_alpha is True
    assert opaque.duration is None and opaque.has_audio is None


def test_image_probe_reports_the_size_after_exif_orientation(tmp_path):
    path = tmp_path / "rot.jpg"
    image = Image.new("RGB", (40, 30), (9, 9, 9))
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(path, exif=exif)

    media = probe(path)

    assert (media.width, media.height) == (30, 40)


@pytest.mark.usefixtures("need_ffmpeg")
def test_video_probe_with_sound(tmp_path):
    media = probe(make_video(tmp_path / "v.mp4", seconds=1.5, size=(64, 48), fps=10))

    assert (media.kind, media.width, media.height, media.has_audio, media.has_alpha) == ("video", 64, 48, True, False)
    assert media.duration == pytest.approx(1.5, abs=0.1)
    assert media.fps == 10.0


@pytest.mark.usefixtures("need_ffmpeg")
def test_video_probe_without_sound(tmp_path):
    media = probe(make_video(tmp_path / "v.mp4", seconds=1.0, audio=False))

    assert media.has_audio is False


@pytest.mark.usefixtures("need_ffmpeg")
def test_audio_probe(tmp_path):
    media = probe(make_audio(tmp_path / "a.wav", 2.0))

    assert (media.kind, media.width, media.height, media.fps, media.has_audio) == ("audio", None, None, None, True)
    assert media.duration == pytest.approx(2.0, abs=0.05)


@pytest.mark.usefixtures("need_ffmpeg")
def test_explicit_kind_overrides_the_suffix(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=1.0)

    as_audio = probe(path, "audio")

    assert as_audio.kind == "audio"
    assert as_audio.width is None


@pytest.mark.usefixtures("need_ffmpeg")
def test_unknown_suffix_is_sniffed(tmp_path):
    image = tmp_path / "blob.bin"
    make_image(tmp_path / "x.png").rename(image)
    video = tmp_path / "clip.dat"
    make_video(tmp_path / "x.mp4", seconds=0.5).rename(video)
    audio = tmp_path / "tone.dat"
    make_audio(tmp_path / "x.wav", 0.5).rename(audio)

    assert [probe(p).kind for p in (image, video, audio)] == ["image", "video", "audio"]


def test_missing_file_and_garbage_are_unsupported_media(tmp_path):
    garbage = tmp_path / "bad.png"
    garbage.write_bytes(b"not an image")

    with pytest.raises(UnsupportedMedia):
        probe(tmp_path / "missing.png")
    with pytest.raises(UnsupportedMedia):
        probe(garbage)


@pytest.mark.usefixtures("need_ffmpeg")
def test_garbage_video_is_unsupported_media(tmp_path):
    garbage = tmp_path / "bad.mp4"
    garbage.write_bytes(b"this is not a video at all")

    with pytest.raises(UnsupportedMedia):
        probe(garbage)


@pytest.mark.usefixtures("need_ffmpeg")
def test_audio_only_file_probed_as_video_is_refused(tmp_path):
    path = make_audio(tmp_path / "tone.wav", 0.5)

    with pytest.raises(UnsupportedMedia):
        probe(path, "video")


def test_ffprobe_json_is_preferred_and_rotation_swaps_sides(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    payload = {
        "streams": [
            {
                "codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001",
                "duration": "4.5", "side_data_list": [{"rotation": -90}],
            },
            {"codec_type": "audio"},
        ],
        "format": {"duration": "4.6"},
    }
    monkeypatch.setattr(probe_module, "find_ffprobe", lambda: "/fake/ffprobe")
    monkeypatch.setattr(
        probe_module.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(payload), stderr=""),
    )

    media = probe(clip)

    assert (media.width, media.height) == (1080, 1920)
    assert media.duration == 4.5
    assert media.fps == pytest.approx(29.97, abs=0.01)
    assert media.has_audio is True


def test_ffprobe_attached_picture_is_not_a_video_stream(monkeypatch, tmp_path):
    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")
    payload = {
        "streams": [
            {"codec_type": "audio", "duration": "12"},
            {"codec_type": "video", "width": 500, "height": 500, "disposition": {"attached_pic": 1}},
        ],
        "format": {"duration": "12"},
    }
    monkeypatch.setattr(probe_module, "find_ffprobe", lambda: "/fake/ffprobe")
    monkeypatch.setattr(
        probe_module.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(payload), stderr=""),
    )

    media = probe(song)

    assert media.kind == "audio"
    assert media.width is None
    assert media.duration == 12.0
    with pytest.raises(UnsupportedMedia):
        probe(song, "video")


@pytest.mark.usefixtures("need_ffmpeg")
def test_a_failing_ffprobe_falls_back_to_the_ffmpeg_banner(monkeypatch, tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=1.0, size=(64, 48))
    monkeypatch.setattr(probe_module, "find_ffprobe", lambda: "/nonexistent/ffprobe")

    media = probe(path)

    assert (media.width, media.height) == (64, 48)


TOKEN = "zq7secrettoken"


def private_dir(tmp_path):
    folder = tmp_path / TOKEN
    folder.mkdir()
    return folder


def assert_message_is_clean(message, tmp_path, *forbidden):
    assert TOKEN not in message
    assert str(tmp_path) not in message
    for text in forbidden:
        assert text not in message


def test_corrupt_image_message_is_fixed_and_leaks_neither_path_nor_exception_text(tmp_path, caplog):
    bad = private_dir(tmp_path) / "pic.png"
    bad.write_bytes(b"definitely not an image")
    try:
        Image.open(bad)
    except Exception as exc:
        raw = str(exc)
    assert TOKEN in raw

    with caplog.at_level("WARNING"):
        with pytest.raises(UnsupportedMedia) as raised:
            probe(bad)

    assert raised.value.message == "This image cannot be read"
    assert str(raised.value) == "This image cannot be read"
    assert_message_is_clean(raised.value.message, tmp_path, raw)
    assert any(TOKEN in record.getMessage() for record in caplog.records)


def test_missing_file_message_does_not_name_the_file(tmp_path):
    with pytest.raises(UnsupportedMedia) as raised:
        probe(private_dir(tmp_path) / "gone.png")

    assert raised.value.message == "The media file is missing"
    assert_message_is_clean(raised.value.message, tmp_path, "gone.png")


@pytest.mark.usefixtures("need_ffmpeg")
def test_unrecognisable_file_message_does_not_name_the_file(tmp_path):
    blob = private_dir(tmp_path) / "blob.zzz"
    blob.write_bytes(b"neither an image nor a video")

    with pytest.raises(UnsupportedMedia) as raised:
        probe(blob)

    assert raised.value.message == "Cannot tell what kind of media this is"
    assert_message_is_clean(raised.value.message, tmp_path, "blob.zzz")


@pytest.mark.usefixtures("need_ffmpeg")
def test_corrupt_video_and_audio_messages_leak_nothing(tmp_path):
    folder = private_dir(tmp_path)
    video = folder / "clip.mp4"
    audio = folder / "tone.wav"
    video.write_bytes(b"this is not a video at all")
    audio.write_bytes(b"this is not audio at all")

    for path in (video, audio):
        with pytest.raises(UnsupportedMedia) as raised:
            probe(path)
        assert_message_is_clean(raised.value.message, tmp_path, path.name)
