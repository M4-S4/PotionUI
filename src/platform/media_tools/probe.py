from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

from src.platform.media_tools.types import ToolFailed, ToolMedia, UnsupportedMedia
from src.platform.util.video_transcode import find_ffmpeg, find_ffprobe

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_SECONDS = 30

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".mpg", ".mpeg", ".ogv"})
AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma", ".aiff", ".aif"})

_ORIENTATION_TAG = 0x0112
_TRANSPOSED_ORIENTATIONS = (5, 6, 7, 8)
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VIDEO_LINE_RE = re.compile(r"Stream #\d+:\d+[^:]*:\s*Video:(.*)")
_SIZE_RE = re.compile(r"(?<![\dx])(\d{2,5})x(\d{2,5})(?![\dx])")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*tbr")
_AUDIO_LINE_RE = re.compile(r"Stream #\d+:\d+[^:]*:\s*Audio:\s*([\w]+)[^\n]*?,\s*[\d.]+ Hz,[^,\n]*,\s*(\w+)(?:\s*\((\d+) bit\))?")
_ROTATION_RE = re.compile(r"rotation of (-?\d+(?:\.\d+)?) degrees")


def kind_for_suffix(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return None


def probe(path: Path | str, kind: Optional[str] = None) -> ToolMedia:
    path = Path(path)
    if not path.is_file():
        raise UnsupportedMedia("The media file is missing")
    resolved = kind or kind_for_suffix(path) or _sniff_kind(path)
    if resolved == "image":
        return _probe_image(path)
    if resolved in ("video", "audio"):
        return _probe_timed(path, resolved)
    raise UnsupportedMedia("Cannot tell what kind of media this is")


def _sniff_kind(path: Path) -> Optional[str]:
    try:
        with Image.open(path):
            return "image"
    except (UnidentifiedImageError, OSError):
        pass
    parsed = _banner_info(path)
    if parsed is None:
        return None
    return "video" if parsed.get("width") else "audio"


def has_alpha_mode(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        image.mode == "P" and "transparency" in image.info
    )


def _probe_image(path: Path) -> ToolMedia:
    try:
        with Image.open(path) as image:
            width, height = image.size
            alpha = has_alpha_mode(image)
            try:
                orientation = image.getexif().get(_ORIENTATION_TAG)
            except Exception:
                orientation = None
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Could not read image %s: %s", path, exc)
        raise UnsupportedMedia("This image cannot be read") from exc
    if orientation in _TRANSPOSED_ORIENTATIONS:
        width, height = height, width
    return ToolMedia(path=path, kind="image", width=width, height=height, has_alpha=alpha)


def _probe_timed(path: Path, kind: str) -> ToolMedia:
    info = _ffprobe_info(path) or _banner_info(path)
    if info is None:
        raise UnsupportedMedia(f"This {kind} cannot be read")
    if kind == "video" and not info.get("width"):
        raise UnsupportedMedia("This video has no picture to read")
    return ToolMedia(
        path=path,
        kind=kind,
        width=info.get("width") if kind == "video" else None,
        height=info.get("height") if kind == "video" else None,
        duration=info.get("duration"),
        fps=info.get("fps") if kind == "video" else None,
        has_audio=bool(info.get("has_audio")),
        has_alpha=False if kind == "video" else None,
        audio_codec=info.get("audio_codec"),
        bit_depth=info.get("bit_depth"),
    )


def _rate(value: object) -> Optional[float]:
    text = str(value or "")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator) if float(denominator) else None
        return float(text) if text else None
    except ValueError:
        return None


def _bits(value: object) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _ffprobe_info(path: Path) -> Optional[dict]:
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return None
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path.absolute())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    streams = payload.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if video is None and audio is None:
        return None
    duration = None
    for source in ((video or {}).get("duration"), (payload.get("format") or {}).get("duration"), (audio or {}).get("duration")):
        duration = _positive(source)
        if duration:
            break
    info: dict = {"duration": duration, "has_audio": audio is not None}
    if audio is not None:
        info["audio_codec"] = audio.get("codec_name")
        info["bit_depth"] = _bits(audio.get("bits_per_raw_sample")) or _bits(audio.get("bits_per_sample"))
    if video is not None:
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        rotation = _stream_rotation(video)
        if rotation in (90, 270):
            width, height = height, width
        info.update(width=width or None, height=height or None, fps=_rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")))
    return info


def _stream_rotation(stream: dict) -> int:
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                return 0
    try:
        return int(float((stream.get("tags") or {}).get("rotate") or 0)) % 360
    except (TypeError, ValueError):
        return 0


def parse_ffmpeg_banner(text: str) -> Optional[dict]:
    duration = None
    found = _DURATION_RE.search(text)
    if found:
        hours, minutes, seconds = found.groups()
        duration = _positive(int(hours) * 3600 + int(minutes) * 60 + float(seconds))
    has_audio = bool(re.search(r"Stream #\d+:\d+[^:]*:\s*Audio:", text))
    video_line = None
    for candidate in _VIDEO_LINE_RE.finditer(text):
        if "attached pic" in candidate.group(0):
            continue
        video_line = candidate.group(1)
        break
    if duration is None and video_line is None and not has_audio:
        return None
    info: dict = {"duration": duration, "has_audio": has_audio}
    audio_line = _AUDIO_LINE_RE.search(text)
    if audio_line:
        codec, sample_format, depth = audio_line.groups()
        info["audio_codec"] = codec
        info["bit_depth"] = int(depth) if depth else {"u8": 8, "s16": 16, "s16p": 16, "s32": 32, "s32p": 32}.get(sample_format)
    if video_line is not None:
        size = _SIZE_RE.search(re.sub(r"\[[^\]]*\]", "", video_line))
        width, height = (int(size.group(1)), int(size.group(2))) if size else (None, None)
        rotation = _ROTATION_RE.search(text)
        if rotation and width and height and int(round(abs(float(rotation.group(1))))) % 180 == 90:
            width, height = height, width
        rate = _FPS_RE.search(video_line) or _TBR_RE.search(video_line)
        info.update(width=width, height=height, fps=float(rate.group(1)) if rate else None)
    return info


def _banner_info(path: Path) -> Optional[dict]:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise ToolFailed("ffmpeg is not available on this server")
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostdin", "-i", str(path.absolute())],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_ffmpeg_banner(result.stderr or "")
