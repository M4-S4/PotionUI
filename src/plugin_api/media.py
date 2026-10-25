"""Working with images.

`convert_image_to_base64` encodes a PIL image for transport - to the frontend, or
to a vision model that takes images inline.

`BackgroundMattingModel` runs BiRefNet background matting (an RGB(A) image in,
RGBA out at the same size, alpha = the matted subject). Load it, use it,
release it - it is never resident:

    model = BackgroundMattingModel.from_checkpoint(checkpoint_path)  # raises
                                                                       # ValueError
                                                                       # if the
                                                                       # file is
                                                                       # missing
                                                                       # or the
                                                                       # wrong
                                                                       # shape
    try:
        model.to("cuda")
        matted = model(image)
    finally:
        model.cpu()
        del model

`checkpoint_path` is a `.safetensors` file the plugin/user supplies (e.g. from
the model depot's `detection_segm/` folder, the convention
`content/plugins/marketplace/trellis2` established) - nothing is ever downloaded.

`await extract_frame(user, ref, at="last")` takes one frame of a video the user
can see (a `resolve_media_ref` reference), stores it as a PNG upload of that
user and returns it as a `ResolvedMedia`, ready for a media field or
`bind_director_media`. `await save_user_media(user, path)` stores a file your
plugin wrote (an absolute path) the same way. Both store a Library item by
default (`purpose="user_upload"`); `purpose="derived_artifact"` keeps the file
servable but out of the Library. They raise `MediaRefError` with a `code`, and
let `LimitExceeded` through when the user's storage plan is full.
"""

import asyncio
import io
import math
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, Union

from src.features.media.validators import UPLOAD_PURPOSE_USER, UPLOAD_PURPOSES
from src.pipelines.pipes._shared.media.frame_extract import extract_frame as _extract_video_frame
from src.platform.filesystem.storage_driver import local_copy
from src.platform.plugins.runtime_registries import get_container
from src.platform.util.imaging import convert_image_to_base64
from src.platform.util.video_stream import (
    FFmpegNotFoundError,
    StreamingMp4Writer,
    has_audio_stream,
    mux_audio_from_source,
)
from src.platform.util.video_transcode import (
    FfmpegUnavailableError,
    TranscodeSpec,
    VideoProbe,
    VideoTranscodeError,
    find_ffmpeg,
    probe_video,
    transcode_video,
)
from src.plugin_api.generation import (
    HistoryMediaRef,
    LibraryMediaRef,
    MediaRefError,
    ResolvedMedia,
    resolve_media_ref,
)

if TYPE_CHECKING:
    from src.platform.runtime.native.matting import BackgroundMattingModel

__all__ = [
    "BackgroundMattingModel",
    "FFmpegNotFoundError",
    "FfmpegUnavailableError",
    "StreamingMp4Writer",
    "TranscodeSpec",
    "VideoProbe",
    "VideoTranscodeError",
    "convert_image_to_base64",
    "extract_frame",
    "find_ffmpeg",
    "has_audio_stream",
    "mux_audio_from_source",
    "probe_video",
    "save_user_media",
    "transcode_video",
]


def __getattr__(name):
    if name == "BackgroundMattingModel":
        from src.platform.runtime.native.matting import BackgroundMattingModel
        return BackgroundMattingModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _owner_id(user) -> str:
    user_id = getattr(user, "id", None)
    if not user_id:
        raise PermissionError("A user is required to store media")
    return user_id


def _checked_purpose(purpose: str) -> str:
    if purpose not in UPLOAD_PURPOSES:
        raise MediaRefError("invalid_purpose", f"purpose must be one of {', '.join(UPLOAD_PURPOSES)}")
    return purpose


def _seconds(at: Any) -> Optional[float]:
    if isinstance(at, str) and at == "last":
        return None
    if isinstance(at, bool) or not isinstance(at, (int, float)) or not math.isfinite(at) or at < 0:
        raise MediaRefError("invalid_ref", "at must be 'last' or a number of seconds from the start")
    return float(at)


def _frame_index(path: Path, seconds: Optional[float]) -> int:
    if seconds is None:
        return -1
    probe = probe_video(path)
    if probe is None or not probe.fps or probe.fps <= 0:
        raise MediaRefError("extract_failed", "The video's frame rate could not be read")
    index = int(math.floor(seconds * probe.fps + 1e-6))
    last = int(round(probe.duration * probe.fps)) - 1
    if 0 <= last < index and seconds <= probe.duration:
        return last
    return index


@contextmanager
def _video_file(container, kind: str, key: str) -> Iterator[Path]:
    suffix = Path(key).suffix
    if kind == "history":
        with container.file_service.local_copy_of(key, suffix) as path:
            yield path
    else:
        with local_copy(container.media_store.storage_driver, key, suffix) as path:
            yield path


def _frame_png(container, kind: str, key: str, seconds: Optional[float]) -> bytes:
    with _video_file(container, kind, key) as path:
        index = _frame_index(path, seconds)
        try:
            frame = _extract_video_frame(path, index)
        except Exception as exc:
            raise MediaRefError("extract_failed", f"The frame could not be extracted: {exc}") from exc
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    return buffer.getvalue()


async def _store(container, data: bytes, filename: str, content_type: str, user_id: str, purpose: str) -> ResolvedMedia:
    try:
        stored = await container.media_store.upload_media(data, filename, content_type, user_id, purpose)
    except ValueError as exc:
        raise MediaRefError("upload_rejected", str(exc)) from exc
    return {
        "value": stored.relative_path,
        "media_type": content_type.split("/", 1)[0],
        "original_filename": filename,
        "origin": None,
    }


async def extract_frame(
    user,
    ref: Union[HistoryMediaRef, LibraryMediaRef],
    at: Union[Literal["last"], float] = "last",
    purpose: str = UPLOAD_PURPOSE_USER,
) -> ResolvedMedia:
    """Store one frame of the referenced video as the user's PNG upload.

    `at` is "last" or seconds from the start of the video, turned into a frame
    with the frame rate `probe_video` reads. Raises `MediaRefError` with
    `invalid_ref`, `not_found`, `not_a_video`, `extract_failed`,
    `invalid_purpose` or `upload_rejected`."""
    user_id = _owner_id(user)
    seconds = _seconds(at)
    purpose = _checked_purpose(purpose)
    resolved = resolve_media_ref(user, ref)
    if str(resolved["media_type"] or "").lower() != "video":
        raise MediaRefError("not_a_video", "The referenced media is not a video")
    container = get_container()
    data = await asyncio.to_thread(_frame_png, container, ref["kind"], resolved["value"], seconds)
    stem = Path(resolved["original_filename"] or "video").stem or "video"
    label = "last" if seconds is None else f"{seconds:g}s"
    return await _store(container, data, f"{stem}-frame-{label}.png", "image/png", user_id, purpose)


async def save_user_media(user, path: Union[str, Path], purpose: str = UPLOAD_PURPOSE_USER) -> ResolvedMedia:
    """Store a file the plugin wrote (an absolute path) as the user's upload.

    Raises `MediaRefError` with `invalid_ref` (relative path), `not_found`,
    `unsupported_media`, `invalid_purpose` or `upload_rejected`."""
    user_id = _owner_id(user)
    purpose = _checked_purpose(purpose)
    source = Path(path)
    if not source.is_absolute():
        raise MediaRefError("invalid_ref", "The file path must be absolute")
    if not source.is_file():
        raise MediaRefError("not_found", "The file does not exist")
    container = get_container()
    media_types = container.media_store.media_types
    content_type = media_types.get_media_type(source.suffix)
    if not media_types.is_valid_media_type(content_type):
        raise MediaRefError("unsupported_media", "Only image, video, audio and 3D files can be stored")
    data = await asyncio.to_thread(source.read_bytes)
    return await _store(container, data, source.name, content_type, user_id, purpose)
