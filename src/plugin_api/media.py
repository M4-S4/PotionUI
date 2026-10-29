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

`embed_images(images)` turns images (paths or PIL images) into the vectors of
the application's image embedder, the SigLIP model the gallery search uses,
L2-normalized so that the dot product of two vectors is their cosine. The model
goes through `ModelLifecycle` for each call, like any other model, and is never
downloaded: without its weights on this server it raises `FileNotFoundError`.
`image_embedder()` tells which model it is (`model`, `slug`, `device`) and
whether its weights are here (`available`) and it is loaded now (`loaded`).

`detect_faces(images)` finds the faces in images (paths or PIL images) with
OpenCV's YuNet detector, on the CPU: for each image, its boxes (`x`, `y`,
`width`, `height` in pixels of the image, `score`, and `landmarks`, the five
points right eye, left eye, nose tip, right and left corners of the mouth, as
the face sees them), best first. Each image is read at its own size, so that
the small face of a wide shot is found too. The model file,
`face_detection_yunet_2023mar.onnx`, is read from a `detection_bbox` folder of
the model roots and is never downloaded: without it `detect_faces` raises
`FileNotFoundError`. `face_detector()` tells which file it is (`model`) and
whether it is here (`available`).

`identify_faces(faces)` gives the identity vector of faces with AuraFace
(`fal/AuraFace-v1`, an ArcFace face recognition model under the Apache-2.0
license), on the CPU: each face is an image (path or PIL image) and the
`landmarks` `detect_faces` found for it, the face aligned on the ArcFace
template before it is read. The vectors are L2-normalized: two faces of one
person have a high cosine, two people a cosine near 0. The model file,
`glintr100.onnx`, is read from `face_recognition/auraface` in the home model
root, or from the file the environment variable `POTIONUI_FACE_IDENTITY_MODEL`
names, and is never downloaded: without it `identify_faces` raises
`FileNotFoundError`. `face_identifier()` tells which file it is (`model`), where
it is looked for (`path`) and whether it is here with onnxruntime to run it
(`available`).
"""

import asyncio
import io
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Sequence, Union

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
    from PIL import Image

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
    "detect_faces",
    "embed_images",
    "extract_frame",
    "face_detector",
    "face_identifier",
    "find_ffmpeg",
    "has_audio_stream",
    "identify_faces",
    "image_embedder",
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


def _embedder():
    return get_container().media_indexer.vision_embedder


def _weights_here(embedder) -> bool:
    return bool(type(embedder).resolve_status(embedder.model_name, embedder.resolver)["present"])


def image_embedder() -> Dict[str, Any]:
    """The application's image embedder: `model`, `slug` (the namespace of its vectors), `device`, `available` (its weights are on this server) and `loaded` (it is resident now)."""
    embedder = _embedder()
    return {
        "model": embedder.model_name,
        "slug": embedder.embedder_slug,
        "device": str(embedder.device),
        "available": _weights_here(embedder),
        "loaded": bool(embedder.is_loaded()),
    }


def embed_images(images: Sequence[Union[str, Path, "Image.Image"]]) -> List[List[float]]:
    """The L2-normalized vectors of images (paths or PIL images) from the application's image embedder, through ModelLifecycle; `FileNotFoundError` when its weights are not on this server, which is never downloaded from here."""
    embedder = _embedder()
    if not _weights_here(embedder):
        raise FileNotFoundError(f"The weights of {embedder.model_name} are not on this server")
    return embedder.embed_images(list(images))


FACE_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_MODEL_TYPE = "detection_bbox"


def _face_model() -> Optional[Path]:
    for entry in get_container().model_roots.type_dirs(FACE_MODEL_TYPE):
        candidate = Path(entry.path) / FACE_MODEL
        if candidate.is_file():
            return candidate
    return None


def face_detector() -> Dict[str, Any]:
    """The face detector the plugins can use: `model`, its file name, and `available`, whether that file is in a detection_bbox folder of the model roots."""
    return {"model": FACE_MODEL, "available": _face_model() is not None}


def detect_faces(images: Sequence[Union[str, Path, "Image.Image"]], min_score: float = 0.5) -> List[List[Dict[str, float]]]:
    """The faces YuNet finds in each image (paths or PIL images), on the CPU: a list of boxes `x`, `y`, `width`, `height` in pixels of the image, `score` and the five `landmarks`, best first, each image read at its own size; `FileNotFoundError` when the model file is not in a detection_bbox folder of the model roots, which is never downloaded from here."""
    from src.platform.util.face_boxes import detect

    model = _face_model()
    if model is None:
        raise FileNotFoundError(f"{FACE_MODEL} is not in a {FACE_MODEL_TYPE} folder of the model roots")
    return detect(model, list(images), min_score)


IDENTITY_MODEL = "glintr100.onnx"
IDENTITY_DIR = "face_recognition/auraface"
IDENTITY_ENV = "POTIONUI_FACE_IDENTITY_MODEL"


def _identity_path() -> Path:
    configured = os.environ.get(IDENTITY_ENV, "").strip()
    return Path(configured) if configured else get_container().model_roots.asset_dir(IDENTITY_DIR) / IDENTITY_MODEL


def face_identifier() -> Dict[str, Any]:
    """The face recognition model the plugins can use: `model`, its file name, `path`, where it is looked for (the file `POTIONUI_FACE_IDENTITY_MODEL` names, else face_recognition/auraface in the home model root), and `available`, whether that file is there and onnxruntime imports to run it."""
    from src.platform.util.face_identity import runtime_imports

    path = _identity_path()
    return {"model": IDENTITY_MODEL, "path": str(path), "available": path.is_file() and runtime_imports()}


def identify_faces(faces: Sequence[Any]) -> List[List[float]]:
    """The L2-normalized AuraFace identity vector of each face, given as an image (path or PIL image) and the five `landmarks` detect_faces found for that face, on the CPU; `FileNotFoundError` when the model file is not where face_identifier says, which is never downloaded from here."""
    from src.platform.util.face_identity import embed

    path = _identity_path()
    if not path.is_file():
        raise FileNotFoundError(f"{IDENTITY_MODEL} is not at {path}")
    return embed(path, [(image, landmarks) for image, landmarks in faces])


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
