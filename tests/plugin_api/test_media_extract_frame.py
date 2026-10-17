import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

import src.plugin_api as plugin_api
from src.features.media.file_resolver import FilePathResolver
from src.features.media.image_processor import ImageProcessor
from src.features.media.media_types import MediaTypeResolver
from src.features.media.records import Upload
from src.features.media.store import MediaStore
from src.features.media.upload_repository import UploadRepository
from src.platform.filesystem.storage_driver import LocalFileStorageDriver
from src.platform.plugins import PluginRegistry
from src.platform.settings.settings import Settings
from src.platform.util.video_transcode import VideoProbe
from src.plugin_api.generation import MediaRefError
from src.plugin_api.media import extract_frame

FRAMES = 10
FPS = 10.0


def red_of(frame_index):
    return frame_index * 25


def write_clip(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (32, 32))
    for index in range(FRAMES):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 2] = red_of(index)
        writer.write(frame)
    writer.release()


def plugin_registry():
    registry = Mock(spec=PluginRegistry)
    registry.execute_hook.return_value = (SimpleNamespace(data={}), [])
    return registry


@pytest.fixture
def storage(tmp_path):
    root = tmp_path / "storage"
    (root / "uploads").mkdir(parents=True)
    return root


@pytest.fixture
def store(mock_db, storage):
    return MediaStore(
        file_resolver=Mock(spec=FilePathResolver),
        image_processor=ImageProcessor(),
        media_type_resolver=MediaTypeResolver(),
        file_repository=Mock(),
        generation_repository=Mock(),
        settings=Mock(spec=Settings),
        file_service=Mock(),
        plugin_registry=plugin_registry(),
        upload_repository=UploadRepository(),
        storage_driver=LocalFileStorageDriver(str(storage)),
    )


@pytest.fixture
def user(mock_db):
    with mock_db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES ('u1', 'u1', 'u1@example.test', 'x')"
        )
    return SimpleNamespace(id="u1")


@pytest.fixture
def library_clip(store, storage, user):
    write_clip(storage / "uploads" / "clip-1.avi")
    upload = store.upload_repo.create(Upload(user_id=user.id, filename="clip-1.avi", media_type="video", original_filename="clip.avi"))
    return {"kind": "library", "item_id": upload.id}


@pytest.fixture
def library_image(store, storage, user):
    Image.new("RGB", (4, 4)).save(storage / "uploads" / "still.png")
    upload = store.upload_repo.create(Upload(user_id=user.id, filename="still.png", media_type="image", original_filename="still.png"))
    return {"kind": "library", "item_id": upload.id}


@pytest.fixture
def container(store):
    world = SimpleNamespace(media_store=store, file_service=SimpleNamespace())
    probe = VideoProbe(duration=FRAMES / FPS, width=32, height=32, fps=FPS, video_codec="mjpeg", has_audio=False, size_bytes=1)
    with patch("src.plugin_api.media.get_container", return_value=world), \
         patch("src.plugin_api.generation.get_container", return_value=world), \
         patch("src.plugin_api.media.probe_video", return_value=probe) as probed:
        world.probe = probed
        yield world


def stored_red(storage, resolved):
    with Image.open(storage / resolved["value"]) as image:
        return image.convert("RGB").getpixel((16, 16))[0]


@pytest.mark.asyncio
async def test_the_last_frame_is_stored_as_a_png_upload_of_the_user(container, storage, user, library_clip):
    resolved = await extract_frame(user, library_clip)

    assert resolved["value"].startswith("uploads/") and resolved["value"].endswith(".png")
    assert (resolved["media_type"], resolved["original_filename"], resolved["origin"]) == ("image", "clip-frame-last.png", None)
    assert abs(stored_red(storage, resolved) - red_of(FRAMES - 1)) <= 8
    assert container.media_store.upload_repo.count_for_user(user.id, media_type="image") == 1
    container.probe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("at, frame_index, label", [(0, 0, "0s"), (0.45, 4, "0.45s"), (1.0, FRAMES - 1, "1s")])
async def test_seconds_pick_the_frame_shown_at_that_time(container, storage, user, library_clip, at, frame_index, label):
    resolved = await extract_frame(user, library_clip, at=at)

    assert resolved["original_filename"] == f"clip-frame-{label}.png"
    assert abs(stored_red(storage, resolved) - red_of(frame_index)) <= 8


@pytest.mark.asyncio
async def test_a_history_video_is_read_through_the_generation_store(container, storage, user, tmp_path):
    clip = tmp_path / "out.avi"
    write_clip(clip)
    opened = []

    @contextmanager
    def local_copy_of(key, suffix=""):
        opened.append((key, suffix))
        yield clip

    container.file_service.local_copy_of = local_copy_of
    history = {"value": "generations/2026-10-17/g1/out.avi", "media_type": "video", "original_filename": "out.avi", "origin": {"generation_id": "g1", "file_index": 0}}
    ref = {"kind": "history", "generation_id": "g1", "file_id": "f1"}

    with patch("src.plugin_api.media.resolve_media_ref", return_value=history) as resolve:
        resolved = await extract_frame(user, ref, at=0.2)

    resolve.assert_called_once_with(user, ref)
    assert opened == [("generations/2026-10-17/g1/out.avi", ".avi")]
    assert resolved["original_filename"] == "out-frame-0.2s.png"
    assert abs(stored_red(storage, resolved) - red_of(2)) <= 8


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["first", -1, True, math.nan, None])
async def test_an_unreadable_at_is_an_invalid_ref(container, user, library_clip, at):
    with pytest.raises(MediaRefError) as refused:
        await extract_frame(user, library_clip, at=at)

    assert refused.value.code == "invalid_ref"


@pytest.mark.asyncio
async def test_refusals_carry_their_codes(container, user, library_clip, library_image):
    cases = [
        ({"kind": "library", "item_id": "missing"}, "last", "not_found"),
        ({"kind": "library", "item_id": "x", "path": "/etc/passwd"}, "last", "invalid_ref"),
        (library_image, "last", "not_a_video"),
        (library_clip, 2.5, "extract_failed"),
    ]
    for ref, at, code in cases:
        with pytest.raises(MediaRefError) as refused:
            await extract_frame(user, ref, at=at)
        assert refused.value.code == code


@pytest.mark.asyncio
async def test_a_video_whose_frame_rate_cannot_be_read_fails_for_seconds_only(container, user, library_clip):
    container.probe.return_value = None

    with pytest.raises(MediaRefError) as refused:
        await extract_frame(user, library_clip, at=0.5)
    resolved = await extract_frame(user, library_clip)

    assert refused.value.code == "extract_failed"
    assert resolved["media_type"] == "image"


@pytest.mark.asyncio
async def test_an_unknown_purpose_and_a_missing_user_are_refused(container, user, library_clip):
    with pytest.raises(MediaRefError) as refused:
        await extract_frame(user, library_clip, purpose="plugin")
    with pytest.raises(PermissionError):
        await extract_frame(None, library_clip)

    assert refused.value.code == "invalid_purpose"


def test_exported_from_the_plugin_api_root():
    assert plugin_api.extract_frame is extract_frame
