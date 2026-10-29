from types import SimpleNamespace

import pytest

from src.features.media.records import Upload
from src.features.media.upload_repository import UploadRepository
from src.platform.plugins import runtime_registries
from src.plugin_api import media
from tests.fixtures.seed import seed_user


class Driver:
    def __init__(self, root=None, stored=None, broken=False):
        self.root = root
        self.stored = stored or {}
        self.broken = broken
        self.chunks = []

    def local_path(self, key):
        return None if self.root is None else str(self.root / key)

    def exists(self, key):
        return key in self.stored

    def get_stream(self, key, chunk_size):
        self.chunks.append(chunk_size)
        data = self.stored[key]
        yield data[:2]
        if self.broken:
            raise OSError("connection reset")
        yield data[2:]


def _world(monkeypatch, history, library, uploads=None):
    container = SimpleNamespace(
        file_service=SimpleNamespace(storage_driver=history),
        media_store=SimpleNamespace(storage_driver=library, upload_repo=uploads),
    )
    monkeypatch.setattr(runtime_registries, "_container", container)


def test_a_media_key_on_this_server_has_its_absolute_path(tmp_path, monkeypatch):
    _world(monkeypatch, Driver(root=tmp_path / "outputs"), Driver(root=tmp_path / "library"))

    assert media.media_local_path("history", "generations/g1/a.png") == (tmp_path / "outputs" / "generations/g1/a.png").resolve()
    assert media.media_local_path("library", "uploads/u.png") == (tmp_path / "library" / "uploads/u.png").resolve()


def test_a_media_key_kept_elsewhere_has_no_local_path(monkeypatch):
    _world(monkeypatch, Driver(), Driver())

    assert media.media_local_path("history", "generations/g1/a.png") is None


def test_a_media_kind_is_history_or_library(monkeypatch):
    _world(monkeypatch, Driver(), Driver())

    with pytest.raises(ValueError):
        media.media_local_path("uploads", "uploads/u.png")


def test_a_stored_key_is_streamed_into_the_target(tmp_path, monkeypatch):
    library = Driver(stored={"uploads/u.png": b"pixels"})
    _world(monkeypatch, Driver(), library)
    target = tmp_path / "copy.png"

    assert media.copy_media_file("library", "uploads/u.png", target) is True
    assert target.read_bytes() == b"pixels"
    assert library.chunks == [1 << 20]


def test_a_key_that_is_not_stored_is_not_copied(tmp_path, monkeypatch):
    _world(monkeypatch, Driver(), Driver())
    target = tmp_path / "copy.png"

    assert media.copy_media_file("history", "generations/g1/a.png", target) is False
    assert not target.exists()


def test_a_failed_copy_leaves_no_file_behind(tmp_path, monkeypatch):
    _world(monkeypatch, Driver(stored={"generations/g1/a.png": b"pixels"}, broken=True), Driver())
    target = tmp_path / "copy.png"

    with pytest.raises(OSError):
        media.copy_media_file("history", "generations/g1/a.png", target)

    assert not target.exists()


def test_an_upload_key_of_the_user_is_their_library_item(mock_db, monkeypatch):
    seed_user(mock_db, "owner")
    seed_user(mock_db, "other")
    uploads = UploadRepository()
    uploads.create(Upload(
        id="up-1", user_id="owner", filename="u1.png", original_filename="cat.png",
        media_type="image", mime_type="image/png", file_size=1, width=640, height=480,
    ))
    _world(monkeypatch, Driver(), Driver(), uploads)

    assert media.find_upload("owner", "uploads/u1.png") == {"id": "up-1", "filename": "cat.png", "width": 640, "height": 480}
    assert media.find_upload("owner", "uploads\\u1.png")["id"] == "up-1"
    assert media.find_upload("other", "uploads/u1.png") is None
    assert media.find_upload("owner", "uploads/u2.png") is None


def test_the_storage_reads_are_part_of_the_media_api():
    assert {"copy_media_file", "find_upload", "media_local_path"} <= set(media.__all__)
