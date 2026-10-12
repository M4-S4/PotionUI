from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.features.generation.records import File
from src.features.media.file_resolver import FilePathResolver
from src.features.media.image_processor import ImageProcessor
from src.features.media.media_types import MediaTypeResolver
from src.features.media.store import MediaStore
from src.platform.filesystem.storage_driver import LocalFileStorageDriver


class _NoopPlugins:
    def execute_hook(self, hook, initial_data=None):
        return SimpleNamespace(data=dict(initial_data or {})), []


@pytest.fixture
def served(tmp_path):
    storage = tmp_path / "storage"
    key = "outputs/2026-10-12/orig/4.png"
    (storage / "outputs/2026-10-12/orig").mkdir(parents=True)
    (storage / key).write_bytes(b"png")
    generations = Mock()
    generations.get_by_id.return_value = SimpleNamespace(user_id="u1")
    generations.get_files.return_value = []
    generations.get_file_in_generation_directory.return_value = File(
        file_path=key, file_type="IMAGE", user_id="u1", file_size=3, is_final=True
    )
    store = MediaStore(
        file_resolver=Mock(spec=FilePathResolver),
        image_processor=ImageProcessor(),
        media_type_resolver=MediaTypeResolver(),
        file_repository=Mock(),
        generation_repository=generations,
        settings=Mock(),
        file_service=Mock(),
        plugin_registry=_NoopPlugins(),
        upload_repository=Mock(),
        storage_driver=LocalFileStorageDriver(str(storage)),
    )
    return store, generations


def test_a_file_moved_to_another_generation_still_serves_under_the_original_id(served):
    store, generations = served

    result = store.get_generation_media("orig", "4.png")

    assert result.headers["Content-Length"] == "3"
    generations.get_file_in_generation_directory.assert_called_once_with("u1", "orig", "4.png")


def test_a_file_nobody_holds_is_still_not_found(served):
    store, generations = served
    generations.get_file_in_generation_directory.return_value = None

    with pytest.raises(ValueError):
        store.get_generation_media("orig", "4.png")


def test_a_file_the_generation_owns_never_needs_the_fallback(served):
    store, generations = served
    generations.get_files.return_value = [
        File(file_path="outputs/2026-10-12/orig/4.png", file_type="IMAGE", user_id="u1", file_size=3, is_final=True)
    ]

    store.get_generation_media("orig", "4.png")

    generations.get_file_in_generation_directory.assert_not_called()
