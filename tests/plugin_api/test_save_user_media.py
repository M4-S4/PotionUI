from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

import src.plugin_api as plugin_api
from src.features.media.file_resolver import FilePathResolver
from src.features.media.hooks import MEDIA_HOOKS
from src.features.media.image_processor import ImageProcessor
from src.features.media.media_types import MediaTypeResolver
from src.features.media.store import MediaStore
from src.features.media.upload_repository import UploadRepository
from src.platform.filesystem.storage_driver import LocalFileStorageDriver
from src.platform.plugins import PluginRegistry
from src.platform.settings.settings import Settings
from src.plugin_api.generation import MediaRefError
from src.plugin_api.media import save_user_media


def plugin_registry(blocked_hook=None):
    def execute_hook(hook, initial_data=None):
        if hook == blocked_hook:
            return SimpleNamespace(data={**initial_data, "blocked": True, "block_reason": "Blocked by policy"}), []
        return SimpleNamespace(data=dict(initial_data or {})), []

    registry = Mock(spec=PluginRegistry)
    registry.execute_hook.side_effect = execute_hook
    return registry


@pytest.fixture
def storage(tmp_path):
    return tmp_path / "storage"


def media_store(storage, registry):
    return MediaStore(
        file_resolver=Mock(spec=FilePathResolver),
        image_processor=ImageProcessor(),
        media_type_resolver=MediaTypeResolver(),
        file_repository=Mock(),
        generation_repository=Mock(),
        settings=Mock(spec=Settings),
        file_service=Mock(),
        plugin_registry=registry,
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
def container(mock_db, storage):
    world = SimpleNamespace(media_store=media_store(storage, plugin_registry()))
    with patch("src.plugin_api.media.get_container", return_value=world):
        yield world


@pytest.fixture
def export(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    path = work / "board.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


def purposes(mock_db, user_id):
    with mock_db.get_cursor() as cursor:
        cursor.execute("SELECT purpose, original_filename FROM uploads WHERE user_id = ?", (user_id,))
        return [(row["purpose"], row["original_filename"]) for row in cursor.fetchall()]


@pytest.mark.asyncio
async def test_a_plugin_file_becomes_a_library_upload_of_the_user(mock_db, container, storage, user, export):
    resolved = await save_user_media(user, export)

    assert resolved["value"].startswith("uploads/") and resolved["value"].endswith(".png")
    assert (resolved["media_type"], resolved["original_filename"], resolved["origin"]) == ("image", "board.png", None)
    assert (storage / resolved["value"]).read_bytes() == export.read_bytes()
    assert purposes(mock_db, user.id) == [("user_upload", "board.png")]


@pytest.mark.asyncio
async def test_a_derived_artifact_stays_out_of_the_library(mock_db, container, user, export):
    first = await save_user_media(user, str(export), purpose="derived_artifact")
    again = await save_user_media(user, str(export), purpose="derived_artifact")

    assert again["value"] == first["value"]
    assert purposes(mock_db, user.id) == [("derived_artifact", "board.png")]


@pytest.mark.asyncio
async def test_unusable_files_are_refused_with_a_code(container, user, export, tmp_path):
    notes = tmp_path / "work" / "notes.txt"
    notes.write_text("not media")
    cases = [
        ("work/board.png", {}, "invalid_ref"),
        (tmp_path / "work" / "gone.png", {}, "not_found"),
        (tmp_path / "work", {}, "not_found"),
        (notes, {}, "unsupported_media"),
        (export, {"purpose": "plugin"}, "invalid_purpose"),
    ]
    for path, options, code in cases:
        with pytest.raises(MediaRefError) as refused:
            await save_user_media(user, path, **options)
        assert refused.value.code == code


@pytest.mark.asyncio
async def test_an_upload_a_plugin_hook_blocks_is_rejected(mock_db, storage, user, export):
    world = SimpleNamespace(media_store=media_store(storage, plugin_registry(MEDIA_HOOKS.before_upload)))

    with patch("src.plugin_api.media.get_container", return_value=world):
        with pytest.raises(MediaRefError) as refused:
            await save_user_media(user, export)

    assert refused.value.code == "upload_rejected"
    assert str(refused.value) == "Blocked by policy"
    assert purposes(mock_db, user.id) == []


@pytest.mark.asyncio
async def test_a_user_is_required(container, export):
    with pytest.raises(PermissionError):
        await save_user_media(SimpleNamespace(id=None), export)


def test_exported_from_the_plugin_api_root():
    assert plugin_api.save_user_media is save_user_media
