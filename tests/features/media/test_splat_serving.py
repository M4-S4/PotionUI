from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.features.generation.file_repository import FileRepository
from src.features.generation.handlers.splat_handler import SplatGenerationOutputHandler
from src.features.generation.repository import GenerationRepository
from src.features.media.file_resolver import FilePathResolver
from src.features.media.image_processor import ImageProcessor
from src.features.media.media_types import MediaTypeResolver
from src.features.media.store import MediaStore
from src.features.media.upload_repository import UploadRepository
from src.pipelines.outputs import SplatGenerationOutput
from src.platform.filesystem.file_store import FileStore
from src.platform.settings.settings import Settings
from src.platform.util.ids import generate_ulid


class _NoopHookContext:
    data: dict = {}


class _NoopPluginRegistry:

    def execute_hook(self, hook, initial_data=None):
        context = _NoopHookContext()
        context.data = dict(initial_data or {})
        return context, []


@pytest.fixture
def repos_on_test_db(mock_db):
    with patch('src.platform.database.database.db', mock_db):
        yield mock_db


@pytest.fixture
def settings(test_storage):
    settings = Mock(spec=Settings)
    settings.get_file_storage_directory.return_value = str(test_storage)
    return settings


@pytest.fixture
def media_store(settings, test_storage):
    return MediaStore(
        file_resolver=FilePathResolver(settings),
        image_processor=ImageProcessor(),
        media_type_resolver=MediaTypeResolver(),
        file_repository=FileRepository(),
        generation_repository=GenerationRepository(),
        settings=settings,
        file_service=FileStore(str(test_storage)),
        plugin_registry=_NoopPluginRegistry(),
        upload_repository=UploadRepository(),
    )


def _saved_splat(mock_db, settings, path, preview=None):
    user_id = generate_ulid()
    gen_id = generate_ulid()
    with mock_db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, 'splat_server_tester', 'splat-serve@example.com', 'hashed'),
        )
        cursor.execute(
            """
            INSERT INTO generations (id, preset_id, preset_version, form_data, user_id, status, progress)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (gen_id, 'workbench/splat/test', '1.0.0', '{}', user_id, 'completed', 1.0),
        )
    metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(
        SplatGenerationOutput(splat_path=path, temporary=False, preview=preview)
    )
    assert metadata['processed'] is True
    return gen_id, Path(metadata['saved_path']).name


class TestSplatServing:

    def test_final_ply_is_served_with_the_splat_media_type(
        self, repos_on_test_db, media_store, settings, gaussian_ply_file, gaussian_ply_bytes
    ):
        gen_id, filename = _saved_splat(repos_on_test_db, settings, gaussian_ply_file)

        result = media_store.get_generation_media(gen_id, filename)

        assert result.media_type == 'model/x-ply'
        assert Path(result.file_path).read_bytes() == gaussian_ply_bytes

    def test_final_spz_is_served_with_its_media_type(self, repos_on_test_db, media_store, settings, spz_file):
        gen_id, filename = _saved_splat(repos_on_test_db, settings, spz_file)

        assert media_store.get_generation_media(gen_id, filename).media_type == 'model/x-spz'

    def test_the_pipe_preview_is_served_as_the_thumbnail(
        self, repos_on_test_db, media_store, settings, gaussian_ply_file
    ):
        preview = Image.new('RGB', (64, 48), (200, 30, 30))
        gen_id, filename = _saved_splat(repos_on_test_db, settings, gaussian_ply_file, preview)

        result = media_store.get_generation_media(gen_id, filename, size='medium')

        assert result.media_type == 'image/webp'
        assert Path(result.file_path).exists()

    def test_without_a_preview_there_is_no_thumbnail(
        self, repos_on_test_db, media_store, settings, gaussian_ply_file
    ):
        gen_id, filename = _saved_splat(repos_on_test_db, settings, gaussian_ply_file)

        with pytest.raises(ValueError) as excinfo:
            media_store.get_generation_media(gen_id, filename, size='small')

        assert 'thumbnail' in str(excinfo.value).lower()

    def test_temporary_splat_is_served_from_the_tmp_route(
        self, repos_on_test_db, media_store, settings, gaussian_ply_file
    ):
        metadata = SplatGenerationOutputHandler(generate_ulid(), None, settings).handle(
            SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=True)
        )

        result = media_store.get_temp_media(Path(metadata['saved_path']).name)

        assert result.media_type == 'model/x-ply'
