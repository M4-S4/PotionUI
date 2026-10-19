import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from src.features.generation.handlers.gallery_handler import (
    GalleryGenerationOutputHandler,
    serialize_gallery_output,
)
from src.features.generation.handlers.splat_handler import (
    SplatGenerationOutputHandler,
    serialize_splat_output,
)
from src.features.generation.output_serializer import GenerationOutputSerializer
from src.features.generation.output_types import SerializeContext, output_type_registry
from src.features.generation.repository import generation_repo
from src.pipelines.outputs import GalleryGenerationOutput, SplatGenerationOutput
from src.platform.settings.settings import Settings
from src.platform.util.ids import generate_ulid
from tests.fixtures.mesh_fixtures import build_minimal_glb
from tests.fixtures.splat_fixtures import (
    GAUSSIAN_PROPERTIES,
    SPLAT_COUNT,
    build_antimatter_splat,
    build_gaussian_ply,
    build_spz,
)


@pytest.fixture
def settings(test_storage):
    settings = Mock(spec=Settings)
    settings.get_file_storage_directory.return_value = str(test_storage)
    return settings


@pytest.fixture
def repos_on_test_db(mock_db):
    with patch('src.platform.database.database.db', mock_db):
        yield mock_db


@pytest.fixture
def generation_id(repos_on_test_db):
    mock_db = repos_on_test_db
    user_id = generate_ulid()
    gen_id = generate_ulid()
    with mock_db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
            (user_id, 'splat_tester', 'splat@example.com', 'hashed'),
        )
        cursor.execute(
            """
            INSERT INTO generations (id, preset_id, preset_version, form_data, user_id, status, progress)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (gen_id, 'workbench/splat/test', '1.0.0', '{}', user_id, 'running', 0.0),
        )
    return gen_id, user_id


def _preview():
    return Image.new('RGB', (96, 64), (40, 90, 160))


class TestSplatSavePath:

    def test_final_splat_is_written_and_recorded(
        self, generation_id, settings, test_storage, gaussian_ply_file, gaussian_ply_bytes
    ):
        gen_id, user_id = generation_id
        handler = SplatGenerationOutputHandler(gen_id, user_id, settings)

        metadata = handler.handle(SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False))

        assert metadata['processed'] is True
        on_disk = test_storage / metadata['saved_path']
        assert on_disk.suffix == '.ply'
        assert on_disk.read_bytes() == gaussian_ply_bytes

        files = generation_repo.get_files(gen_id, is_final=True)
        assert len(files) == 1
        assert files[0].file_type == 'SPLAT'
        assert files[0].file_path == metadata['saved_path']
        assert files[0].file_size == len(gaussian_ply_bytes)
        assert files[0].thumbnail_medium is None

    def test_splat_count_comes_from_the_file(self, generation_id, settings, gaussian_ply_file):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False)

        SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        assert output.splat_count == SPLAT_COUNT

    def test_count_set_by_the_pipe_is_kept(self, generation_id, settings, gaussian_ply_file):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, splat_count=99)

        SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        assert output.splat_count == 99

    def test_the_pipe_preview_becomes_the_file_thumbnails(
        self, generation_id, settings, test_storage, gaussian_ply_file
    ):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, preview=_preview())

        with patch('src.platform.runtime.native.mesh_preview.render_mesh_preview') as mesh_render:
            metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        mesh_render.assert_not_called()
        record = generation_repo.get_files(gen_id, is_final=True)[0]
        assert record.thumbnail_medium is not None
        assert record.thumbnail_profile is not None
        base = test_storage / Path(metadata['saved_path']).parent
        thumbnail = base / record.thumbnail_medium
        assert thumbnail.exists()
        with Image.open(thumbnail) as image:
            assert image.format == 'WEBP'

    def test_spz_is_accepted_from_a_pipe(self, generation_id, settings, spz_file):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=spz_file, temporary=False)

        metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        assert metadata['processed'] is True
        assert Path(metadata['saved_path']).suffix == '.spz'
        assert output.splat_count == SPLAT_COUNT

    def test_temporary_splat_goes_to_tmp_without_a_record(
        self, generation_id, settings, test_storage, gaussian_ply_file, gaussian_ply_bytes
    ):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=True, preview=_preview())

        metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        saved = Path(metadata['saved_path'])
        assert saved.parts[0] == 'tmp'
        assert (test_storage / saved).read_bytes() == gaussian_ply_bytes
        assert metadata['file_record'] is None
        assert generation_repo.get_files(gen_id) == []

    def test_two_final_splats_do_not_overwrite_each_other(
        self, generation_id, settings, test_storage, gaussian_ply_file
    ):
        gen_id, user_id = generation_id
        handler = SplatGenerationOutputHandler(gen_id, user_id, settings)

        first = handler.handle(SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, preview=_preview()))
        second = handler.handle(SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, preview=_preview()))

        assert first['saved_path'] != second['saved_path']
        records = generation_repo.get_files(gen_id, is_final=True)
        assert len({r.thumbnail_medium for r in records}) == 2


class TestSplatValidation:

    @pytest.mark.parametrize("name,payload,reason", [
        ("mesh.ply", build_gaussian_ply(properties=('x', 'y', 'z')), "a mesh ply"),
        ("short.ply", build_gaussian_ply(truncate=3), "truncated ply"),
        ("bad.spz", build_spz(magic=b'XXXX'), "bad spz magic"),
        ("scene.glb", build_minimal_glb(), "a mesh extension"),
        ("scene.splat", build_antimatter_splat(), "an import-only format"),
    ])
    def test_invalid_splat_never_reaches_storage(
        self, generation_id, settings, test_storage, tmp_path, name, payload, reason
    ):
        gen_id, user_id = generation_id
        path = tmp_path / name
        path.write_bytes(payload)

        metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(
            SplatGenerationOutput(splat_path=path, temporary=False)
        )

        assert metadata['processed'] is False, f"accepted {reason}"
        assert 'save_error' in metadata
        assert metadata['saved_path'] is None
        assert generation_repo.get_files(gen_id) == []
        assert not [p for p in (test_storage / 'generations').rglob('*') if p.is_file()]


class TestSplatOutputTypeRegistration:

    def test_spec_is_registered(self):
        output = SplatGenerationOutput(splat_path=Path('/tmp/x.ply'))
        spec = output_type_registry.spec_for(output)

        assert spec.key == 'splat'
        assert spec.resolve_message_type(output) == 'workbench_update'
        assert spec.handler_cls is SplatGenerationOutputHandler
        assert spec.serializer is serialize_splat_output


class TestSplatThroughOutputProcessor:

    @pytest.mark.asyncio
    async def test_processor_routes_a_splat_to_its_handler(
        self, generation_id, settings, test_storage, gaussian_ply_file, gaussian_ply_bytes
    ):
        from src.features.generation.output_processor import OutputProcessor

        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False)

        metadata = await OutputProcessor(settings).process_output(gen_id, output, user_id)

        assert metadata['handler'] == 'SplatGenerationOutputHandler'
        assert (test_storage / metadata['saved_path']).read_bytes() == gaussian_ply_bytes
        assert [f.file_type for f in generation_repo.get_files(gen_id, is_final=True)] == ['SPLAT']


class TestSplatSerializedEnvelope:

    def test_final_splat_envelope(self, generation_id, settings, gaussian_ply_file):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, seed=7, preview=_preview())
        output.pipe_id = 2
        output.pipe_name = 'generator_image_to_splat'
        metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        message = GenerationOutputSerializer(generation_id=gen_id).serialize_output(output)

        filename = Path(metadata['saved_path']).name
        assert message == {
            'type': 'workbench_update',
            'generation_id': gen_id,
            'pipe_id': 2,
            'pipe_name': 'generator_image_to_splat',
            'output_type': 'splat',
            'file_type': 'splat',
            'splat_format': 'ply',
            'temporary': False,
            'derived': False,
            'seed': 7,
            'splat_count': SPLAT_COUNT,
            'path': f'/api/media/generations/{gen_id}/{filename}',
            'splat_name': filename,
        }
        json.dumps(message)

    def test_temporary_splat_envelope_points_at_the_tmp_route(self, generation_id, settings, gaussian_ply_file):
        gen_id, user_id = generation_id
        output = SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=True)
        metadata = SplatGenerationOutputHandler(gen_id, user_id, settings).handle(output)

        message = GenerationOutputSerializer(generation_id=gen_id).serialize_output(output)

        assert message['path'] == f"/api/media/tmp/{Path(metadata['saved_path']).name}"

    def test_unsaved_splat_carries_no_path(self):
        message = GenerationOutputSerializer(generation_id='g').serialize_output(
            SplatGenerationOutput(splat_path=Path('/tmp/never.spz'))
        )

        assert message['file_type'] == 'splat'
        assert message['splat_format'] == 'spz'
        assert 'path' not in message


class TestSplatsInAGallery:

    def test_gallery_stores_and_serializes_its_splats(self, generation_id, settings, test_storage, gaussian_ply_file):
        gen_id, user_id = generation_id
        gallery = GalleryGenerationOutput(
            images=[],
            splats=[SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, preview=_preview())],
        )

        metadata = GalleryGenerationOutputHandler(gen_id, user_id, settings).handle(gallery)

        assert metadata['splat_count'] == 1
        assert metadata['processed_splats'][0]['processed'] is True
        assert [f.file_type for f in generation_repo.get_files(gen_id, is_final=True)] == ['SPLAT']

        payload = serialize_gallery_output(gallery, SerializeContext(generation_id=gen_id))
        assert payload['splats'][0]['file_type'] == 'splat'
        assert payload['splats'][0]['splat_format'] == 'ply'
        assert payload['splats'][0]['splat_count'] == SPLAT_COUNT
        assert payload['splat_urls_list'][0]['path'].startswith(f'/api/media/generations/{gen_id}/')
        assert payload['meshes'] == []

    def test_gallery_counter_keeps_images_and_splats_apart(self, generation_id, settings, gaussian_ply_file):
        from src.pipelines.outputs import ImageGenerationOutput

        gen_id, user_id = generation_id
        gallery = GalleryGenerationOutput(
            images=[ImageGenerationOutput(image=_preview(), temporary=False)],
            splats=[SplatGenerationOutput(splat_path=gaussian_ply_file, temporary=False, preview=_preview())],
        )

        GalleryGenerationOutputHandler(gen_id, user_id, settings).handle(gallery)

        records = generation_repo.get_files(gen_id, is_final=True)
        assert sorted(r.file_type for r in records) == ['IMAGE', 'SPLAT']
        assert len({r.thumbnail_medium for r in records}) == 2


def test_fixture_properties_match_the_contract():
    from src.platform.filesystem.splat_formats import GAUSSIAN_PLY_PROPERTIES

    assert GAUSSIAN_PROPERTIES == GAUSSIAN_PLY_PROPERTIES
