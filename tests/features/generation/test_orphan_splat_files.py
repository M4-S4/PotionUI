from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from src.features.generation.file_repository import FileRepository
from src.features.generation.history_query import GenerationHistoryQuery
from src.features.generation.records import File, Generation
from src.features.generation.repository import GenerationRepository
from src.features.library.operations.mutations import copy_generation_file
from src.features.media.file_resolver import FilePathResolver
from src.features.media.image_processor import ImageProcessor
from src.features.media.media_types import MediaTypeResolver
from src.features.media.store import MediaStore
from src.features.media.upload_repository import UploadRepository
from src.features.media_index.indexer import PASS_CLIP_EMBED, PASS_TAGS, MediaIndexer
from src.features.media_index.repository import MediaIndexRepository
from src.features.media_index.tagger import TaggingResult
from src.features.plans.repository import UsageRepository
from src.platform.database.rows import now_utc
from src.platform.filesystem.file_store import FileStore
from src.platform.plugins import runtime_registries
from src.platform.security.user import AccountType, User
from src.platform.settings.settings import Settings
from src.plugin_api import generation as plugin_generation
from tests.features.media_index.test_indexer import (
    FakeGalleryPromptStore,
    FakeGalleryStore,
    FakeTextEmbedder,
    FakeVisionEmbedder,
)
from tests.fixtures.seed import seed_user

GENERATION_DIR = "generations/2026-10-22/gen-splat"
PLY_BYTES = (
    b"ply\nformat binary_little_endian 1.0\ncomment potionui_view fov_y=50 aspect=1.5 depth=4 "
    b"width=1536 height=1024 convention=opencv\nelement vertex 1\nproperty float x\nend_header\n"
    + b"\x00" * 4
)


class _NoopHookContext:
    data: dict = {}


class _NoopPluginRegistry:

    def execute_hook(self, hook, initial_data=None):
        context = _NoopHookContext()
        context.data = dict(initial_data or {})
        return context, []


class _OpeningTagger:
    provenance = "model-a"

    def __init__(self):
        self.calls = []

    def tag_image_file(self, path):
        self.calls.append(path)
        with Image.open(path):
            pass
        return TaggingResult(tags=[], ratings={})


@pytest.fixture
def orphan(mock_db, test_storage):
    seed_user(mock_db, "owner")
    GenerationRepository().create(Generation(
        id="gen-splat", preset_id="01KX47GSP4TQ0000000000SP3D", form_data={}, user_id="owner",
        status="completed", mode="img2splat",
    ))
    folder = Path(test_storage) / GENERATION_DIR
    (folder / "thumbnails").mkdir(parents=True)
    (folder / "1.ply").write_bytes(PLY_BYTES)
    thumbnails = {}
    for size, width in (("small", 64), ("medium", 128), ("large", 256)):
        name = f"thumbnails/1_{size}.webp"
        Image.new("RGB", (width, width * 2 // 3), (40, 90, 160)).save(folder / name, "WEBP")
        thumbnails[f"thumbnail_{size}"] = name
    row = FileRepository().create(File(
        file_path=f"{GENERATION_DIR}/1.ply", file_type="SPLAT", user_id="owner", mime_type="model/x-ply",
        file_size=len(PLY_BYTES), pipe_name=None, is_final=True, is_derived=False, **thumbnails,
    ))
    FileRepository().associate_with_generation("gen-splat", row.id)
    return SimpleNamespace(db=mock_db, storage=Path(test_storage), file_id=row.id)


@pytest.fixture
def history():
    return GenerationHistoryQuery(GenerationRepository())


@pytest.fixture
def media_store(test_storage):
    settings = Mock(spec=Settings)
    settings.get_file_storage_directory.return_value = str(test_storage)
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


def _owner():
    return User(
        id="owner", username="owner", email="owner@example.test", password_hash="$2b$12$x",
        account_type=AccountType.USER, created_at=now_utc(), last_login=None,
    )


def test_history_still_lists_the_generation_with_its_splat_row(orphan, history):
    result = history.get_history(user_id="owner", include_tags=False)

    [generation] = result["generations"]
    assert generation["id"] == "gen-splat"
    assert [(f["id"], f["file_type"], f["file_path"]) for f in generation["files"]] == [
        (orphan.file_id, "SPLAT", f"{GENERATION_DIR}/1.ply"),
    ]


@pytest.mark.parametrize("media_type", ["image", "video", "audio", "mesh"])
def test_history_media_filters_leave_the_splat_generation_out(orphan, history, media_type):
    result = history.get_history(user_id="owner", include_tags=False, media_type=media_type)

    assert result["generations"] == []


def test_generation_detail_reads_the_splat_row(orphan, history):
    detail = history.get_by_id("gen-splat", "owner")

    assert [f["file_type"] for f in detail["files"]] == ["SPLAT"]


def test_the_ply_is_served_as_a_plain_download_and_its_preview_as_an_image(orphan, media_store):
    original = media_store.get_generation_media("gen-splat", "1.ply")
    thumbnail = media_store.get_generation_media("gen-splat", "1.ply", size="medium")

    assert original.media_type == "application/octet-stream"
    assert Path(original.file_path).read_bytes() == PLY_BYTES
    assert thumbnail.media_type == "image/webp"
    assert Path(thumbnail.file_path) == orphan.storage / GENERATION_DIR / "thumbnails" / "1_medium.webp"


def test_the_library_refuses_the_splat_row(orphan):
    collaborators = SimpleNamespace(file_repository=FileRepository())

    with pytest.raises(ValueError, match="Cannot add a SPLAT file to the library"):
        copy_generation_file(collaborators, orphan.file_id, "owner")


def test_plugins_still_see_the_splat_row(orphan, history, monkeypatch):
    container = SimpleNamespace(
        generation_orchestrator=SimpleNamespace(status_tracker=SimpleNamespace(get=lambda gid: None)),
        generation_repository=GenerationRepository(),
        generation_history_facade=SimpleNamespace(query=history),
    )
    monkeypatch.setattr(runtime_registries, "_container", container)

    files = plugin_generation.list_generation_files(_owner(), "gen-splat")

    assert [(f["file_type"], f["index"]) for f in files] == [("SPLAT", 0)]


def test_storage_usage_still_counts_the_splat_bytes(orphan):
    usage = UsageRepository()

    assert usage.storage_bytes("owner") == len(PLY_BYTES)
    assert sum(group["bytes"] for group in usage.storage_breakdown("owner")) == len(PLY_BYTES)


@pytest.mark.parametrize("pass_type", [PASS_TAGS, PASS_CLIP_EMBED])
def test_the_media_index_skips_the_splat_row_without_failing(orphan, pass_type):
    repository = MediaIndexRepository()
    repository.enqueue_files([orphan.file_id], pass_type)
    tagger = _OpeningTagger()
    embedder = FakeVisionEmbedder()
    indexer = MediaIndexer(
        repository=repository,
        tagger_provider=tagger,
        file_service=FileStore(str(orphan.storage)),
        vision_embedder=embedder,
        gallery_vector_store=FakeGalleryStore(),
        text_embedding_provider=FakeTextEmbedder(),
        gallery_prompt_vector_store=FakeGalleryPromptStore(),
    )

    assert indexer.process_pending(pass_type, batch_size=10) == {"processed": 1, "failed": 0}
    assert repository.queue_counts(pass_type)[pass_type] == {"done": 1}
    assert embedder.image_calls == []
