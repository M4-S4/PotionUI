from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.features.media_index.vision_embedder import SiglipVisionEmbedder
from src.plugin_api import generation, media


class Resolver:
    def __init__(self, root):
        self.root = root

    def asset_dir(self, name):
        return self.root / name


class Embedder(SiglipVisionEmbedder):
    def __init__(self, resolver):
        super().__init__(resolver=resolver, model_name="google/siglip-base-patch16-224", device="cpu", auto_download=True)
        self.calls = []

    def embed_images(self, images):
        self.calls.append(images)
        return [[1.0, 0.0] for _ in images]

    def is_loaded(self):
        return False


class Queue:
    def __init__(self, pending=(), running=None):
        self.pending = list(pending)
        self.running = running or {}

    def snapshot(self):
        return {"pending": self.pending, "running": self.running}


def _world(embedder=None, queue=None):
    return SimpleNamespace(media_indexer=SimpleNamespace(vision_embedder=embedder), generation_orchestrator=SimpleNamespace(queue=queue))


@pytest.fixture
def embedder(tmp_path):
    return Embedder(Resolver(tmp_path))


def _weights(embedder):
    folder = embedder.local_dir_for(embedder.model_name, embedder.resolver)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")


def test_the_image_embedder_says_which_model_it_is_and_whether_its_weights_are_here(embedder):
    with patch("src.plugin_api.media.get_container", return_value=_world(embedder)):
        before = media.image_embedder()
        _weights(embedder)
        after = media.image_embedder()

    assert before == {"model": "google/siglip-base-patch16-224", "slug": "local-google-siglip-base-patch16-224", "device": "cpu", "available": False, "loaded": False}
    assert after["available"] is True


def test_embedding_images_goes_through_the_application_embedder_once_its_weights_are_here(embedder):
    _weights(embedder)
    with patch("src.plugin_api.media.get_container", return_value=_world(embedder)):
        vectors = media.embed_images(("a.png", "b.png"))

    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.calls == [["a.png", "b.png"]]


def test_embedding_never_downloads_weights_that_are_not_here(embedder):
    with patch("src.plugin_api.media.get_container", return_value=_world(embedder)):
        with pytest.raises(FileNotFoundError):
            media.embed_images(["a.png"])

    assert embedder.calls == []


@pytest.mark.parametrize(
    ("queue", "active"),
    [(Queue(), False), (Queue(running={"b-local": []}), False), (Queue(pending=[{"generation_id": "g-1"}]), True), (Queue(running={"b-local": ["g-2"]}), True)],
)
def test_a_generation_is_active_while_the_queue_holds_one_pending_or_running(queue, active):
    with patch("src.plugin_api.generation.get_container", return_value=_world(queue=queue)):
        assert generation.generation_active() is active


def test_the_new_names_are_listed_by_their_modules():
    assert {"embed_images", "image_embedder"} <= set(media.__all__)
    assert "generation_active" in generation.__all__
