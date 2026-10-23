"""ChromaClientProvider: lazy shared construction, concurrency, and reset."""

import os
import subprocess
import sys
import threading

import pytest

from src.platform.vector.chroma_client import ChromaClientProvider, ChromaCollectionStore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_get_lazily_constructs_and_caches_one_client(monkeypatch, tmp_path):
    calls = []

    def fake_persistent_client(path):
        calls.append(path)
        return object()

    import chromadb

    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)

    provider = ChromaClientProvider(str(tmp_path))
    first = provider.get()
    second = provider.get()

    assert first is second
    assert calls == [str(tmp_path)]


def test_concurrent_first_use_constructs_exactly_one_client(monkeypatch, tmp_path):
    call_count = 0
    call_lock = threading.Lock()

    def fake_persistent_client(path):
        nonlocal call_count
        with call_lock:
            call_count += 1
        return object()

    import chromadb

    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)

    provider = ChromaClientProvider(str(tmp_path))
    thread_count = 8
    barrier = threading.Barrier(thread_count)
    results = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait(timeout=5)
        client = provider.get()
        with results_lock:
            results.append(client)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert call_count == 1
    assert len(results) == thread_count
    assert len({id(client) for client in results}) == 1


def test_failed_initialization_surfaces_error_and_retry_succeeds(monkeypatch, tmp_path):
    import chromadb

    attempts = []

    def flaky_persistent_client(path):
        attempts.append(path)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return object()

    monkeypatch.setattr(chromadb, "PersistentClient", flaky_persistent_client)

    provider = ChromaClientProvider(str(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        provider.get()

    client = provider.get()

    assert client is not None
    assert len(attempts) == 2


def test_close_releases_client_and_next_get_builds_a_fresh_one(monkeypatch, tmp_path):
    import chromadb

    built = []

    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def fake_persistent_client(path):
        client = FakeClient()
        built.append(client)
        return client

    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)

    provider = ChromaClientProvider(str(tmp_path))
    first = provider.get()
    provider.close()

    assert first.closed is True

    second = provider.get()

    assert second is not first
    assert len(built) == 2


def test_close_before_any_get_is_a_no_op(tmp_path):
    provider = ChromaClientProvider(str(tmp_path))

    provider.close()  # must not raise


def test_stores_sharing_a_provider_use_one_underlying_client(monkeypatch, tmp_path):
    from src.features.media_index.gallery_prompt_vector_store import GalleryPromptVectorStore
    from src.features.media_index.gallery_vector_store import GalleryVectorStore
    from src.features.prompt_database.vector_store import PromptVectorStore

    calls = []

    def fake_persistent_client(path):
        calls.append(path)
        return object()

    import chromadb

    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)

    provider = ChromaClientProvider(str(tmp_path))
    gallery = GalleryVectorStore(persist_dir=str(tmp_path), client_provider=provider)
    gallery_prompts = GalleryPromptVectorStore(persist_dir=str(tmp_path), client_provider=provider)
    prompts = PromptVectorStore(persist_dir=str(tmp_path), client_provider=provider)

    assert gallery.client is gallery_prompts.client is prompts.client
    assert len(calls) == 1


class _FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def shared_stores(monkeypatch, tmp_path):
    """Three real stores sharing one provider, with a fake chromadb client
    factory - used to prove no store keeps serving a client the provider
    has since closed."""
    from src.features.media_index.gallery_prompt_vector_store import GalleryPromptVectorStore
    from src.features.media_index.gallery_vector_store import GalleryVectorStore
    from src.features.prompt_database.vector_store import PromptVectorStore

    built = []

    def fake_persistent_client(path):
        client = _FakeClient()
        built.append(client)
        return client

    import chromadb

    monkeypatch.setattr(chromadb, "PersistentClient", fake_persistent_client)

    provider = ChromaClientProvider(str(tmp_path))
    stores = [
        GalleryVectorStore(persist_dir=str(tmp_path), client_provider=provider),
        GalleryPromptVectorStore(persist_dir=str(tmp_path), client_provider=provider),
        PromptVectorStore(persist_dir=str(tmp_path), client_provider=provider),
    ]
    return provider, stores, built


def test_shared_store_lifecycle_never_serves_a_closed_client_across_close_boundary(shared_stores):
    provider, stores, built = shared_stores

    before = [store.client for store in stores]
    assert len({id(client) for client in before}) == 1
    assert len(built) == 1

    provider.close()

    assert built[0].closed is True

    after = [store.client for store in stores]
    assert len({id(client) for client in after}) == 1
    assert len(built) == 2
    assert all(client is not before[0] for client in after)
    assert all(client.closed is False for client in after)


def test_importing_bootstrap_app_does_not_import_chromadb():
    env = dict(os.environ)
    site_packages = os.path.join(REPO_ROOT, "venv", "lib", "python3.12", "site-packages")
    env["PYTHONPATH"] = os.pathsep.join([site_packages, REPO_ROOT, env.get("PYTHONPATH", "")])

    result = subprocess.run(
        [sys.executable, "-c", "import sys; import src.bootstrap.app; print('chromadb' in sys.modules)"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", result.stderr


class _Collection:
    def __init__(self, count=0, fail=False):
        self._count = count
        self._fail = fail
        self.deleted = []

    def count(self):
        if self._fail:
            raise RuntimeError("boom")
        return self._count

    def delete(self, ids):
        if self._fail:
            raise RuntimeError("boom")
        self.deleted.append(ids)


class _Client:
    def __init__(self, collection):
        self.collection = collection
        self.names = []

    def get_or_create_collection(self, name, metadata):
        self.names.append(name)
        return self.collection


class _PrefixedStore(ChromaCollectionStore):
    collection_prefix = "pre_"


def _store(collection):
    store = _PrefixedStore(embedder_slug="slug")
    store._client = _Client(collection)
    return store


def test_collection_name_is_prefix_user_and_embedder():
    store = _store(_Collection())

    store.get_collection("u1")

    assert store.client.names == ["pre_u1__slug"]


def test_is_collection_empty_reflects_count_and_defaults_false_on_error(caplog):
    assert _store(_Collection(count=0)).is_collection_empty("u1") is True
    assert _store(_Collection(count=2)).is_collection_empty("u1") is False
    with caplog.at_level("ERROR"):
        assert _store(_Collection(fail=True)).is_collection_empty("u1") is False
    assert "pre_u1" in caplog.text


def test_bulk_delete_skips_empty_ids_and_swallows_errors(caplog):
    collection = _Collection()
    store = _store(collection)

    store.bulk_delete("u1", [])
    store.bulk_delete("u1", ["a", "b"])

    assert collection.deleted == [["a", "b"]]
    assert store.client.names == ["pre_u1__slug"]
    with caplog.at_level("ERROR"):
        _store(_Collection(fail=True)).bulk_delete("u1", ["a"])
    assert "pre_u1" in caplog.text
