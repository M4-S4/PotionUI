import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from src.features.backends import native_remote_backend as backend_module
from src.features.backends.backend_config import NativeRemoteBackendConfig
from src.features.backends.backend_registry import BackendRegistry
from src.features.backends.native_remote_backend import RemoteNativeBackend
from src.features.remote_execution.model_bundle_builder import ModelBundleResolutionError


def _backend():
    config = NativeRemoteBackendConfig(id="remote-1", name="Remote", base_url="http://fake", worker_token="t")
    return RemoteNativeBackend(config)


class TestRemoteBackend:
    def test_the_bundle_is_built_with_the_digest_service_it_was_bound_to(self, mock_db, monkeypatch):
        calls = []

        def refusing(pipes, **kwargs):
            calls.append(kwargs)
            raise ModelBundleResolutionError("stop here")

        monkeypatch.setattr(backend_module, "build_model_bundle", refusing)
        monkeypatch.setattr(backend_module, "build_processed_pipeline", lambda pipes, catalog: SimpleNamespace(pipes=[]))
        monkeypatch.setattr(backend_module, "collect_input_assets", lambda pipes, storage: (None, None, []))
        backend = _backend()
        digests = object()
        backend.bind_model_locator(locator="locator")
        backend.bind_model_digests(digests=digests)
        emitted = []

        asyncio.run(backend._dispatch("g1", "p1", [], emitted.append))

        assert calls == [{"model_locator": "locator", "digests": digests}]

    def test_a_backend_that_was_never_bound_builds_with_no_digest_service(self, mock_db, monkeypatch):
        calls = []

        def refusing(pipes, **kwargs):
            calls.append(kwargs)
            raise ModelBundleResolutionError("stop here")

        monkeypatch.setattr(backend_module, "build_model_bundle", refusing)
        monkeypatch.setattr(backend_module, "build_processed_pipeline", lambda pipes, catalog: SimpleNamespace(pipes=[]))
        monkeypatch.setattr(backend_module, "collect_input_assets", lambda pipes, storage: (None, None, []))

        asyncio.run(_backend()._dispatch("g1", "p1", [], lambda output: None))

        assert calls[0]["digests"] is None


class Bindable:
    def __init__(self):
        self.digests = None

    def bind_model_digests(self, *, digests):
        self.digests = digests


class TestRegistry:
    def _registry(self):
        return BackendRegistry(generation_engine_factory=Mock())

    def test_a_backend_created_after_the_registry_got_the_service_is_bound_to_it(self):
        registry = self._registry()
        registry._registered_backend_types["bindable"] = lambda backend_config: Bindable()
        registry.bind_model_digests("digests")
        config = SimpleNamespace(effective_driver="bindable", engine="x")

        backend = registry._create_backend_instance(config)

        assert backend.digests == "digests"

    def test_a_backend_created_before_the_service_exists_is_left_unbound(self):
        registry = self._registry()
        registry._registered_backend_types["bindable"] = lambda backend_config: Bindable()
        config = SimpleNamespace(effective_driver="bindable", engine="x")

        assert registry._create_backend_instance(config).digests is None

    def test_a_backend_that_already_exists_is_bound_when_the_service_arrives(self):
        registry = self._registry()
        existing = Bindable()
        other = SimpleNamespace()
        registry._backends_cache = {"a": existing, "b": other}

        registry.bind_model_digests("digests")

        assert existing.digests == "digests"
        assert registry.model_digests == "digests"
