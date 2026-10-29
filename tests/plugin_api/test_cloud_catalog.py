from types import SimpleNamespace

from src.platform.plugins import runtime_registries
from src.plugin_api import cloud


def test_the_enabled_models_of_a_backend_come_from_its_catalog(monkeypatch):
    entries = [SimpleNamespace(model_id="m1"), SimpleNamespace(model_id="m2")]
    asked = []

    def list_enabled(backend_id):
        asked.append(backend_id)
        return entries

    container = SimpleNamespace(cloud_catalog=SimpleNamespace(repository=SimpleNamespace(list_enabled=list_enabled)))
    monkeypatch.setattr(runtime_registries, "_container", container)

    assert cloud.enabled_cloud_models("fal") == entries
    assert asked == ["fal"]


def test_a_cloud_model_resolves_to_its_backend_and_catalog_entry(monkeypatch):
    resolved = SimpleNamespace(backend_id="fal", entry=SimpleNamespace(model_id="m1"))
    container = SimpleNamespace(cloud_capabilities=SimpleNamespace(resolve={"m1": resolved}.get))
    monkeypatch.setattr(runtime_registries, "_container", container)

    assert cloud.resolve_cloud_model("m1") is resolved
    assert cloud.resolve_cloud_model("m2") is None


def test_the_catalog_reads_are_part_of_the_cloud_api():
    assert {"CloudCatalogEntry", "ResolvedCloudModel", "enabled_cloud_models", "resolve_cloud_model"} <= set(cloud.__all__)
