import pytest
import yaml
from pydantic import ValidationError

from src.platform.plugins.llm_providers import LLMProviderRegistry
from src.platform.plugins.manifest import PluginManifestSchema
from src.platform.plugins.registry import PluginRegistry, PluginState

CLIENT_SOURCE = """
class ExampleClient:
    async def list_models(self, base_url, api_key=None):
        return []
"""

TYPE_NAME = "example-llm"
BASE_URL = "https://llm.example.org/v1"
ENTRY = {"type": TYPE_NAME, "label": "Example LLM", "backend": "client:ExampleClient", "default_base_url": BASE_URL}
MANIFEST = {
    "id": "p", "name": "P", "version": "1.0.0", "description": "d", "author": "a", "type": "backend-only",
}


def test_schema_accepts_an_llm_provider():
    schema = PluginManifestSchema.model_validate({**MANIFEST, "llm_providers": [ENTRY]})

    assert [(s.type, s.label, s.backend, s.default_base_url) for s in schema.llm_providers] == [
        (TYPE_NAME, "Example LLM", "client:ExampleClient", BASE_URL),
    ]


@pytest.mark.parametrize("entry", [
    {key: value for key, value in ENTRY.items() if key != "backend"},
    {key: value for key, value in ENTRY.items() if key != "label"},
    {**ENTRY, "models": ["a"]},
])
def test_schema_rejects_an_incomplete_or_unknown_entry(entry):
    with pytest.raises(ValidationError):
        PluginManifestSchema.model_validate({**MANIFEST, "llm_providers": [entry]})


@pytest.fixture
def providers():
    return LLMProviderRegistry()


@pytest.fixture
def registry(tmp_path, providers):
    marketplace = tmp_path / "marketplace"
    local = tmp_path / "local"
    marketplace.mkdir()
    local.mkdir()
    return PluginRegistry(str(marketplace), str(local), llm_provider_registry=providers)


def write_plugin(registry, plugin_id, entry):
    plugin_dir = registry.loader.marketplace_dir / plugin_id
    plugin_dir.mkdir()
    (plugin_dir / "client.py").write_text(CLIENT_SOURCE)
    (plugin_dir / "manifest.yml").write_text(yaml.dump({**MANIFEST, "id": plugin_id, "name": plugin_id, "llm_providers": [entry]}))


def test_enabling_registers_the_type_and_disabling_removes_it(registry, providers):
    write_plugin(registry, "llm-plugin", ENTRY)

    assert registry.enable_plugin("llm-plugin")
    registration = providers.get(TYPE_NAME)
    assert (registration.label, registration.default_base_url, registration.source) == (
        "Example LLM", BASE_URL, "llm-plugin",
    )
    assert type(registration.client).__name__ == "ExampleClient"

    registry.disable_plugin("llm-plugin")

    assert providers.get(TYPE_NAME) is None


def test_a_type_another_plugin_already_owns_fails_the_enable(registry, providers):
    write_plugin(registry, "first-plugin", ENTRY)
    write_plugin(registry, "second-plugin", ENTRY)

    assert registry.enable_plugin("first-plugin")
    assert not registry.enable_plugin("second-plugin")

    assert registry.get_plugin_state("second-plugin") == PluginState.ERROR
    assert providers.get(TYPE_NAME).source == "first-plugin"
