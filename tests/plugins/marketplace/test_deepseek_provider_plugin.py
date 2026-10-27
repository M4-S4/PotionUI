import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features.llm.gateway import LLMGateway
from src.features.llm.harness.adapters.openai_chat import OpenAIChatAdapter
from src.features.llm.harness.profile_seed import CONTENT_DIR, SeedKey
from src.features.llm.harness.profile_seed_catalog import SeedCatalog
from src.platform.plugins.llm_providers import LLMProviderRegistry
from src.platform.plugins.registry import PluginRegistry
from tests.fixtures.llm_engine import core_gateway
from tests.fixtures.llm_profiles import FakeClock, shipped_seed

PLUGIN_ID = "deepseek-provider"
PACKAGE = "deepseek_provider"
REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPO_ROOT / "content" / "plugins" / "marketplace" / PLUGIN_ID


@pytest.fixture(autouse=True)
def plugin_modules_removed_afterwards():
    modules, path = set(sys.modules), list(sys.path)
    try:
        yield
    finally:
        for name in set(sys.modules) - modules:
            if name.split(".")[0] in (PACKAGE, "backend"):
                del sys.modules[name]
        sys.path[:] = path


@pytest.fixture
def providers():
    return LLMProviderRegistry()


@pytest.fixture
def registry(tmp_path, providers):
    marketplace = tmp_path / "marketplace"
    shutil.copytree(PLUGIN_ROOT, marketplace / PLUGIN_ID, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    (tmp_path / "local").mkdir()
    return PluginRegistry(str(marketplace), str(tmp_path / "local"), llm_provider_registry=providers)


def test_enabling_the_plugin_adds_deepseek_to_the_configuration_types(registry, providers):
    assert registry.enable_plugin(PLUGIN_ID)

    types = core_gateway(SimpleNamespace(), provider_registry=providers).provider_types()

    assert types[-1] == {
        "type": "deepseek",
        "label": "DeepSeek",
        "adapter": "openai-chat",
        "default_base_url": "https://api.deepseek.com",
        "supports_model_listing": True,
        "source": PLUGIN_ID,
    }
    adapter = providers.get("deepseek").adapter
    assert isinstance(adapter, OpenAIChatAdapter)
    assert type(adapter).__module__ == f"{PACKAGE}.backend.adapter"


def test_disabling_the_plugin_takes_the_type_away(registry, providers):
    assert registry.enable_plugin(PLUGIN_ID)

    registry.disable_plugin(PLUGIN_ID)

    assert providers.get("deepseek") is None


def test_the_manifest_and_the_backend_name_the_same_plugin(registry):
    manifest = next(m for m in registry.loader.discover_plugins() if m.id == PLUGIN_ID)
    backend = registry.loader.load_plugin_module(manifest, "backend.adapter")

    assert manifest.validation_error is None
    assert manifest.id == backend.PLUGIN_ID


def test_the_plugin_supplies_the_deepseek_profile_rows_only_while_enabled(registry):
    clock = FakeClock()
    catalog = SeedCatalog(CONTENT_DIR, registry, clock=clock)
    key = SeedKey("openai-chat", "deepseek-v4-pro")
    assert catalog.layer(key).fields["replay_required_with_tools"] is False

    assert registry.enable_plugin(PLUGIN_ID)
    clock.now += 3
    enabled = catalog.layer(key).fields
    assert enabled["replay_required_with_tools"] is True
    assert enabled == shipped_seed().layer(key).fields

    registry.disable_plugin(PLUGIN_ID)
    clock.now += 3
    assert catalog.layer(key).fields["replay_required_with_tools"] is False


def test_the_manifest_declares_a_profiles_root_that_exists(registry):
    manifest = next(m for m in registry.loader.discover_plugins() if m.id == PLUGIN_ID)

    assert manifest.llm_profiles == [{"path": "profiles"}]
    assert (PLUGIN_ROOT / "profiles" / "deepseek.yml").is_file()
