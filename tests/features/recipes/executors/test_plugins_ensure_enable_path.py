import functools
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.features.providers.registry as module
from src.features.plugins import operations
from src.features.plugins.records import Plugin
from src.features.providers.registry import ProviderRegistry
from src.features.recipes.catalog import RecipeCatalog
from src.features.recipes.executors._provider_credentials import resolve_provider_registry
from src.features.recipes.executors.artifacts_fetch import resolve_download_url
from src.features.recipes.executors.base import StepContext
from src.features.recipes.executors.plugins_ensure import PluginsEnsureExecutor
from src.features.recipes.records import RecipeRun, RecipeRunStatus
from src.features.recipes.schema import Recipe, RecipeArtifact, RecipeChecksum, RecipeStep
from src.platform.plugins.registry import PluginState

PLUGIN_ID = "huggingface-provider"
HF_URL = "https://huggingface.example.test/org/repo/resolve/main/model.safetensors"
RECIPE_YAML = """
schema_version: 1
id: "shipped-by-plugin"
version: 1
name: "Shipped By Plugin"
engine: "native"
category: "utility"
plugins: []
presets:
  - preset_id: "PRESET1"
steps:
  - key: "backend.ensure"
    kind: "backend.ensure"
    title: "Ensure backend"
    params:
      engine: "native"
"""


class _HuggingFaceProvider:
    def get_metadata(self):
        return type("Meta", (), {"has_capability": lambda self, capability: True})()

    async def initialize(self, settings):
        return True

    async def get_download_url(self, model_id, version_id=None):
        return HF_URL if model_id == "org/repo" else None


class _PluginRegistry:
    def __init__(self, plugin_dir):
        self.enabled = set()
        self.manifest = SimpleNamespace(
            id=PLUGIN_ID,
            name="Hugging Face",
            recipes=[{"path": "recipes"}],
            plugin_dir=plugin_dir,
            provides_hooks=None,
            category="other",
            tags=[],
            capabilities=[],
            source="local",
            shadows=None,
            homepage=None,
            repository=None,
            hooks={},
            frontend_hooks=[],
            settings=[],
        )
        self.hook_chain = Mock()
        self.boot = []
        self.ready = []

    def get_plugin(self, plugin_id):
        return self.manifest if plugin_id == PLUGIN_ID else None

    def get_enabled_plugins(self):
        return [self.manifest] if PLUGIN_ID in self.enabled else []

    def get_plugin_state(self, plugin_id):
        return PluginState.ENABLED if plugin_id in self.enabled else PluginState.DISABLED

    def enable_plugin(self, plugin_id):
        self.enabled.add(plugin_id)
        return True

    def get_plugin_error(self, plugin_id):
        return None

    def run_boot_hook(self, plugin_id):
        self.boot.append(plugin_id)

    def run_ready_hook(self, plugin_id):
        self.ready.append(plugin_id)


class _PluginRepository:
    def __init__(self):
        self.enabled = False

    def get_plugin_by_id(self, plugin_id):
        return Plugin(
            id=plugin_id, name="Hugging Face", version="1.0.0", type="backend-only", enabled=self.enabled,
            manifest_path="manifest.yml", description="d", author="a",
            installed_at=datetime(2024, 1, 1), updated_at=datetime(2024, 1, 1),
        )

    def enable_plugin(self, plugin_id):
        self.enabled = True
        return True


@pytest.fixture
def harness(monkeypatch, tmp_path):
    plugin_dir = tmp_path / "plugin"
    (plugin_dir / "recipes").mkdir(parents=True)
    (plugin_dir / "recipes" / "shipped-by-plugin.yml").write_text(RECIPE_YAML, encoding="utf-8")
    plugins = _PluginRegistry(plugin_dir)
    repository = _PluginRepository()
    catalog = RecipeCatalog(str(tmp_path / "recipes"), plugin_registry=plugins)

    async def discover(self):
        if PLUGIN_ID in plugins.enabled:
            self._providers["huggingface"] = _HuggingFaceProvider()
        return []

    monkeypatch.setattr(module, "_provider_registry", None)
    monkeypatch.setattr(module, "_discovery_done", False)
    monkeypatch.setattr(module, "_discovery_task", None)
    monkeypatch.setattr(module, "PluginRepository", lambda: None)
    monkeypatch.setattr(ProviderRegistry, "discover_providers", discover)
    monkeypatch.setattr(ProviderRegistry, "_get_provider_settings", lambda self, provider_id: {})
    enable = functools.partial(operations.enable_plugin, repository, plugins, recipe_catalog=catalog)
    return SimpleNamespace(
        plugins=plugins, repository=repository, catalog=catalog, executor=PluginsEnsureExecutor(plugins, enable)
    )


def _context():
    run = RecipeRun(id="r1", recipe_id="x", recipe_version=1, scope="instance", status=RecipeRunStatus.RUNNING, created_by="owner-1")
    recipe = Recipe(id="x", schema_version=1, version=1, name="X", engine="native")
    step = RecipeStep(key="plugins.ensure", kind="plugins.ensure", title="Enable", params={"plugin_ids": [PLUGIN_ID]})
    return StepContext(run=run, recipe=recipe, step=step)


def _artifact():
    return RecipeArtifact(
        id="ckpt", kind="checkpoint", model_type="checkpoint", filename="model.safetensors", display_name="Model",
        checksum=RecipeChecksum(algorithm="sha256", value=None),
        provider_hint={"source": "huggingface", "model_id": "org/repo"},
    )


def test_enabling_through_the_recipe_persists_the_plugin(harness):
    assert harness.repository.enabled is False

    result = harness.executor.execute(_context())

    assert result.success is True
    assert harness.repository.enabled is True
    assert harness.plugins.boot == [PLUGIN_ID]
    assert harness.plugins.ready == [PLUGIN_ID]


def test_recipes_shipped_by_the_enabled_plugin_become_visible(harness):
    assert harness.catalog.get_recipe("shipped-by-plugin") is None

    harness.executor.execute(_context())

    assert harness.catalog.get_recipe("shipped-by-plugin") is not None


def test_provider_enabled_through_the_recipe_is_visible_to_download_resolution(harness):
    assert resolve_provider_registry().get_provider("huggingface") is None

    harness.executor.execute(_context())

    url, via = resolve_download_url(_artifact(), resolve_provider_registry)
    assert url == HF_URL
    assert via == "huggingface (matched by id)"
