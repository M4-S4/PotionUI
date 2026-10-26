"""Plugins that declare `default_enabled` come up live on a fresh database and
never override an admin's recorded decision."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import yaml

from src.features.plugins import operations
from src.features.plugins.repository import PluginRepository
from src.features.providers.registry import ProviderRegistry
from src.platform.plugins.registry import PluginRegistry
from tests.fixtures.persistence_base import PersistenceTestBase

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_MARKETPLACE = REPO_ROOT / "content" / "plugins" / "marketplace"

HANDLER_SOURCE = """
def handle(context):
    context.set("handled", True)
    return context
"""


class DefaultEnabledPluginTests(PersistenceTestBase):
    def setUp(self):
        super().setUp()
        self.plugins_dir = Path(tempfile.mkdtemp())
        self.marketplace_dir = self.plugins_dir / "marketplace"
        self.local_dir = self.plugins_dir / "local"
        self.marketplace_dir.mkdir()
        self.local_dir.mkdir()
        self.repo = PluginRepository()

    def tearDown(self):
        shutil.rmtree(self.plugins_dir)
        super().tearDown()

    def _registry(self, marketplace=None):
        return PluginRegistry(
            marketplace_dir=str(marketplace or self.marketplace_dir),
            local_dir=str(self.local_dir),
        )

    def _write_plugin(self, plugin_id, default_enabled):
        plugin_dir = self.marketplace_dir / plugin_id
        plugin_dir.mkdir()
        (plugin_dir / "handlers.py").write_text(HANDLER_SOURCE)
        manifest = {
            "id": plugin_id,
            "name": plugin_id,
            "version": "1.0.0",
            "description": "Test plugin",
            "author": "Test Author",
            "type": "backend-only",
            "hooks": {"backend": [{"hook": "generation.before_start", "handler": "handlers.handle"}]},
        }
        if default_enabled is not None:
            manifest["default_enabled"] = default_enabled
        (plugin_dir / "manifest.yml").write_text(yaml.dump(manifest))

    def test_fresh_database_enables_only_flagged_plugins(self):
        self._write_plugin("flagged", True)
        self._write_plugin("plain", None)
        registry = self._registry()

        enabled = operations.enable_default_plugins(self.repo, registry)

        self.assertEqual(enabled, ["flagged"])
        self.assertTrue(self.repo.get_plugin_by_id("flagged").enabled)
        self.assertIsNone(self.repo.get_plugin_by_id("plain"))
        self.assertIn("flagged", [m.id for m in registry.get_enabled_plugins()])

    def test_recorded_disabled_plugin_stays_disabled_across_boots(self):
        self._write_plugin("flagged", True)
        registry = self._registry()
        operations.enable_default_plugins(self.repo, registry)
        operations.disable_plugin(self.repo, registry, "flagged")

        restarted = self._registry()
        enabled = operations.enable_default_plugins(self.repo, restarted)

        self.assertEqual(enabled, [])
        self.assertFalse(self.repo.get_plugin_by_id("flagged").enabled)
        self.assertEqual(restarted.get_enabled_plugins(), [])

    def test_scan_does_not_change_a_disabled_flagged_plugin(self):
        self._write_plugin("flagged", True)
        registry = self._registry()
        operations.enable_default_plugins(self.repo, registry)
        operations.disable_plugin(self.repo, registry, "flagged")

        operations.scan_plugins(self.repo, registry)
        operations.enable_default_plugins(self.repo, registry)

        self.assertFalse(self.repo.get_plugin_by_id("flagged").enabled)

    def test_enabling_again_is_a_no_op(self):
        self._write_plugin("flagged", True)
        registry = self._registry()
        operations.enable_default_plugins(self.repo, registry)

        self.assertEqual(operations.enable_default_plugins(self.repo, registry), [])
        self.assertTrue(self.repo.get_plugin_by_id("flagged").enabled)

    def test_a_failing_plugin_does_not_raise(self):
        self._write_plugin("flagged", True)
        registry = self._registry()
        registry.enable_plugin = lambda plugin_id: False

        self.assertEqual(operations.enable_default_plugins(self.repo, registry), [])

    def test_huggingface_provider_is_live_on_a_fresh_database(self):
        registry = self._registry(marketplace=REAL_MARKETPLACE)

        enabled = operations.enable_default_plugins(self.repo, registry)

        self.assertEqual(enabled, ["huggingface-provider"])
        self.assertTrue(self.repo.get_plugin_by_id("huggingface-provider").enabled)
        providers = ProviderRegistry(registry)
        asyncio.run(providers.discover_providers())
        self.assertIsNotNone(providers.get_provider("huggingface"))
