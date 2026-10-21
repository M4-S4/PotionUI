import threading
import unittest

from tests.fixtures.persistence_base import PersistenceTestBase
from src.features.plugins.repository import PluginRepository
from src.features.plugins.records import Plugin, PluginHook
from datetime import datetime


class TestPluginRepositoryAdvanced(PersistenceTestBase):

    def setUp(self):
        super().setUp()
        self.repo = PluginRepository()

    def _create_sample_plugin(self, plugin_id="test_plugin_01", **kwargs):
        defaults = {
            "id": plugin_id,
            "name": "Test Plugin",
            "version": "1.0.0",
            "type": "full-stack",
            "enabled": True,
            "manifest_path": "/plugins/test/manifest.json",
            "description": "A test plugin",
            "author": "Test Author",
            "installed_at": datetime.now(),
            "updated_at": datetime.now()
        }
        defaults.update(kwargs)
        return Plugin(**defaults)

    def test_setting_overwrite_preserves_metadata(self):
        plugin = self._create_sample_plugin()
        self.repo.create_plugin(plugin)

        setting1 = self.repo.set_plugin_setting(plugin.id, "api_key", "secret123", is_secret=True)
        self.assertTrue(setting1.is_secret)

        setting2 = self.repo.set_plugin_setting(plugin.id, "api_key", "secret456", is_secret=True)
        self.assertTrue(setting2.is_secret)
        self.assertEqual(setting2.setting_value, "secret456")

    def test_get_settings_with_multiple_users(self):
        plugin = self._create_sample_plugin()
        self.repo.create_plugin(plugin)

        user1_id = self.create_test_user("user1", "user1", "user1@test.com")
        user2_id = self.create_test_user("user2", "user2", "user2@test.com")

        self.repo.set_plugin_setting(plugin.id, "global1", "value1")
        self.repo.set_plugin_setting(plugin.id, "global2", "value2")
        self.repo.set_plugin_setting(plugin.id, "user_pref", "user1_value", user_id=user1_id)
        self.repo.set_plugin_setting(plugin.id, "user_pref", "user2_value", user_id=user2_id)

        self.assertEqual(len(self.repo.get_plugin_settings(plugin.id)), 2)

        user1_settings = self.repo.get_plugin_settings(plugin.id, user_id=user1_id)
        self.assertEqual([s.setting_value for s in user1_settings], ["user1_value"])

        user2_settings = self.repo.get_plugin_settings(plugin.id, user_id=user2_id)
        self.assertEqual([s.setting_value for s in user2_settings], ["user2_value"])

    def test_delete_nonexistent_setting(self):
        plugin = self._create_sample_plugin()
        self.repo.create_plugin(plugin)

        self.assertFalse(self.repo.delete_plugin_setting(plugin.id, "nonexistent_key"))

    def test_get_hooks_by_type_mixed_plugins(self):
        for i in range(3):
            plugin = self._create_sample_plugin(f"plugin_{i}")
            self.repo.create_plugin(plugin)

            self.repo.register_hook(PluginHook(
                id=0,
                plugin_id=plugin.id,
                hook_name=f"backend_hook_{i}",
                hook_type="backend",
                handler_path=f"/path/to/backend_{i}.py"
            ))
            self.repo.register_hook(PluginHook(
                id=0,
                plugin_id=plugin.id,
                hook_name=f"frontend_hook_{i}",
                hook_type="frontend",
                component_path=f"/path/to/Component_{i}.svelte"
            ))

        self.assertEqual(len(self.repo.get_hooks_by_type("backend")), 3)
        self.assertEqual(len(self.repo.get_hooks_by_type("frontend")), 3)

    def test_clear_hooks_does_not_affect_other_plugins(self):
        plugin1 = self._create_sample_plugin("plugin_1")
        plugin2 = self._create_sample_plugin("plugin_2")

        self.repo.create_plugin(plugin1)
        self.repo.create_plugin(plugin2)

        for plugin in [plugin1, plugin2]:
            self.repo.register_hook(PluginHook(
                id=0,
                plugin_id=plugin.id,
                hook_name="test_hook",
                hook_type="backend",
                handler_path=f"/path/to/{plugin.id}.py"
            ))

        self.assertEqual(self.repo.clear_plugin_hooks(plugin1.id), 1)
        self.assertEqual(len(self.repo.get_plugin_hooks(plugin2.id)), 1)

    def test_hook_with_optional_fields_persist(self):
        plugin = self._create_sample_plugin()
        self.repo.create_plugin(plugin)

        self.repo.register_hook(PluginHook(
            id=0,
            plugin_id=plugin.id,
            hook_name="optional_hook",
            hook_type="backend",
            handler_path="/path/to/hook.py",
            component_path=None,
            position="top",
            sort_order=5
        ))

        stored = self.repo.get_plugin_hooks(plugin.id)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].sort_order, 5)
        self.assertEqual(stored[0].position, "top")

    def test_concurrent_plugin_creation(self):
        errors = []
        created_ids = []

        def create_plugin(index):
            try:
                created = self.repo.create_plugin(self._create_sample_plugin(f"concurrent_{index}"))
                created_ids.append(created.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_plugin, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(created_ids), 10)

    def test_concurrent_setting_updates(self):
        plugin = self._create_sample_plugin()
        self.repo.create_plugin(plugin)

        errors = []

        def update_setting(value):
            try:
                self.repo.set_plugin_setting(plugin.id, "counter", str(value))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_setting, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

        final = self.repo.get_plugin_setting(plugin.id, "counter")
        self.assertIn(final.setting_value, [str(i) for i in range(5)])

    def test_plugin_with_very_long_name(self):
        plugin = self._create_sample_plugin()
        plugin.name = "X" * 500

        self.repo.create_plugin(plugin)

        self.assertEqual(self.repo.get_plugin_by_id(plugin.id).name, "X" * 500)

    def test_plugin_with_special_characters_in_id(self):
        plugin = self._create_sample_plugin("plugin-with-dashes_and_underscores.v1")

        self.repo.create_plugin(plugin)

        fetched = self.repo.get_plugin_by_id(plugin.id)
        self.assertEqual(fetched.id, plugin.id)


if __name__ == '__main__':
    unittest.main()
