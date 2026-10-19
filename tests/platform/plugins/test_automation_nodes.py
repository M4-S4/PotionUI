"""Tests for NodeTypeRegistry (src/core/automation/registry.py)."""

import unittest

from src.platform.plugins.automation_nodes import (
    NodeResult,
    NodeTypeRegistry,
    NodeTypeSpec,
    resolved_config_schema,
)


class TestNodeTypeRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = NodeTypeRegistry()

    def test_condition_defaults_to_true_false_ports(self):
        spec = NodeTypeSpec(key="condition.compare", kind="condition", title="Compare")

        self.assertEqual(spec.output_ports, ("true", "false"))

    def test_trigger_and_action_default_to_out_port(self):
        trigger_spec = NodeTypeSpec(key="trigger.manual", kind="trigger", title="Manual")
        action_spec = NodeTypeSpec(key="action.noop", kind="action", title="Noop")

        self.assertEqual(trigger_spec.output_ports, ("out",))
        self.assertEqual(action_spec.output_ports, ("out",))

    def test_condition_explicit_ports_not_overridden(self):
        spec = NodeTypeSpec(key="condition.custom", kind="condition", title="Custom", output_ports=("yes", "no"))

        self.assertEqual(spec.output_ports, ("yes", "no"))

    def test_by_kind_filters(self):
        self.registry.register(NodeTypeSpec(key="trigger.manual", kind="trigger", title="Manual"))
        self.registry.register(NodeTypeSpec(key="condition.compare", kind="condition", title="Compare"))
        self.registry.register(NodeTypeSpec(key="action.noop", kind="action", title="Noop"))

        self.assertEqual([s.key for s in self.registry.by_kind("trigger")], ["trigger.manual"])
        self.assertEqual([s.key for s in self.registry.by_kind("condition")], ["condition.compare"])
        self.assertEqual([s.key for s in self.registry.by_kind("action")], ["action.noop"])

    def test_node_result_defaults(self):
        result = NodeResult()

        self.assertIsNone(result.output)
        self.assertEqual(result.branch, "out")
        self.assertFalse(result.waiting)


class TestResolvedConfigSchema(unittest.TestCase):
    """The generic dynamic-options mechanism: `options_provider` callables are
    resolved into `configuration.options` and never leak into the output."""

    def test_field_without_provider_passes_through_unchanged(self):
        spec = NodeTypeSpec(key="action.noop", kind="action", title="Noop", config_schema=[
            {"name": "title", "type": "string", "label": "Title"},
        ])

        resolved = resolved_config_schema(spec)

        self.assertEqual(resolved, [{"name": "title", "type": "string", "label": "Title"}])

    def test_options_provider_is_called_and_inlined(self):
        def provider():
            return [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}]

        spec = NodeTypeSpec(key="trigger.custom", kind="trigger", title="Custom", config_schema=[
            {"name": "choice", "type": "select", "title": "Choice", "options_provider": provider},
        ])

        resolved = resolved_config_schema(spec)

        # Top-level `options`, NOT nested under `configuration` - matches what
        # SelectField.svelte actually reads (`config.options`).
        self.assertEqual(resolved[0]["options"], [
            {"value": "a", "label": "A"}, {"value": "b", "label": "B"},
        ])
        self.assertNotIn("configuration", resolved[0])

    def test_options_provider_callable_never_leaks_into_output(self):
        spec = NodeTypeSpec(key="trigger.custom", kind="trigger", title="Custom", config_schema=[
            {"name": "choice", "type": "select", "label": "Choice", "options_provider": lambda: []},
        ])

        resolved = resolved_config_schema(spec)

        self.assertNotIn("options_provider", resolved[0])
        # Must be plain-JSON-serializable - no callables anywhere in the tree.
        import json
        json.dumps(resolved)

    def test_provider_output_sits_alongside_other_top_level_keys(self):
        def provider():
            return [{"value": "x", "label": "X"}]

        spec = NodeTypeSpec(key="trigger.custom", kind="trigger", title="Custom", config_schema=[
            {"name": "choice", "type": "select", "title": "Choice", "default": "x", "options_provider": provider},
        ])

        resolved = resolved_config_schema(spec)

        self.assertEqual(resolved[0]["default"], "x")
        self.assertEqual(resolved[0]["options"], [{"value": "x", "label": "X"}])

    def test_provider_is_called_fresh_each_resolution_not_cached(self):
        calls = []

        def provider():
            calls.append(1)
            return [{"value": str(len(calls)), "label": str(len(calls))}]

        spec = NodeTypeSpec(key="trigger.custom", kind="trigger", title="Custom", config_schema=[
            {"name": "choice", "type": "select", "label": "Choice", "options_provider": provider},
        ])

        resolved_config_schema(spec)
        resolved_config_schema(spec)

        self.assertEqual(len(calls), 2)


class TestBuiltinNodeRegistration(unittest.TestCase):
    """Ensures the core node modules register cleanly onto a fresh registry (no duplicate keys)."""

    def test_register_builtin_nodes_onto_fresh_registry(self):
        from src.features.automation.nodes import register_builtin_nodes

        registry = NodeTypeRegistry()
        register_builtin_nodes(registry)

        keys = {spec.key for spec in registry.all()}
        self.assertIn("trigger.filesystem", keys)
        self.assertIn("trigger.schedule", keys)
        self.assertIn("trigger.gpu_threshold", keys)
        self.assertIn("trigger.manual", keys)
        self.assertIn("trigger.hook_event", keys)
        self.assertIn("condition.compare", keys)
        self.assertIn("condition.path_match", keys)
        self.assertIn("condition.jinja_expression", keys)
        self.assertIn("action.index_model", keys)
        self.assertIn("action.add_tag", keys)
        self.assertIn("action.fetch_provider_metadata", keys)
        self.assertIn("action.send_notification", keys)
        self.assertIn("action.wait_for_gpu", keys)


if __name__ == '__main__':
    unittest.main()
