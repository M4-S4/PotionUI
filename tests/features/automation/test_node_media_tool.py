import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.features.automation.context import AutomationServices, NodeExecutionContext, RunContext
from src.features.automation.nodes import register_builtin_nodes
from src.features.automation.nodes.actions import _media_tool_options
from src.features.automation.routes import AutomationController
from src.platform.plugins.automation_nodes import NodeTypeRegistry, resolved_config_schema
from src.platform.security.user import AccountType, User
from tests.features.media_tools.support import MediaToolsTestBase

KEY = "action.media_tool"


class MediaToolNodeCase(MediaToolsTestBase):

    def setUp(self):
        super().setUp()
        self.nodes = NodeTypeRegistry()
        register_builtin_nodes(self.nodes)
        self.spec = self.nodes.get(KEY)

    def ctx(self, config, event=None, upstream=None, service="default"):
        services = AutomationServices(media_tool_service=self.service if service == "default" else service)
        run = RunContext(automation_id="a1", run_id="r1", event=event or {}, services=services)
        run.upstream = upstream or {}
        return NodeExecutionContext(run=run, node_id="resize", node_type=KEY, config=config)

    def run_node(self, config, event=None, upstream=None, service="default"):
        return asyncio.run(self.spec.execute(self.ctx(config, event, upstream, service)))

    def upload_event(self, item, user_id=None):
        return {"upload_id": item.id, "user_id": user_id or self.user_id, "media_type": item.media_type, "purpose": "user_upload"}

    def row(self, item_id, user_id=None):
        return self.upload_repo.get_by_id(item_id, user_id or self.user_id)


class TestSpec(MediaToolNodeCase):

    def test_the_node_is_an_admin_grade_media_action_with_defaults_that_read_an_upload_event(self):
        fields = {field["name"]: field for field in self.spec.config_schema}

        self.assertEqual((self.spec.kind, self.spec.category), ("action", "media"))
        self.assertTrue(self.spec.requires_admin)
        self.assertEqual(fields["item_id"]["default"], "{{ event.upload_id }}")
        self.assertEqual(fields["user_id"]["default"], "{{ event.user_id }}")
        self.assertEqual(fields["mode"]["default"], "new")
        self.assertEqual([option["value"] for option in fields["mode"]["options"]], ["new", "replace"])
        self.assertEqual(
            {field.key for field in self.spec.outputs},
            {"item_id", "item_ids", "items", "count", "replaced", "tool"},
        )

    def test_a_non_admin_cannot_author_a_graph_that_uses_it(self):
        controller = AutomationController(MagicMock(), registry=self.nodes)
        graph = {"nodes": [{"id": "n", "type": KEY, "config": {}}], "edges": []}
        member = User(
            id="u", username="u", email="u@example.com", password_hash="h", account_type=AccountType.USER,
        )
        admin = User(
            id="a", username="a", email="a@example.com", password_hash="h", account_type=AccountType.ADMIN,
        )

        self.assertIsNotNone(controller._admin_denial(graph, member))
        self.assertIsNone(controller._admin_denial(graph, admin))

    def test_the_tool_picker_lists_runnable_catalog_tools_including_plugin_ones(self):
        container = SimpleNamespace(media_tool_registry=self.registry)

        with patch("src.platform.plugins.runtime_registries.get_container", return_value=container):
            values = {option["value"] for option in _media_tool_options()}
            schema = resolved_config_schema(self.spec)

        self.assertIn("shrink", values)
        self.assertIn("acme.mirror", values)
        self.assertNotIn("acme.panel", values)
        self.assertEqual({o["value"] for o in schema[0]["options"]}, values)

    def test_the_tool_picker_falls_back_to_the_core_tools_before_the_app_is_composed(self):
        values = {option["value"] for option in _media_tool_options()}

        self.assertTrue({"trim", "resize", "rotate", "concat"} <= values)


class TestRun(MediaToolNodeCase):

    def test_a_new_upload_event_runs_the_tool_on_a_tiny_image_and_hands_over_the_new_item(self):
        item = self.image_item(size=(40, 30))

        result = self.run_node(
            {"tool": "shrink", "params": '{"size": 12}', "mode": "new"}, event=self.upload_event(item)
        )

        output = result.output
        self.assertEqual(result.branch, "out")
        self.assertEqual(output["tool"], "shrink")
        self.assertEqual(output["count"], 1)
        self.assertFalse(output["replaced"])
        self.assertEqual(output["item_ids"], [output["item_id"]])
        self.assertNotEqual(output["item_id"], item.id)
        [entry] = output["items"]
        self.assertEqual((entry["width"], entry["height"], entry["media_type"]), (12, 12, "image"))
        self.assertTrue(entry["url"].startswith("/api/media/uploads/"))
        created = self.row(output["item_id"])
        self.assertEqual((created.width, created.height), (12, 12))
        self.assertEqual(self.row(item.id).filename, item.filename)

    def test_defaults_alone_are_enough_when_the_tool_has_defaults(self):
        item = self.image_item()

        result = self.run_node({"tool": "shrink"}, event=self.upload_event(item))

        self.assertEqual(result.output["items"][0]["width"], 8)

    def test_a_later_node_chains_on_the_item_id_an_earlier_node_produced(self):
        item = self.image_item()
        first = self.run_node({"tool": "shrink", "params": {"size": 20}}, event=self.upload_event(item))

        second = self.run_node(
            {"tool": "shrink", "params": {"size": 6}, "item_id": "{{ upstream.resize.item_id }}"},
            event=self.upload_event(item),
            upstream={"resize": first.output},
        )

        self.assertNotEqual(second.output["item_id"], first.output["item_id"])
        self.assertEqual(second.output["items"][0]["width"], 6)
        self.assertEqual(len(self.upload_rows()), 3)

    def test_replace_overwrites_the_original_and_keeps_its_id(self):
        item = self.image_item()

        result = self.run_node(
            {"tool": "shrink", "params": {"size": 9}, "mode": "replace"}, event=self.upload_event(item)
        )

        self.assertTrue(result.output["replaced"])
        self.assertEqual(result.output["item_id"], item.id)
        self.assertNotEqual(self.row(item.id).filename, item.filename)
        self.assertEqual(len(self.upload_rows()), 1)

    def test_param_values_are_templated_and_coerced_by_the_platform_validator(self):
        item = self.image_item()
        event = {**self.upload_event(item), "target": "10"}

        result = self.run_node({"tool": "shrink", "params": {"size": "{{ event.target }}"}}, event=event)

        self.assertEqual(result.output["items"][0]["width"], 10)

    def test_a_json_text_param_may_embed_a_template(self):
        item = self.image_item()
        event = {**self.upload_event(item), "target": 7}

        result = self.run_node({"tool": "shrink", "params": '{"size": {{ event.target }}}'}, event=event)

        self.assertEqual(result.output["items"][0]["width"], 7)

    def test_a_plugin_tool_runs_like_a_core_tool(self):
        item = self.image_item()

        result = self.run_node({"tool": "acme.mirror", "params": {"size": 5}}, event=self.upload_event(item))

        self.assertEqual(result.output["items"][0]["width"], 5)

    def test_the_node_timeout_is_passed_to_the_service(self):
        item = self.image_item()
        seen = {}
        original = self.service.run_sync

        async def spy(*args, **kwargs):
            seen.update(kwargs)
            return await original(*args, **kwargs)

        self.service.run_sync = spy

        self.run_node({"tool": "shrink", "timeout_s": 90}, event=self.upload_event(item))

        self.assertEqual(seen, {"timeout": 90.0})


class TestFailures(MediaToolNodeCase):

    def failure(self, config, event=None, **kwargs):
        with self.assertRaises(RuntimeError) as caught:
            self.run_node(config, event=event, **kwargs)
        return str(caught.exception)

    def test_invalid_params_fail_the_node_with_the_step_and_the_field(self):
        item = self.image_item()

        message = self.failure({"tool": "shrink", "params": {"size": 9999}}, self.upload_event(item))

        self.assertTrue(message.startswith("action.media_tool: invalid_params (step 1, 'size'): "), message)
        self.assertIn("at most 64", message)
        self.assertEqual(len(self.upload_rows()), 1)

    def test_an_unknown_param_fails_the_node(self):
        item = self.image_item()

        message = self.failure({"tool": "shrink", "params": {"colour": 1}}, self.upload_event(item))

        self.assertIn("Unknown parameter: colour", message)

    def test_an_unknown_tool_fails_the_node(self):
        item = self.image_item()

        message = self.failure({"tool": "nope"}, self.upload_event(item))

        self.assertIn("unknown_tool", message)

    def test_a_tool_that_does_not_take_the_media_kind_fails_the_node(self):
        item = self.image_item()

        message = self.failure({"tool": "toimage"}, self.upload_event(item))

        self.assertIn("unsupported_media", message)

    def test_a_missing_item_fails_the_node(self):
        message = self.failure({"tool": "shrink"}, {"upload_id": "gone", "user_id": self.user_id})

        self.assertIn("action.media_tool: not_found: Media not found", message)
        self.assertEqual(self.upload_rows(), [])

    def test_an_item_that_belongs_to_another_user_is_not_found_for_the_user_it_runs_as(self):
        item = self.image_item(user_id=self.other_user_id)

        message = self.failure({"tool": "shrink"}, self.upload_event(item, user_id=self.user_id))

        self.assertIn("not_found", message)
        self.assertEqual(len(self.upload_rows(self.other_user_id)), 1)
        self.assertEqual(self.upload_rows(), [])

    def test_running_as_the_owner_edits_that_owners_item_and_publishes_into_their_library(self):
        item = self.image_item(user_id=self.other_user_id)

        result = self.run_node({"tool": "shrink"}, self.upload_event(item, user_id=self.other_user_id))

        self.assertIsNotNone(self.row(result.output["item_id"], self.other_user_id))
        self.assertIsNone(self.row(result.output["item_id"], self.user_id))

    def test_an_event_without_an_item_or_a_user_fails_before_any_run(self):
        item = self.image_item()

        no_item = self.failure({"tool": "shrink"}, {"user_id": self.user_id})
        no_user = self.failure({"tool": "shrink"}, {"upload_id": item.id})

        self.assertIn("no library item id", no_item)
        self.assertIn("no user to run as", no_user)
        self.assertEqual(len(self.upload_rows()), 1)

    def test_a_generated_file_id_is_not_a_library_item(self):
        record, _path = self.generation_file()

        message = self.failure(
            {"tool": "shrink", "mode": "replace", "item_id": "{{ event.file }}"},
            {"file": record.id, "user_id": self.user_id},
        )

        self.assertIn("not_found", message)

    def test_a_quota_refusal_fails_the_node_with_the_limit_message(self):
        item = self.image_item()
        self.guard.refuse_check = True

        message = self.failure({"tool": "shrink"}, self.upload_event(item))

        self.assertIn("limit_exceeded", message)
        self.assertIn("Upload limit reached", message)

    def test_a_tool_that_blows_up_fails_the_node_without_leaking_internals(self):
        item = self.image_item()

        message = self.failure({"tool": "explode"}, self.upload_event(item))

        self.assertIn("tool_failed", message)
        self.assertNotIn("/srv/private", message)

    def test_bad_configuration_is_named(self):
        item = self.image_item()
        event = self.upload_event(item)

        self.assertIn("no tool selected", self.failure({}, event))
        self.assertIn("mode must be one of new, replace", self.failure({"tool": "shrink", "mode": "copy"}, event))
        self.assertIn("params is not valid JSON", self.failure({"tool": "shrink", "params": "{size:"}, event))
        self.assertIn("params must be a JSON object", self.failure({"tool": "shrink", "params": "[1]"}, event))
        self.assertIn("params must be a JSON object", self.failure({"tool": "shrink", "params": 5}, event))
        self.assertIn(
            "no MediaToolService", self.failure({"tool": "shrink"}, event, service=None)
        )
        self.assertEqual(len(self.upload_rows()), 1)

    def test_blank_params_mean_the_tool_defaults(self):
        item = self.image_item()

        for blank in ("", "   ", None):
            with self.subTest(params=blank):
                result = self.run_node({"tool": "shrink", "params": blank}, self.upload_event(item))
                self.assertEqual(result.output["items"][0]["width"], 8)

    def test_the_outputs_are_json_serialisable_for_the_run_record(self):
        item = self.image_item()

        result = self.run_node({"tool": "shrink"}, self.upload_event(item))

        self.assertEqual(json.loads(json.dumps(result.output)), result.output)
