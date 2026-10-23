import json
from unittest.mock import Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from src.features.automation import http_node
from src.features.automation.context import AutomationServices
from src.features.automation.engine import AutomationEngine
from src.features.automation.expr import render_template
from src.features.automation.hooks import AUTOMATION_HOOKS
from src.features.automation.header_secrets import HTTP_REQUEST_TYPE, mask_graph, protect_graph
from src.features.automation.http_request import MAX_RESPONSE_BYTES, HttpRequestError, PreparedRequest, RawResponse, RequestSpec, aiohttp_transport, execute
from src.features.automation.nodes import actions
from src.features.automation.records import Automation
from src.features.automation.runtime import AutomationRuntime
from src.platform.plugins.automation_nodes import NodeResult, NodeTypeRegistry, NodeTypeSpec
from src.platform.security.secrets import SecretCipher, configure_secret_cipher, generate_key


@pytest.fixture(autouse=True)
def cipher():
    configure_secret_cipher(SecretCipher([generate_key()]))
    yield
    configure_secret_cipher(None)


def http_node_config(**overrides):
    config = {
        "method": "POST",
        "url": "https://api.example.com/items/{{ event.item }}",
        "query": [{"name": "q", "value": "{{ event.item }}"}],
        "headers": [
            {"name": "Authorization", "value": "Bearer {{ event.token }}"},
            {"name": "X-Plain", "value": "visible"},
        ],
        "body_mode": "json",
        "body": '{"name": "{{ event.item }}"}',
    }
    config.update(overrides)
    return config


class Recorder:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __call__(self, request: PreparedRequest) -> RawResponse:
        self.calls.append(request)
        return self.response


@pytest.mark.asyncio
async def test_url_headers_query_and_body_are_templated_and_secret_headers_revealed():
    graph = protect_graph({"nodes": [{"id": "call", "type": HTTP_REQUEST_TYPE, "config": http_node_config()}]})
    config = graph["nodes"][0]["config"]
    assert config["headers"][0]["value"].startswith("enc:v1:")
    assert config["headers"][1]["value"] == "visible"

    recorder = Recorder(RawResponse(200, {"Content-Type": "application/json"}, b'{"id": 42}'))
    result = await http_node.execute(http_node.build_spec(config, {"event": {"item": "cat", "token": "t0k3n"}, "upstream": {}}), recorder)

    sent = recorder.calls[0]
    assert sent.url == "https://api.example.com/items/cat?q=cat"
    assert sent.headers["Authorization"] == "Bearer t0k3n"
    assert sent.body == b'{"name": "cat"}'
    assert result.body == {"id": 42}


@pytest.mark.asyncio
async def test_pick_is_rendered_against_the_response():
    recorder = Recorder(RawResponse(200, {"Content-Type": "application/json"}, b'{"id": 42}'))
    config = http_node_config(response_pick="item-{{ response.body.id }}", headers=[], body_mode="none")
    output = await http_node.perform(config, {"event": {"item": "a"}, "upstream": {}}, recorder)
    assert output["pick"] == "item-42"
    assert output["status"] == 200


def test_secret_flag_and_sensitive_names_are_sealed_others_left_alone():
    graph = {"nodes": [{"id": "n", "type": HTTP_REQUEST_TYPE, "config": {"headers": [
        {"name": "X-Custom", "value": "s3cret", "secret": True},
        {"name": "X-Api-Key", "value": "k"},
        {"name": "Accept", "value": "json"},
    ]}}]}
    rows = protect_graph(graph)["nodes"][0]["config"]["headers"]
    assert rows[0]["value"].startswith("enc:v1:")
    assert rows[1]["value"].startswith("enc:v1:")
    assert rows[2]["value"] == "json"
    assert graph["nodes"][0]["config"]["headers"][0]["value"] == "s3cret"


def test_masked_value_round_trips_to_the_stored_secret():
    stored = protect_graph({"nodes": [{"id": "n", "type": HTTP_REQUEST_TYPE, "config": {"headers": [
        {"name": "Authorization", "value": "Bearer abc"}]}}]})
    masked = mask_graph(stored)
    assert masked["nodes"][0]["config"]["headers"][0]["value"] == "***"
    assert "Bearer abc" not in json.dumps(masked)
    resaved = protect_graph(masked, stored)
    assert resaved["nodes"][0]["config"]["headers"][0]["value"] == stored["nodes"][0]["config"]["headers"][0]["value"]


def test_masked_value_with_nothing_stored_becomes_empty_not_literal_mask():
    graph = {"nodes": [{"id": "n", "type": HTTP_REQUEST_TYPE, "config": {"headers": [{"name": "Authorization", "value": "***"}]}}]}
    assert protect_graph(graph)["nodes"][0]["config"]["headers"][0]["value"] == ""


def test_automation_api_dict_never_exposes_the_secret():
    stored = protect_graph({"nodes": [{"id": "n", "type": HTTP_REQUEST_TYPE, "config": {"headers": [
        {"name": "Authorization", "value": "Bearer abc"}]}}]})
    payload = Automation(id="a", name="x", graph=stored).to_dict()
    assert "abc" not in json.dumps(payload)
    assert "enc:v1:" not in json.dumps(payload)


class FakeRepo:
    def __init__(self, automation):
        self.automation = automation
        self.run_nodes = {}
        self.runs = {}
        self.n = 0

    def get_by_id(self, automation_id, user_id=None):
        return self.automation

    def create_run(self, run):
        run.id = "run1"
        self.runs[run.id] = run
        return run

    def finish_run(self, run_id, status, error=None, duration_ms=None):
        self.runs[run_id].status = status
        return True

    def touch_last_run(self, *args, **kwargs):
        return True

    def create_run_node(self, node):
        self.n += 1
        node.id = f"rn{self.n}"
        self.run_nodes[node.id] = node
        return node

    def update_run_node(self, run_node_id, status, output=None, error=None, finished=False):
        node = self.run_nodes[run_node_id]
        node.status, node.output, node.error = status, output, error
        return True

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def list_run_nodes(self, run_id):
        return list(self.run_nodes.values())


@pytest.mark.asyncio
async def test_engine_run_passes_the_response_to_a_later_node_and_redacts_the_run_record(monkeypatch):
    recorder = Recorder(RawResponse(200, {"Content-Type": "application/json"}, b'{"id": 42}'))
    monkeypatch.setattr(http_node, "aiohttp_transport", recorder)
    rendered = []

    async def later(ctx):
        rendered.append(render_template("{{ upstream.call.body.id }}", ctx.template_context()))
        return NodeResult(output={})

    registry = NodeTypeRegistry()
    registry.register(NodeTypeSpec(key="trigger.manual", kind="trigger", title="Manual"))
    registry.register(NodeTypeSpec(key="action.later", kind="action", title="Later", execute=later))
    actions.register(registry)

    graph = protect_graph({
        "nodes": [
            {"id": "t", "type": "trigger.manual", "config": {}},
            {"id": "call", "type": HTTP_REQUEST_TYPE, "config": http_node_config()},
            {"id": "after", "type": "action.later", "config": {}},
        ],
        "edges": [
            {"id": "e1", "source": "t", "source_handle": "out", "target": "call", "target_handle": "in"},
            {"id": "e2", "source": "call", "source_handle": "out", "target": "after", "target_handle": "in"},
        ],
    })
    repo = FakeRepo(Automation(id="a1", name="x", graph=graph))
    plugin_registry = Mock()
    engine = AutomationEngine(repository=repo, registry=registry, plugin_registry=plugin_registry)

    run_id = await engine.run("a1", "t", {"item": "cat", "token": "t0k3n"})

    assert repo.get_run(run_id).status == "success"
    assert rendered == ["42"]
    assert recorder.calls[0].headers["Authorization"] == "Bearer t0k3n"
    stored = json.dumps([n.input for n in repo.run_nodes.values()]) + json.dumps([n.output for n in repo.run_nodes.values()])
    assert "t0k3n" not in stored
    assert "enc:v1:" not in stored
    hook_payloads = [
        call.kwargs["initial_data"]
        for call in plugin_registry.execute_hook.call_args_list
        if call.args[0] == AUTOMATION_HOOKS.node_before_execute
    ]
    assert hook_payloads
    assert "t0k3n" not in json.dumps(hook_payloads)
    assert "enc:v1:" not in json.dumps(hook_payloads)
    call_input = next(n.input for n in repo.run_nodes.values() if n.node_id == "call")
    assert call_input["config"]["headers"][0]["value"] == "***"


@pytest.mark.asyncio
async def test_node_failure_message_carries_status_and_no_secret(monkeypatch):
    recorder = Recorder(RawResponse(500, {}, b"boom"))
    monkeypatch.setattr(http_node, "aiohttp_transport", recorder)
    registry = NodeTypeRegistry()
    actions.register(registry)
    spec = registry.get(HTTP_REQUEST_TYPE)
    assert spec.requires_admin is True

    from src.features.automation.context import NodeExecutionContext, RunContext

    ctx = NodeExecutionContext(
        run=RunContext(automation_id="a", run_id="r", event={}, services=AutomationServices()),
        node_id="n", node_type=HTTP_REQUEST_TYPE,
        config={"url": "https://api.example.com", "method": "GET"},
    )
    with pytest.raises(RuntimeError, match="HTTP 500: boom"):
        await spec.execute(ctx)


@pytest.mark.asyncio
async def test_runtime_test_endpoint_resolves_a_masked_stored_secret(monkeypatch):
    recorder = Recorder(RawResponse(200, {"Content-Type": "text/plain"}, b"ok"))
    monkeypatch.setattr(http_node, "aiohttp_transport", recorder)
    stored = protect_graph({"nodes": [{"id": "call", "type": HTTP_REQUEST_TYPE, "config": {"headers": [
        {"name": "Authorization", "value": "Bearer real"}]}}]})

    class Repo:
        def get_by_id(self, automation_id, user_id=None):
            return Automation(id="a1", name="x", graph=stored)

    runtime = AutomationRuntime.__new__(AutomationRuntime)
    runtime.repository = Repo()
    config = {"url": "https://api.example.com", "method": "GET", "headers": [{"name": "Authorization", "value": "***"}]}

    output = await runtime.test_http_request(config, {}, {}, automation_id="a1", node_id="call")

    assert recorder.calls[0].headers["Authorization"] == "Bearer real"
    assert output["body"] == "ok"


@pytest.mark.asyncio
async def test_real_transport_caps_the_body_and_does_not_follow_redirects():
    async def big(request):
        return web.Response(body=b"x" * (MAX_RESPONSE_BYTES + 5000), content_type="text/plain")

    async def redirect(request):
        raise web.HTTPFound("/big")

    app = web.Application()
    app.router.add_get("/big", big)
    app.router.add_get("/go", redirect)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    try:
        base = f"http://127.0.0.1:{server.port}"
        result = await execute(RequestSpec("GET", f"{base}/big", allow_private_targets=True), aiohttp_transport)
        assert result.truncated is True
        assert len(result.body) == MAX_RESPONSE_BYTES
        with pytest.raises(Exception, match="HTTP 302"):
            await execute(RequestSpec("GET", f"{base}/go", allow_private_targets=True), aiohttp_transport)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_secret_flagged_header_with_a_neutral_name_is_scrubbed_everywhere(monkeypatch):
    recorder = Recorder(RawResponse(400, {"Content-Type": "text/plain"}, b"rejected zz-secret-9999"))
    monkeypatch.setattr(http_node, "aiohttp_transport", recorder)
    sealed = protect_graph({"nodes": [{"id": "call", "type": HTTP_REQUEST_TYPE, "config": {
        "url": "https://api.example.com",
        "headers": [{"name": "X-Foo", "value": "zz-secret-9999", "secret": True}],
    }}]})
    config = sealed["nodes"][0]["config"]

    with pytest.raises(HttpRequestError) as caught:
        await http_node.perform(config, {"event": {}, "upstream": {}})
    assert "zz-secret-9999" not in str(caught.value)

    registry = NodeTypeRegistry()
    registry.register(NodeTypeSpec(key="trigger.manual", kind="trigger", title="Manual"))
    actions.register(registry)
    graph = {
        "nodes": [{"id": "t", "type": "trigger.manual", "config": {}}, {"id": "call", "type": HTTP_REQUEST_TYPE, "config": config}],
        "edges": [{"id": "e1", "source": "t", "source_handle": "out", "target": "call", "target_handle": "in"}],
    }
    repo = FakeRepo(Automation(id="a1", name="x", graph=graph))
    run_id = await AutomationEngine(repository=repo, registry=registry).run("a1", "t", {})
    assert repo.get_run(run_id).status == "failed"
    stored = json.dumps([n.error for n in repo.run_nodes.values()]) + json.dumps([n.input for n in repo.run_nodes.values()])
    assert "zz-secret-9999" not in stored

    class Repo:
        def get_by_id(self, automation_id, user_id=None):
            return Automation(id="a1", name="x", graph=sealed)

    runtime = AutomationRuntime.__new__(AutomationRuntime)
    runtime.repository = Repo()
    with pytest.raises(HttpRequestError) as probe:
        await runtime.test_http_request(
            {"url": "https://api.example.com", "headers": [{"name": "X-Foo", "value": "***", "secret": True}]},
            {}, {}, automation_id="a1", node_id="call",
        )
    assert "zz-secret-9999" not in str(probe.value)
