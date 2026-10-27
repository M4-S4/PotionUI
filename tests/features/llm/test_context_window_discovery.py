import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.chat.conversation import ConversationRunner
from src.features.llm import context_budget
from src.features.llm import routes as llm_mod
from src.features.llm import ttl_cache
from src.features.llm.clients.native import NativeLLMClient, read_checkpoint_context_window
from src.features.llm.clients.ollama import OllamaClient, parse_show_context_window
from src.features.llm.clients.openai import OpenAIClient
from src.features.llm.gateway import LLMGateway
from src.features.llm.model_listing import DiscoveredWindow
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User
from tests.features.llm.wire_capture import OLLAMA_REPLY, install_wire_capture, json_response, make_config
from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.build import user_message
from tests.fixtures.llm_engine import client_gateway, drain, scripted_gateway
from tests.fixtures.llm_harness import ScriptedAdapter, text_step

SHOW_RUNTIME = {"parameters": "stop   <|im_end|>\nnum_ctx                        6000\ntemperature 0.7"}
SHOW_TRAINED = {"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 131072}}


def raise_timeout():
    raise httpx.ReadTimeout("slow")


def window(tokens, kind="runtime"):
    return DiscoveredWindow(tokens=tokens, kind=kind)


class TestOllamaParsing:
    @pytest.mark.parametrize("payload, expected", [
        (SHOW_RUNTIME, window(6000, "runtime")),
        (SHOW_TRAINED, window(131072, "trained")),
        ({**SHOW_RUNTIME, **SHOW_TRAINED}, window(6000, "runtime")),
    ])
    def test_modelfile_num_ctx_is_runtime_and_beats_trained_context_length(self, payload, expected):
        assert parse_show_context_window(payload) == expected

    @pytest.mark.parametrize("payload", [
        None,
        [],
        {},
        {"parameters": 5, "model_info": "x"},
        {"parameters": "temperature 0.7", "model_info": {"general.architecture": "x"}},
        {"model_info": {"llama.context_length": "131072"}},
        {"model_info": {"llama.context_length": 0}},
        {"model_info": {"llama.context_length": True}},
        {"parameters": "num_ctx abc"},
    ])
    def test_missing_or_odd_fields_yield_nothing(self, payload):
        assert parse_show_context_window(payload) is None

    @pytest.mark.asyncio
    async def test_probe_posts_to_api_show_with_the_model_and_key(self, monkeypatch):
        capture = install_wire_capture(monkeypatch, [json_response(SHOW_RUNTIME)])

        result = await OllamaClient().discover_context_window("http://ollama.local:11434/", "sk-1", "qwen3:8b")

        assert result == window(6000, "runtime")
        assert capture.requests[0].method == "POST"
        assert capture.url == "http://ollama.local:11434/api/show"
        assert capture.body == {"model": "qwen3:8b"}
        assert capture.headers["authorization"] == "Bearer sk-1"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", [
        json_response({"error": "model not found"}, status=404),
        raise_timeout,
    ])
    async def test_http_failures_and_timeouts_yield_nothing(self, monkeypatch, failure):
        install_wire_capture(monkeypatch, [failure])
        assert await OllamaClient().discover_context_window("http://x:11434", None, "m") is None

    @pytest.mark.asyncio
    async def test_bad_base_url_yields_nothing(self):
        assert await OllamaClient().discover_context_window("not a url", None, "m") is None


class TestOpenAIParsing:
    MODELS = {"data": [
        {"id": "other", "context_length": 1},
        {"id": "or/model", "context_length": 200000},
        {"id": "vllm/model", "max_model_len": 32768},
        {"id": "deepseek/model", "context_window": 1048576},
        {"id": "plain/model"},
    ]}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model, expected", [
        ("or/model", window(200000)),
        ("vllm/model", window(32768)),
        ("deepseek/model", window(1048576)),
        ("plain/model", None),
        ("absent/model", None),
    ])
    async def test_reads_runtime_window_from_the_models_listing(self, monkeypatch, model, expected):
        capture = install_wire_capture(monkeypatch, [json_response(self.MODELS)])

        result = await OpenAIClient().discover_context_window("https://api.example.com/v1/", "sk-2", model)

        assert result == expected
        assert capture.url == "https://api.example.com/v1/models"
        assert capture.headers["authorization"] == "Bearer sk-2"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [
        {"data": "nope"},
        {"data": [None, 5, {"id": "m", "context_length": "big"}]},
        [],
    ])
    async def test_odd_payloads_yield_nothing(self, monkeypatch, payload):
        install_wire_capture(monkeypatch, [json_response(payload)])
        assert await OpenAIClient().discover_context_window("http://x/v1", None, "m") is None

    @pytest.mark.asyncio
    async def test_connection_failure_yields_nothing(self, monkeypatch):
        def refuse():
            raise httpx.ConnectError("refused")

        install_wire_capture(monkeypatch, [refuse])
        assert await OpenAIClient().discover_context_window("http://x/v1", None, "m") is None


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestNativeReading:
    @pytest.mark.parametrize("files", [
        {"config.json": {"max_position_embeddings": 40960}},
        {"config.json": {"text_config": {"max_position_embeddings": 40960}}},
        {"config.json": {"max_position_embeddings": 40960}, "tokenizer_config.json": {"model_max_length": 1024}},
        {"config.json": {"hidden_size": 8}, "tokenizer_config.json": {"model_max_length": 40960}},
    ])
    def test_reads_trained_window_preferring_config_json(self, tmp_path, files):
        for name, payload in files.items():
            write_json(tmp_path / name, payload)
        assert read_checkpoint_context_window(str(tmp_path)) == window(40960, "trained")

    def test_hf_sentinel_is_ignored(self, tmp_path):
        write_json(tmp_path / "tokenizer_config.json", {"model_max_length": int(1e30)})
        assert read_checkpoint_context_window(str(tmp_path)) is None

    def test_missing_or_broken_files_yield_nothing(self, tmp_path):
        assert read_checkpoint_context_window(str(tmp_path)) is None
        (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
        write_json(tmp_path / "tokenizer_config.json", ["x"])
        assert read_checkpoint_context_window(str(tmp_path)) is None

    @pytest.mark.asyncio
    async def test_client_resolves_the_checkpoint_and_reads_local_files_only(self, tmp_path):
        write_json(tmp_path / "config.json", {"max_position_embeddings": 32768})
        client = NativeLLMClient()
        client._resolve_model = lambda name: (str(tmp_path), False)

        assert await client.discover_context_window("", None, "qwen") == window(32768, "trained")

    @pytest.mark.asyncio
    async def test_adopted_text_encoder_and_unresolvable_names_yield_nothing(self, tmp_path):
        client = NativeLLMClient()
        client._resolve_model = lambda name: (str(tmp_path / "te.safetensors"), True)
        assert await client.discover_context_window("", None, "text_encoders/te.safetensors") is None

        def unresolvable(name):
            raise ValueError("no such checkpoint")

        client._resolve_model = unresolvable
        assert await client.discover_context_window("", None, "missing") is None


class TestResolveCapacityOrder:
    def test_config_window_beats_discovered(self):
        config = make_config("openai", provider_options={"context_window": 4096})
        assert context_budget.resolve_capacity(config, window(100000)) == context_budget.CapacityInfo(4096, "config")

    def test_ollama_num_ctx_beats_discovered(self):
        config = make_config("ollama", provider_options={"num_ctx": 2048})
        assert context_budget.resolve_capacity(config, window(100000)) == context_budget.CapacityInfo(2048, "config")

    def test_discovered_beats_unknown(self):
        config = make_config("openai")
        assert context_budget.resolve_capacity(config, window(100000)) == context_budget.CapacityInfo(100000, "discovered")
        assert context_budget.resolve_capacity(config) == context_budget.CapacityInfo(
            context_budget.UNKNOWN_CAPACITY_TOKENS, "unknown",
        )

    @pytest.mark.parametrize("provider, discovered, tokens", [
        ("ollama", window(131072, "trained"), 32768),
        ("ollama", window(8192, "trained"), 8192),
        ("ollama", window(65536, "runtime"), 65536),
        ("native", window(131072, "trained"), 131072),
    ])
    def test_discovered_window_is_capped_only_for_ollama_trained(self, provider, discovered, tokens):
        capacity = context_budget.resolve_capacity(make_config(provider), discovered)
        assert capacity == context_budget.CapacityInfo(tokens, "discovered")


@pytest.fixture
def gateway():
    return client_gateway(Mock())


async def _tokens():
    yield {"type": "token", "content": "ok"}


def stub_probe(gateway, type_id, result=None, error=None):
    probe = AsyncMock(return_value=result, side_effect=error)
    gateway._clients[type_id].discover_context_window = probe
    return probe


class TestGatewayCache:
    @pytest.mark.asyncio
    async def test_ensure_fills_the_cache_and_the_profile_reads_it(self, gateway):
        probe = stub_probe(gateway, "openai", window(100000))
        config = make_config("openai")

        assert gateway.context_profile_for(config).capacity.source == "unknown"
        await gateway.ensure_context_window(config)
        capacity = gateway.context_profile_for(config).capacity

        assert capacity == context_budget.CapacityInfo(100000, "discovered")
        probe.assert_awaited_once_with("http://peer.invalid:1234", None, "golden-model")

    @pytest.mark.asyncio
    async def test_positive_result_is_not_probed_again(self, gateway):
        probe = stub_probe(gateway, "openai", window(100000))
        config = make_config("openai")

        await gateway.ensure_context_window(config)
        await gateway.ensure_context_window(config)

        assert probe.await_count == 1

    @pytest.mark.asyncio
    async def test_negative_result_is_cached_and_does_not_poison_the_profile(self, gateway):
        probe = stub_probe(gateway, "openai", None)
        config = make_config("openai")

        await gateway.ensure_context_window(config)
        await gateway.ensure_context_window(config)

        assert probe.await_count == 1
        assert gateway.context_profile_for(config).capacity.source == "unknown"

    @pytest.mark.asyncio
    async def test_negative_entries_expire_sooner_than_positive_ones(self, gateway, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(ttl_cache.time, "monotonic", lambda: now[0])
        gateway = client_gateway(Mock())
        miss_probe = stub_probe(gateway, "openai", None)
        hit_probe = stub_probe(gateway, "ollama", window(4096, "runtime"))
        missing = make_config("openai")
        found = make_config("ollama")

        await gateway.ensure_context_window(missing)
        await gateway.ensure_context_window(found)
        now[0] += 301
        await gateway.ensure_context_window(missing)
        await gateway.ensure_context_window(found)
        assert (miss_probe.await_count, hit_probe.await_count) == (2, 1)

        now[0] += 3600
        await gateway.ensure_context_window(found)
        assert hit_probe.await_count == 2

    @pytest.mark.asyncio
    async def test_explicit_window_skips_the_probe(self, gateway):
        probe = stub_probe(gateway, "openai", window(100000))

        await gateway.ensure_context_window(make_config("openai", provider_options={"context_window": 4096}))
        await gateway.ensure_context_window(
            make_config("ollama", provider_options={"num_ctx": 2048}),
        )

        probe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_is_keyed_by_type_url_and_model(self, gateway):
        probe = stub_probe(gateway, "openai", window(100000))

        await gateway.ensure_context_window(make_config("openai"))
        await gateway.ensure_context_window(make_config("openai", model="other"))
        await gateway.ensure_context_window(make_config("openai", base_url="http://other.invalid"))

        assert probe.await_count == 3

    @pytest.mark.asyncio
    async def test_probe_exceptions_never_escape(self, gateway):
        stub_probe(gateway, "openai", error=RuntimeError("boom"))
        config = make_config("openai")

        await gateway.ensure_context_window(config)

        assert gateway.context_profile_for(config).capacity.source == "unknown"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("streaming", [False, True])
    @pytest.mark.parametrize("probe_error", [None, RuntimeError("boom")])
    async def test_every_send_path_primes_the_window_and_survives_a_failing_probe(self, streaming, probe_error):
        adapter = ScriptedAdapter(text_step("reply"))
        adapter.discover_context_window = AsyncMock(return_value=window(4096, "runtime"), side_effect=probe_error)
        gateway = scripted_gateway(adapter)

        if streaming:
            text = "".join(event.text for event in await drain(gateway.stream("cfg-scripted", [user_message("hi")])) if hasattr(event, "text"))
        else:
            text = (await gateway.complete("cfg-scripted", [user_message("hi")])).text

        adapter.discover_context_window.assert_awaited_once()
        assert text == "reply"

    @pytest.mark.asyncio
    async def test_provider_without_a_hook_is_skipped(self, gateway):
        assert all(gateway.supports_context_window_discovery(t) for t in ("ollama", "openai", "native"))
        assert gateway.supports_context_window_discovery("anthropic") is False

        await gateway.ensure_context_window(make_config("anthropic"))


class TestOllamaWire:
    CHAT_REPLY = OLLAMA_REPLY

    async def send(self, monkeypatch, config, show_payload):
        capture = install_wire_capture(monkeypatch, [json_response(show_payload), json_response(self.CHAT_REPLY)])
        gateway = client_gateway(Mock())
        gateway.repository.get_configuration.return_value = config
        await gateway.complete("cfg-1", [user_message("hi")])
        return capture

    @pytest.mark.asyncio
    @pytest.mark.parametrize("show, num_ctx", [(SHOW_RUNTIME, 6000), (SHOW_TRAINED, 32768)])
    async def test_discovered_window_is_sent_as_num_ctx_capped_when_trained(self, monkeypatch, show, num_ctx):
        capture = await self.send(monkeypatch, make_config("ollama"), show)

        assert [r.url.path for r in capture.requests] == ["/api/show", "/api/chat"]
        assert capture.body["options"]["num_ctx"] == num_ctx

    @pytest.mark.asyncio
    async def test_explicit_num_ctx_wins_and_nothing_is_probed(self, monkeypatch):
        capture = await self.send(
            monkeypatch, make_config("ollama", provider_options={"num_ctx": 2048}), SHOW_RUNTIME,
        )

        assert "/api/show" not in [r.url.path for r in capture.requests]
        assert capture.body["options"]["num_ctx"] == 2048

    @pytest.mark.asyncio
    async def test_undiscoverable_window_sends_no_num_ctx(self, monkeypatch):
        capture = await self.send(monkeypatch, make_config("ollama"), {})

        assert "num_ctx" not in capture.body["options"]

    @pytest.mark.asyncio
    async def test_budget_for_the_send_uses_the_discovered_window(self, monkeypatch):
        capture = install_wire_capture(monkeypatch, [json_response(SHOW_RUNTIME), json_response(OLLAMA_REPLY)])
        gateway = client_gateway(Mock())
        config = make_config("ollama", max_tokens=60000)
        gateway.repository.get_configuration.return_value = config

        await gateway.complete("cfg-1", [user_message("hi")])

        assert capture.body["options"]["num_predict"] < 6000


class TestPreflightUsesDiscoveredWindow:
    @pytest.mark.asyncio
    async def test_first_preflight_after_priming_sees_the_discovered_window(self, monkeypatch):
        install_wire_capture(monkeypatch, [json_response(SHOW_RUNTIME)])
        gateway = client_gateway(Mock())
        config = make_config("ollama")
        gateway.repository.get_configuration.return_value = config
        manager = Mock()
        manager.llm_service = gateway
        runner = ConversationRunner(manager)
        session = Mock(llm_config_id="cfg-1")

        assert runner._resolve_budget_inputs(session, Settings()).capacity.source == "unknown"
        await runner._prime_context_window(session)
        profile = runner._resolve_budget_inputs(session, Settings())

        assert profile.capacity == context_budget.CapacityInfo(6000, "discovered")

    @pytest.mark.asyncio
    async def test_priming_never_raises(self):
        manager = Mock()
        manager.llm_service.repository.get_configuration.side_effect = RuntimeError("db down")
        runner = ConversationRunner(manager)

        await runner._prime_context_window(Mock(llm_config_id="cfg-1"))
        await runner._prime_context_window(Mock(llm_config_id=None))


def admin_app(gateway, repository=None) -> TestClient:
    controller = llm_mod.LLMController(
        llm_repository=repository or Mock(), llm_service=gateway, settings=Mock(), plugin_registry=Mock(),
    )
    app = FastAPI()
    app.include_router(llm_mod.build_router(SimpleNamespace(llm_controller=controller)))
    user = User(username="u", email="u@example.com", password_hash="x", account_type=AccountType.ADMIN, id="u-1")
    app.dependency_overrides[get_current_active_user] = lambda: user
    return TestClient(app, raise_server_exceptions=False)


URL = "/api/llm/context-window/discover"
BODY = {"type": "ollama", "base_url": "http://ollama:11434", "api_key": None, "config_id": None, "model": "qwen3:8b"}
DATA_KEYS = {"supported", "detected", "detected_kind", "effective", "discovered_at", "error"}


class TestEndpoint:
    def test_non_admin_is_forbidden(self):
        user = User(username="u", email="u@example.com", password_hash="x", account_type=AccountType.USER, id="u-1")
        controller = llm_mod.LLMController(
            llm_repository=Mock(), llm_service=Mock(), settings=Mock(), plugin_registry=Mock(),
        )
        app = FastAPI()
        app.include_router(llm_mod.build_router(SimpleNamespace(llm_controller=controller)))
        app.dependency_overrides[get_current_active_user] = lambda: user

        assert TestClient(app, raise_server_exceptions=False).post(URL, json=BODY).status_code == 403

    def test_unsupported_type(self):
        gateway = client_gateway(Mock())
        data = admin_app(gateway).post(URL, json={**BODY, "type": "anthropic"}).json()["data"]

        assert data == {
            "supported": False, "detected": None, "detected_kind": None,
            "effective": None, "discovered_at": None, "error": None,
        }

    def test_trained_window_reports_detected_and_capped_effective(self, monkeypatch):
        capture = install_wire_capture(monkeypatch, [json_response(SHOW_TRAINED)])
        data = admin_app(client_gateway(Mock())).post(URL, json=BODY).json()["data"]

        assert set(data) == DATA_KEYS
        assert data["supported"] is True
        assert data["detected"] == 131072
        assert data["detected_kind"] == "trained"
        assert data["effective"] == 32768
        assert data["error"] is None
        assert data["discovered_at"].endswith("+00:00")
        assert capture.body == {"model": "qwen3:8b"}

    def test_runtime_window_is_effective_as_is(self, monkeypatch):
        install_wire_capture(monkeypatch, [json_response(SHOW_RUNTIME)])
        data = admin_app(client_gateway(Mock())).post(URL, json=BODY).json()["data"]

        assert (data["detected"], data["detected_kind"], data["effective"]) == (6000, "runtime", 6000)

    def test_nothing_detected_is_a_plain_error(self, monkeypatch):
        install_wire_capture(monkeypatch, [json_response({}, status=500)])
        data = admin_app(client_gateway(Mock())).post(URL, json=BODY).json()["data"]

        assert set(data) == DATA_KEYS
        assert data["supported"] is True
        assert data["detected"] is None and data["effective"] is None
        assert data["error"] and "http" not in data["error"].lower() and "500" not in data["error"]
        assert data["discovered_at"]

    def test_unexpected_failure_hides_the_exception_text(self):
        gateway = Mock()
        gateway.supports_context_window_discovery.return_value = True
        gateway.refresh_context_window = AsyncMock(side_effect=RuntimeError("secret /srv/path"))
        resp = admin_app(gateway).post(URL, json=BODY)

        assert "secret" not in resp.text and "/srv/path" not in resp.text
        assert resp.json()["data"]["error"]

    @pytest.mark.parametrize("stored_type, typed, expected", [
        ("ollama", None, "Bearer sk-stored"),
        ("ollama", "sk-typed", "Bearer sk-typed"),
        ("openai", None, None),
        (None, None, None),
    ])
    def test_key_resolution_prefers_typed_then_same_type_stored_and_never_echoes(
        self, monkeypatch, stored_type, typed, expected,
    ):
        capture = install_wire_capture(monkeypatch, [json_response(SHOW_RUNTIME)])
        repository = Mock()
        repository.get_configuration.return_value = (
            SimpleNamespace(type=stored_type, api_key="sk-stored") if stored_type else None
        )
        resp = admin_app(client_gateway(Mock()), repository).post(
            URL, json={**BODY, "api_key": typed, "config_id": "c1"},
        )

        assert capture.headers.get("authorization") == expected
        assert "sk-stored" not in resp.text

    @pytest.mark.asyncio
    async def test_probe_bypasses_and_refreshes_the_cache(self, gateway):
        probe = stub_probe(gateway, "ollama", window(4096, "runtime"))
        config = make_config("ollama", base_url="http://ollama:11434", model="qwen3:8b")
        await gateway.ensure_context_window(config)
        probe.return_value = window(8192, "runtime")

        data = admin_app(gateway).post(URL, json=BODY).json()["data"]

        assert data["detected"] == 8192
        assert probe.await_count == 2
        assert gateway.context_profile_for(config).capacity.capacity_tokens == 8192
