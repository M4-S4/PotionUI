import pytest

from src.features.llm.harness.build import user_message
from tests.features.llm.wire_capture import install_wire_capture, json_response, make_config, ndjson_response
from tests.fixtures.llm_engine import ConfigRepository, core_gateway

SHOW_RUNTIME = {"parameters": "stop   <|im_end|>\nnum_ctx                        6000\ntemperature 0.7"}
SHOW_TRAINED = {"model_info": {"general.architecture": "qwen3", "qwen3.context_length": 131072}}
CHAT_REPLY = [{"message": {"content": "canned reply"}}, {"done": True, "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 10}]


async def send(monkeypatch, config, show):
    capture = install_wire_capture(monkeypatch, [json_response(show), ndjson_response(CHAT_REPLY)])
    gateway = core_gateway(ConfigRepository(config))
    completion = await gateway.complete(config.id, [user_message("hi")])
    return capture, completion


def chat_body(capture):
    return next(request for request in capture.requests if request.url.path == "/api/chat")


@pytest.mark.asyncio
@pytest.mark.parametrize("show, num_ctx", [(SHOW_RUNTIME, 6000), (SHOW_TRAINED, 32768)])
async def test_the_discovered_window_is_the_context_ollama_is_asked_to_load(monkeypatch, show, num_ctx):
    capture, completion = await send(monkeypatch, make_config("ollama"), show)

    assert [request.url.path for request in capture.requests] == ["/api/show", "/api/chat"]
    assert capture.body["options"]["num_ctx"] == num_ctx
    assert completion.text == "canned reply"


@pytest.mark.asyncio
async def test_an_explicit_requested_size_wins_over_the_discovered_window(monkeypatch):
    config = make_config("ollama", profile_overrides={"context_request": 2048}, max_tokens=60000)

    capture, _ = await send(monkeypatch, config, SHOW_RUNTIME)

    assert capture.body["options"]["num_ctx"] == 2048
    assert capture.body["options"]["num_predict"] < 2048


@pytest.mark.asyncio
async def test_an_undiscoverable_window_sends_no_num_ctx(monkeypatch):
    capture, _ = await send(monkeypatch, make_config("ollama"), {})

    assert "num_ctx" not in capture.body["options"]


@pytest.mark.asyncio
async def test_the_budget_for_the_send_plans_with_the_window_ollama_loads(monkeypatch):
    capture, _ = await send(monkeypatch, make_config("ollama", max_tokens=60000), SHOW_RUNTIME)

    assert capture.body["options"]["num_ctx"] == 6000
    assert capture.body["options"]["num_predict"] < 6000
