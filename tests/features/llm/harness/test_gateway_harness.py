from typing import Any, Dict, List

import pytest

from src.features.llm import trace_collector
from src.features.llm.harness.adapter import Settings, ToolSpec
from src.features.llm.harness.build import user_message
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, Usage
from src.features.llm.harness.messages import (
    Message,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    credential_fingerprint,
)
from src.features.llm.harness.profile import ReplayPolicy
from tests.fixtures.llm_engine import drain, scripted_config, scripted_gateway
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, ScriptedAdapter, text_step, tool_step


class Recorder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def record(self, **row: Any) -> None:
        self.rows.append(row)


@pytest.fixture
def recorder():
    recorder = Recorder()
    previous = trace_collector._recorder
    trace_collector.set_recorder(recorder)
    yield recorder
    trace_collector.set_recorder(previous)


async def test_complete_assembles_the_step_and_reports_usage_and_finish():
    adapter = ScriptedAdapter(text_step("Hi!", reasoning="greet"))
    gateway = scripted_gateway(adapter)

    completion = await gateway.complete("cfg-scripted", [user_message("hello")])

    assert completion.text == "Hi!"
    assert completion.message.reasoning() == (ReasoningPart(text="greet"),)
    assert completion.usage == Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert completion.finish is FinishReason.STOP and completion.finish_raw == "stop"
    assert completion.message.origin.adapter == "scripted"


async def test_prepare_resolves_the_system_message_profile_and_origin():
    config = scripted_config(api_key="sk-x", provider_options={"context_window": 32000})
    gateway = scripted_gateway(ScriptedAdapter(), config)

    custom = await gateway.prepare("cfg-scripted", [user_message("x")], system="Custom")
    default = await gateway.prepare("cfg-scripted", [user_message("x")])

    assert custom.request.system == "Custom"
    assert default.request.system == "You are terse."
    assert default.profile.context_window == 32000
    assert default.profile.replay is ReplayPolicy.DROP_EARLIER
    assert default.origin.credential == credential_fingerprint("http://scripted.invalid", "sk-x")
    assert default.origin.provider == "scripted" and default.origin.model == "scripted-model"
    assert default.request.stream is True


async def test_a_disabled_system_prompt_sends_none():
    gateway = scripted_gateway(ScriptedAdapter(), scripted_config(disable_system_prompt=True))

    prepared = await gateway.prepare("cfg-scripted", [user_message("x")])

    assert prepared.request.system == ""


@pytest.mark.parametrize(
    "config, error",
    [(None, "not found"), (scripted_config(enabled=False), "disabled"), (scripted_config(type="nope"), "Unsupported LLM type")],
)
async def test_prepare_refuses_unknown_disabled_or_unsupported_configs(config, error):
    gateway = scripted_gateway(ScriptedAdapter(), config or scripted_config(id="other"))

    with pytest.raises(ValueError, match=error):
        await gateway.prepare("cfg-scripted", [user_message("x")])


async def test_a_type_without_a_harness_adapter_is_refused():
    gateway = scripted_gateway(object())

    with pytest.raises(ValueError, match="no harness adapter"):
        await gateway.prepare("cfg-scripted", [user_message("x")])


async def test_a_known_window_clamps_max_tokens_and_trims_whole_old_units():
    config = scripted_config(max_tokens=4000, provider_options={"context_window": 2000})
    gateway = scripted_gateway(ScriptedAdapter(), config)
    old_call = Message(
        id="a1", role=Role.ASSISTANT, origin=SCRIPTED_ORIGIN,
        parts=(ToolCallPart(call_id="c1", name="echo", args={"message": "x" * 3000}),),
    )
    old_result = Message(id="t1", role=Role.TOOL, parts=(ToolResultPart(call_id="c1", name="echo", content=(TextPart(text="y" * 3000),)),))
    history = [user_message("old", message_id="u0"), old_call, old_result, user_message("new", message_id="u1")]

    prepared = await gateway.prepare("cfg-scripted", history)

    assert [message.id for message in prepared.request.messages] == ["u1"]
    assert prepared.ledger["groups_dropped"] == 2
    assert prepared.ledger["messages_dropped"] == 3
    assert prepared.request.settings.max_tokens == prepared.ledger["max_tokens_sent"] < 4000


async def test_an_unknown_window_never_clamps_max_tokens():
    gateway = scripted_gateway(ScriptedAdapter(), scripted_config(max_tokens=4000))

    prepared = await gateway.prepare("cfg-scripted", [user_message("x")])

    assert prepared.request.settings.max_tokens is None
    assert prepared.ledger["capacity_source"] == "unknown"


async def test_stream_yields_the_adapter_events_in_order():
    adapter = ScriptedAdapter([TextDelta(block=1, text="a"), TextDelta(block=1, text="b"), Finish(reason=FinishReason.STOP)])
    gateway = scripted_gateway(adapter)

    events = await drain(gateway.stream("cfg-scripted", [user_message("x")], settings=Settings(temperature=0.1)))

    assert events == [TextDelta(block=1, text="a"), TextDelta(block=1, text="b"), Finish(reason=FinishReason.STOP)]
    assert adapter.requests[0].settings.temperature == 0.1


async def test_every_step_is_traced_inside_an_active_context(recorder):
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "hi"}}, text="checking"), text_step("untraced"))
    gateway = scripted_gateway(adapter)
    tools = [ToolSpec(name="echo", description="Echo", parameters={"type": "object"})]

    with trace_collector.activate("s1", "u1", purpose="chat"):
        await gateway.complete("cfg-scripted", [user_message("x")], tools=tools)
    await gateway.complete("cfg-scripted", [user_message("x")])

    assert len(recorder.rows) == 1
    row = recorder.rows[0]
    assert row["provider"] == "scripted" and row["model"] == "scripted-model"
    assert row["request_system"] == "You are terse."
    assert row["request_tools"] == ["echo"]
    assert row["response_text"] == "checking"
    assert row["response_tool_calls"] == [{"id": "c1", "name": "echo", "arguments": {"message": "hi"}}]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (10, 5)
    assert row["purpose"] == "chat" and row["session_id"] == "s1"


async def test_provider_types_describe_core_entries():
    gateway = scripted_gateway(ScriptedAdapter())

    assert gateway.provider_types() == [{
        "type": "scripted", "label": None, "adapter": "scripted", "default_base_url": None,
        "supports_model_listing": True, "source": "core",
    }]
