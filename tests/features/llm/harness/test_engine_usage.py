import asyncio
from dataclasses import replace
from typing import Any, List, Sequence

import pytest

from src.features.llm.harness import engine as engine_module
from src.features.llm.harness.build import user_message
from src.features.llm.harness.engine import TurnInput, TurnOutcome
from src.features.llm.harness.errors import Overloaded
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, Usage, UsageReported
from src.features.llm.harness.messages import Role
from src.features.llm.harness.profile import Prices
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from tests.fixtures.llm_engine import drain, scripted_engine, tool_context
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes a message back."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=f"echo: {kwargs.get('message', '')}")


class DeleteTool(BaseTool):
    name = "delete_prompt"
    description = "Deletes a prompt after approval."
    parameters = {"type": "object", "properties": {"id": {"type": "string"}}}
    requires_approval = True

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Delete", target=kwargs.get("id")))


class PricedResolver:
    def __init__(self, inner: Any, prices: Prices) -> None:
        self._inner = inner
        self._prices = prices

    def resolve(self, config: Any) -> Any:
        return replace(self._inner.resolve(config), prices=self._prices)

    async def refresh(self, config: Any) -> Any:
        return self.resolve(config)


class ReleasingAdapter(ScriptedAdapter):
    def __init__(self, *steps: Any, fail: bool = False, hang: bool = False) -> None:
        super().__init__(*steps)
        self.released: List[str] = []
        self._fail = fail
        self._hang = hang

    async def release(self, config: Any) -> None:
        self.released.append(config.id)
        if self._fail:
            raise RuntimeError("server gone")
        if self._hang:
            await asyncio.Event().wait()


def turn(**overrides: Any) -> TurnInput:
    values: dict = dict(
        session_id="s1",
        llm_id="cfg-scripted",
        system="SYS",
        history=(),
        new_messages=(user_message("find cats", message_id="u1"),),
        tool_context=tool_context(),
    )
    values.update(overrides)
    return TurnInput(**values)


def outcome_of(events: Sequence[Any]) -> TurnOutcome:
    return next(event for event in events if isinstance(event, TurnOutcome))


def with_usage(events: Sequence[Any], usage: Usage) -> List[Any]:
    return [UsageReported(usage) if isinstance(event, UsageReported) else event for event in events]


def model_steps(rig: Any) -> List[Any]:
    return [step for step in rig.store.steps("s1") if step.message.role is Role.ASSISTANT]


def priced_rig(adapter: Any, prices: Prices, tools: Sequence[BaseTool] = ()) -> Any:
    rig = scripted_engine(adapter, tools)
    rig.gateway.profiles = PricedResolver(rig.gateway.profiles, prices)
    return rig


FIRST = Usage(input_tokens=1000, output_tokens=50, total_tokens=1050)
SECOND = Usage(input_tokens=1200, output_tokens=80, total_tokens=1280, cached_input_tokens=1000)
THIRD = Usage(input_tokens=1400, output_tokens=30, total_tokens=1430, cached_input_tokens=1200, reasoning_tokens=10)


def three_step_script() -> ScriptedAdapter:
    return ScriptedAdapter(
        with_usage(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), FIRST),
        with_usage(tool_step({"id": "c2", "name": "echo", "args": {"message": "b"}}), SECOND),
        with_usage(text_step("done"), THIRD),
    )


async def test_the_turn_usage_adds_up_every_step_and_each_step_keeps_its_own():
    rig = scripted_engine(three_step_script(), [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    assert [step.usage for step in model_steps(rig)] == [FIRST, SECOND, THIRD]
    assert outcome.usage == Usage(
        input_tokens=3600, output_tokens=160, total_tokens=3760, reasoning_tokens=10, cached_input_tokens=2200,
    )
    assert outcome.last_usage == THIRD


async def test_each_step_and_the_turn_carry_a_cost_priced_from_the_profile():
    rig = priced_rig(three_step_script(), Prices(input=2, output=10, cached_input=0.2), [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    assert [step.usage.cost for step in model_steps(rig)] == [0.0025, 0.0014, 0.00094]
    assert outcome.usage.cost == 0.00484
    assert outcome.last_usage.cost == 0.00094


async def test_a_cost_reported_by_the_provider_is_kept_and_summed():
    adapter = ScriptedAdapter(
        with_usage(tool_step({"id": "c1", "name": "echo", "args": {}}), Usage(input_tokens=10, output_tokens=5, cost=0.5)),
        with_usage(text_step("done"), Usage(input_tokens=10, output_tokens=5, cost=0.25)),
    )
    rig = priced_rig(adapter, Prices(input=2, output=10), [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    assert [step.usage.cost for step in model_steps(rig)] == [0.5, 0.25]
    assert outcome.usage.cost == 0.75


async def test_a_steered_step_is_recorded_with_its_usage():
    truncated = [
        TextDelta(block=1, text='Working <tool_call>{"name": "echo", "arguments": {"mess'),
        UsageReported(FIRST),
        Finish(reason=FinishReason.LENGTH, raw_reason="length"),
    ]
    adapter = ScriptedAdapter(truncated, with_usage(text_step("Recovered."), SECOND))
    rig = scripted_engine(adapter, [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    steps = model_steps(rig)
    assert [step.usage for step in steps] == [FIRST, SECOND]
    assert steps[0].finish is FinishReason.LENGTH
    assert outcome.usage.input_tokens == 2200


async def test_a_discarded_empty_attempt_still_counts_toward_the_turn_usage():
    empty = [UsageReported(Usage(input_tokens=500, output_tokens=40, total_tokens=540)), Finish(reason=FinishReason.STOP)]
    adapter = ScriptedAdapter(empty, with_usage(text_step("second try"), SECOND))
    rig = scripted_engine(adapter, [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    assert [step.usage for step in model_steps(rig)] == [SECOND]
    assert outcome.usage.input_tokens == 1700 and outcome.usage.output_tokens == 120


async def test_the_thinking_mode_comes_from_the_last_step_not_the_total():
    mode = {"mode": "forced_off"}
    adapter = ScriptedAdapter(
        with_usage(tool_step({"id": "c1", "name": "echo", "args": {}}), Usage(input_tokens=1, output_tokens=1, raw={"thinking_mode": {"mode": "on"}})),
        with_usage(text_step("done"), Usage(input_tokens=1, output_tokens=1, raw={"thinking_mode": mode})),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    assert outcome.thinking_mode == mode


async def test_a_turn_with_no_reported_usage_has_none():
    adapter = ScriptedAdapter([TextDelta(block=1, text="hi"), Finish(reason=FinishReason.STOP)])
    rig = scripted_engine(adapter)

    outcome = outcome_of(await drain(rig.engine.run(turn(tool_context=None))))

    assert outcome.usage is None and outcome.last_usage is None


async def test_steps_that_may_be_followed_by_another_keep_the_model_loaded_and_the_last_one_does_not():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {}}),
        text_step("out of budget"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn(max_rounds=1)))

    assert [request.settings.keep_model_loaded for request in adapter.requests] == [True, False]


async def test_an_answer_without_tools_does_not_keep_the_model_loaded():
    adapter = ScriptedAdapter(text_step("hello"))
    rig = scripted_engine(adapter)

    await drain(rig.engine.run(turn(tool_context=None)))

    assert adapter.requests[0].settings.keep_model_loaded is False


async def test_the_caller_settings_survive_and_the_cache_key_defaults_to_the_session():
    from src.features.llm.harness.adapter import Settings

    adapter = ScriptedAdapter(text_step("a"), text_step("b"))
    rig = scripted_engine(adapter)

    await drain(rig.engine.run(turn(tool_context=None, settings=Settings(temperature=0.3))))
    await drain(rig.engine.run(turn(tool_context=None, session_id="s2", settings=Settings(cache_key="mine"))))

    assert adapter.requests[0].settings.temperature == 0.3
    assert [request.settings.cache_key for request in adapter.requests] == ["s1", "mine"]


async def test_the_model_is_released_once_a_turn_completes():
    adapter = ReleasingAdapter(tool_step({"id": "c1", "name": "echo", "args": {}}), text_step("done"))
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn()))

    assert adapter.released == ["cfg-scripted"]


async def test_the_model_is_released_while_a_turn_waits_for_approval():
    adapter = ReleasingAdapter(tool_step({"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}}))
    rig = scripted_engine(adapter, [DeleteTool()])

    events = await drain(rig.engine.run(turn()))

    assert outcome_of(events).pending
    assert adapter.released == ["cfg-scripted"]


async def test_the_model_is_released_when_the_provider_fails():
    adapter = ReleasingAdapter(Overloaded("busy"))
    rig = scripted_engine(adapter)

    with pytest.raises(Overloaded):
        await drain(rig.engine.run(turn(tool_context=None)))

    assert adapter.released == ["cfg-scripted"]


async def test_the_model_is_released_when_the_stream_is_closed_midway():
    adapter = ReleasingAdapter([TextDelta(block=1, text="partial"), Finish(reason=FinishReason.STOP)])
    rig = scripted_engine(adapter)
    events = rig.engine.run(turn(tool_context=None))

    async for event in events:
        if isinstance(event, TextDelta):
            break
    await events.aclose()

    assert adapter.released == ["cfg-scripted"]


async def test_a_failing_release_does_not_fail_the_turn():
    adapter = ReleasingAdapter(text_step("fine"), fail=True)
    rig = scripted_engine(adapter)

    outcome = outcome_of(await drain(rig.engine.run(turn(tool_context=None))))

    assert outcome.content == "fine"
    assert adapter.released == ["cfg-scripted"]


async def test_a_hanging_release_is_cut_off_instead_of_holding_the_turn_open(monkeypatch):
    monkeypatch.setattr(engine_module, "RELEASE_TIMEOUT_SECONDS", 0.01)
    adapter = ReleasingAdapter(text_step("fine"), hang=True)
    rig = scripted_engine(adapter)

    outcome = outcome_of(await drain(rig.engine.run(turn(tool_context=None))))

    assert outcome.content == "fine"
    assert adapter.released == ["cfg-scripted"]
