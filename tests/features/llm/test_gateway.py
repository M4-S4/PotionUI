import asyncio
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from src.features.chat.conversation import ConversationRunner
from src.features.llm import context_budget
from src.features.llm.context_windows import ContextWindows
from src.features.llm.gateway import LLMGateway
from src.features.llm.harness.adapter import Settings, ToolSpec, WireRequest
from src.features.llm.harness.profiles import LayeredProfileResolver
from src.features.llm.harness.build import user_message
from src.features.llm.harness.errors import ConnectionFailed
from src.features.llm.harness.legacy import messages_from_dicts
from src.features.llm.repository import LLMConfig
from src.features.llm.tools.builtin.manage_prompts_tool import AddPromptTool
from src.features.llm.tools.builtin.update_form_settings_tool import UpdateFormSettingsTool
from src.features.llm.types import LLMType, LLMTypes
from tests.fixtures.llm_engine import ConfigRepository, drain
from tests.fixtures.llm_harness import text_step


def make_config(**overrides) -> LLMConfig:
    defaults = dict(
        id="cfg-1",
        name="Test Config",
        type="ollama",
        enabled=True,
        base_url="http://internal-secret-host:11434",
        model="llama3",
        system_message="You are helpful.",
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


class _Replay:
    def __init__(self, events):
        self._events = list(events)

    def feed(self, chunk):
        events, self._events = self._events, []
        return events

    def close(self):
        return []


class ProbeAdapter:
    name = "probe"

    def __init__(self, reply: str = "ok", error: Optional[BaseException] = None) -> None:
        self.reply = reply
        self.error = error
        self.requests: List[Any] = []
        self.markers: List[str] = []
        self.park = False

    def render(self, request, profile):
        self.requests.append(request)
        return WireRequest(method="POST", url="probe://", headers={}, body={})

    def decoder(self, request, profile):
        return _Replay(text_step(self.reply))

    async def send(self, wire, request):
        if self.error is not None:
            raise self.error
        try:
            yield b""
            if self.park:
                await asyncio.Event().wait()
        finally:
            self.markers.append("send_closed")

    def classify_error(self, failure):
        raise NotImplementedError

    async def discover(self, config):
        raise NotImplementedError

    async def list_models(self, base_url, api_key=None):
        return []


class CountingProbe(ProbeAdapter):
    def __init__(self, counter=None, messages_counter=None, messages_counter_error=None):
        super().__init__()
        self._counter = counter
        self._messages_counter = messages_counter
        self._error = messages_counter_error

    def token_counter(self, config):
        return self._counter

    def messages_token_counter(self, config):
        if self._error is not None:
            raise self._error
        return self._messages_counter


def gateway_for(adapter, *configs) -> LLMGateway:
    types = LLMTypes(core=tuple(LLMType(type=type_id, adapter=adapter) for type_id in ("ollama", "openai", "native")))
    windows = ContextWindows(types)
    return LLMGateway(ConfigRepository(*(configs or (make_config(),))), types, LayeredProfileResolver(types.adapter_for), windows)


def history(turns=6, size=20):
    entries = []
    for index in range(turns):
        entries.append({"role": "user", "content": f"turn {index} question " * size})
        entries.append({"role": "assistant", "content": f"turn {index} answer " * size})
    entries.append({"role": "user", "content": "current question"})
    return messages_from_dicts(entries)[0]


class RepliesProbe(ProbeAdapter):
    def __init__(self, *replies: str) -> None:
        super().__init__()
        self.replies = list(replies)

    def decoder(self, request, profile):
        return _Replay(text_step(self.replies.pop(0)))


class TestToolSchemaBudget:
    @pytest.mark.asyncio
    async def test_a_large_real_tool_schema_can_exceed_a_small_context_window(self):
        adapter = ProbeAdapter()
        gateway = gateway_for(adapter, make_config(provider_options={"context_window": 50}, max_tokens=10))
        tools = [ToolSpec.from_schema(tool.to_schema()) for tool in (AddPromptTool(), UpdateFormSettingsTool())]

        with pytest.raises(context_budget.ContextBudgetExceededError) as exc_info:
            await gateway.complete("cfg-1", "hi", tools=tools)

        assert exc_info.value.breakdown["tool_schema_tokens"] > 0
        assert exc_info.value.breakdown["measured"] is False
        assert adapter.requests == []


class TestCompleteEmptyRetry:
    @pytest.mark.asyncio
    async def test_empty_completions_are_retried_until_content(self):
        adapter = RepliesProbe("", "", "canned reply")

        completion = await gateway_for(adapter).complete("cfg-1", "hello")

        assert len(adapter.requests) == 3
        assert completion.text == "canned reply"

    @pytest.mark.asyncio
    async def test_gives_up_after_three_empty_completions(self):
        adapter = RepliesProbe("", "", "", "never")

        completion = await gateway_for(adapter).complete("cfg-1", "hello")

        assert len(adapter.requests) == 3
        assert completion.text == ""


class TestGatewayTestConfiguration:
    async def test_success_reports_response(self):
        adapter = ProbeAdapter(reply="Test successful!")
        config = make_config()

        result = await gateway_for(adapter, config).test_configuration(config)

        assert result == {"success": True, "response": "Test successful!", "model": "llama3", "tokens_used": 15}
        assert adapter.requests[0].messages[0].text().startswith("Hello, this is a test.")

    async def test_failure_does_not_leak_exception_detail(self):
        secret = "connection failed to http://internal-secret-host:11434 using key sk-ABC123XYZ"
        config = make_config()
        gateway = gateway_for(ProbeAdapter(error=ConnectionFailed(secret)), config)

        with patch("src.features.llm.gateway.logging") as mock_logging:
            result = await gateway.test_configuration(config)

        assert result["success"] is False
        assert secret not in result["error"] and "sk-ABC123XYZ" not in result["error"]
        assert "internal-secret-host" not in result["error"]
        _, kwargs = mock_logging.error.call_args
        assert kwargs.get("exc_info") is True


class TestMaxTokensClamp:
    async def test_known_window_clamps_max_tokens_sent(self):
        adapter = ProbeAdapter()
        config = make_config(provider_options={"context_window": 32768}, max_tokens=50000)
        gateway = gateway_for(adapter, config)

        await gateway.complete("cfg-1", [user_message("hi")])

        expected = gateway.estimate_context_budget(config, config.system_message, [user_message("hi")]).ledger["max_tokens_sent"]
        assert adapter.requests[0].settings.max_tokens == expected < 50000

    async def test_unknown_window_leaves_max_tokens_to_the_config(self):
        adapter = ProbeAdapter()

        await gateway_for(adapter, make_config(max_tokens=100_000)).complete("cfg-1", [user_message("hi")])

        assert adapter.requests[0].settings.max_tokens is None


class TestBudgetRunsBeforeTheAdapter:
    async def test_complete_and_stream_both_run_the_budget(self):
        gateway = gateway_for(ProbeAdapter())
        with patch.object(gateway, "estimate_context_budget", wraps=gateway.estimate_context_budget) as spy:
            await gateway.complete("cfg-1", [user_message("hello")])
            await drain(gateway.stream("cfg-1", [user_message("hello")]))
        assert spy.call_count == 2

    async def test_trims_before_the_adapter_sees_the_messages(self):
        adapter = ProbeAdapter()
        gateway = gateway_for(adapter, make_config(provider_options={"context_window": 200}, max_tokens=50))
        big = messages_from_dicts(
            [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 200} for i in range(10)]
            + [{"role": "user", "content": "current question"}]
        )[0]

        await gateway.complete("cfg-1", big)

        sent = adapter.requests[0].messages
        assert len(sent) < len(big)
        assert sent[-1].text() == "current question"

    async def test_an_irreducibly_oversized_request_raises_instead_of_sending(self):
        adapter = ProbeAdapter()
        gateway = gateway_for(adapter, make_config(provider_options={"context_window": 10}, max_tokens=5))

        with pytest.raises(context_budget.ContextBudgetExceededError):
            await gateway.complete("cfg-1", [user_message("x" * 500)])

        assert adapter.requests == []

    def test_reserve_tokens_follow_the_settings_max_tokens(self):
        gateway = gateway_for(ProbeAdapter())
        outcome = gateway.estimate_context_budget(
            make_config(), "", [user_message("hi")], settings=Settings(max_tokens=4096),
        )
        assert outcome.ledger["max_tokens"] == 4096
        assert outcome.ledger["reserve_tokens"] == 1229


class TestContextProfileFor:
    def test_bundles_capacity_reserve_counters_and_image_override(self):
        adapter = CountingProbe(counter=lambda text: len(text), messages_counter=lambda s, m, t: 99)
        config = make_config(
            type="native", provider_options={"context_window": 4096, "image_token_estimate": 200}, max_tokens=777,
        )

        inputs = gateway_for(adapter, config).context_profile_for(config)

        assert inputs.capacity == context_budget.CapacityInfo(4096, "config")
        assert inputs.reserve_tokens == 777
        assert inputs.counter("hi") == 2
        assert inputs.messages_counter(None, [], None) == 99
        assert inputs.image_tokens_override == 200

    def test_settings_max_tokens_win_over_the_config_default(self):
        config = make_config(max_tokens=2000, provider_options={"think": False})
        inputs = gateway_for(ProbeAdapter(), config).context_profile_for(config, Settings(max_tokens=4096))
        assert inputs.max_tokens == 4096
        assert inputs.reserve_tokens == 1229

    def test_a_raising_messages_counter_hook_degrades_to_none(self):
        config = make_config(type="native")
        adapter = CountingProbe(messages_counter_error=RuntimeError("boom"))
        assert gateway_for(adapter, config).context_profile_for(config).messages_counter is None

    def test_an_adapter_without_hooks_gets_no_counters(self):
        inputs = gateway_for(ProbeAdapter()).context_profile_for(make_config())
        assert inputs.counter is None and inputs.messages_counter is None

    def test_thinking_adds_a_reserve_allowance(self):
        gateway = gateway_for(ProbeAdapter())
        off = gateway.context_profile_for(make_config(provider_options={"context_window": 32768, "think": False}, max_tokens=1000))
        on = gateway.context_profile_for(make_config(provider_options={"context_window": 32768, "think": True}, max_tokens=1000))
        high = gateway.context_profile_for(make_config(provider_options={"context_window": 32768, "think": "high"}, max_tokens=1000))
        unset = gateway.context_profile_for(make_config(provider_options={"context_window": 32768, "think": None}, max_tokens=1000))

        assert (off.thinking_enabled, on.thinking_enabled, high.thinking_enabled, unset.thinking_enabled) == (False, True, True, False)
        assert on.reserve_tokens == off.reserve_tokens + on.thinking_allowance_tokens
        assert unset.thinking_allowance_tokens == 0

    def test_settings_can_turn_thinking_off_for_one_call(self):
        config = make_config(provider_options={"context_window": 32768, "think": True}, max_tokens=1000)
        assert gateway_for(ProbeAdapter()).context_profile_for(config, Settings(reasoning=False)).thinking_enabled is False


class TestReserveClamp:
    async def test_a_configured_window_with_large_max_tokens_still_sends_prior_turns(self):
        adapter = ProbeAdapter()
        gateway = gateway_for(adapter, make_config(provider_options={"context_window": 32768}, max_tokens=50000))

        await gateway.complete("cfg-1", history())

        sent = adapter.requests[0].messages
        assert len(sent) > 1 and sent[-1].text() == "current question"

    def test_reserve_is_clamped_on_known_and_unknown_windows(self):
        gateway = gateway_for(ProbeAdapter())
        known = gateway.estimate_context_budget(
            make_config(provider_options={"context_window": 32768, "think": False}, max_tokens=50000), "sys", [user_message("hi")],
        )
        unknown = gateway.estimate_context_budget(
            make_config(provider_options={"think": False}, max_tokens=100_000), "sys", [user_message("hi")],
        )
        assert known.ledger["reserve_tokens"] == 4916
        assert unknown.ledger["capacity_source"] == "unknown" and unknown.ledger["reserve_tokens"] == 1229

    def test_the_preflight_ledger_and_the_real_send_agree(self):
        config = make_config(provider_options={"context_window": 32768}, max_tokens=50000)
        gateway = gateway_for(ProbeAdapter(), config)
        messages = [user_message("hi")]
        send_outcome = gateway.estimate_context_budget(config, "", messages)

        manager = type("Manager", (), {"llm_service": gateway})()
        runner = ConversationRunner(manager)
        session = type("Session", (), {"llm_config_id": "cfg-1"})()
        preflight = ConversationRunner._build_context_ledger(
            "", [], {}, messages, accounting=runner._resolve_budget_inputs(session, Settings()),
        )

        for key in ("reserve_tokens", "max_tokens", "max_tokens_sent", "thinking_enabled"):
            assert preflight["budget"][key] == send_outcome.ledger[key]


class TestSentMaxTokensNeverExceedsWindow:
    WINDOW = 32768

    @pytest.mark.parametrize("max_tokens", [8192, 30000, 50000])
    @pytest.mark.parametrize("with_tools", [False, True])
    async def test_prompt_estimate_plus_sent_max_tokens_fits_the_window(self, max_tokens, with_tools):
        adapter = ProbeAdapter()
        config = make_config(provider_options={"context_window": self.WINDOW}, max_tokens=max_tokens)
        tools = (ToolSpec(name="noop", description="", parameters={}),) if with_tools else ()

        await gateway_for(adapter, config).complete("cfg-1", history(turns=20, size=200), tools=tools)

        request = adapter.requests[0]
        sent = request.settings.max_tokens or config.max_tokens
        estimated = (
            context_budget.count_text(request.system, None).tokens
            + context_budget.count_messages(request.messages, None).tokens
            + context_budget.count_tool_schemas([tool.function_schema() for tool in tools] or None, None).tokens
        )
        assert estimated + sent <= self.WINDOW
        assert sent <= max_tokens


class TestImageTokenOverride:
    @pytest.mark.parametrize("override, expected", [({"image_token_estimate": 5}, 5), ({}, None)])
    async def test_the_override_reaches_the_enforced_budget(self, override, expected):
        config = make_config(provider_options={"context_window": 32768, **override})
        gateway = gateway_for(ProbeAdapter(), config)

        with patch("src.features.llm.context_budget.enforce_budget", wraps=context_budget.enforce_budget) as spy:
            await gateway.complete("cfg-1", [user_message("look", images=("aW1n",))])

        assert spy.call_args.kwargs["image_tokens"] == expected
        assert spy.call_args.kwargs["image_attached"] is True


class TestStreamClosePropagation:
    async def test_closing_the_stream_closes_the_adapter_at_once(self):
        adapter = ProbeAdapter()
        adapter.park = True
        gateway = gateway_for(adapter)

        events = gateway.stream("cfg-1", [user_message("hi")])
        await events.__anext__()
        assert adapter.markers == []

        await events.aclose()

        assert adapter.markers == ["send_closed"]
