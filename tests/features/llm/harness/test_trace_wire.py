from dataclasses import replace
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import pytest

from src.features.llm import trace_collector
from src.features.llm.harness.adapter import RenderRequest, ToolSpec, WireRequest
from src.features.llm.harness.build import user_message
from src.features.llm.harness.errors import MalformedStream, Overloaded
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, ToolArgsDelta, Usage, UsageReported
from src.features.llm.harness.profile import CapabilityProfile, Prices
from tests.fixtures.llm_engine import drain, scripted_gateway
from tests.fixtures.llm_harness import ScriptedAdapter, text_step


class Recorder:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def record(self, **row: Any) -> None:
        self.rows.append(row)


class WireAdapter(ScriptedAdapter):
    def __init__(
        self,
        events: Sequence[Any],
        chunks: Sequence[bytes],
        *,
        name: str = "scripted",
        fail_after: Optional[Exception] = None,
    ) -> None:
        super().__init__(events)
        self.name = name
        self._chunks = list(chunks)
        self._fail_after = fail_after

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        super().render(request, profile)
        body = {
            "model": request.config.model,
            "stream": True,
            "messages": [{"role": "system", "content": request.system}, {"role": "user", "content": "hi"}],
            "tools": [{"name": tool.name} for tool in request.tools],
            "temperature": 0.2,
        }
        return WireRequest(method="POST", url="scripted://wire", headers={}, body=body)

    async def send(self, wire: WireRequest, request: RenderRequest) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._fail_after is not None:
            raise self._fail_after


class PricedResolver:
    def __init__(self, inner: Any, prices: Prices) -> None:
        self._inner = inner
        self._prices = prices

    def resolve(self, config: Any) -> Any:
        return replace(self._inner.resolve(config), prices=self._prices)

    async def refresh(self, config: Any) -> Any:
        return self.resolve(config)


@pytest.fixture
def recorder():
    recorder = Recorder()
    previous = trace_collector._recorder
    trace_collector.set_recorder(recorder)
    yield recorder
    trace_collector.set_recorder(previous)


def answer(text: str, usage: Optional[Usage] = None) -> List[Any]:
    return [
        TextDelta(block=1, text=text),
        UsageReported(usage or Usage(input_tokens=1000, output_tokens=100, total_tokens=1100)),
        Finish(reason=FinishReason.STOP, raw_reason="stop"),
    ]


async def traced(gateway: Any, **kwargs: Any) -> None:
    with trace_collector.activate("s1", "u1", purpose="chat"):
        await gateway.complete("cfg-scripted", [user_message("hi")], **kwargs)


async def test_a_step_is_traced_with_the_wire_body_the_raw_response_the_usage_and_the_cost(recorder):
    adapter = WireAdapter(answer("Hello"), [b"data: {\"t\":\"Hel\"}\n\n", b"data: {\"t\":\"lo\"}\n\n"])
    gateway = scripted_gateway(adapter)
    gateway.profiles = PricedResolver(gateway.profiles, Prices(input=2, output=10))

    await traced(gateway, tools=[ToolSpec(name="echo", description="Echo", parameters={"type": "object"})])

    row = recorder.rows[0]
    assert row["adapter"] == "scripted"
    assert row["request_body"] == {
        "model": "scripted-model", "stream": True, "temperature": 0.2,
        "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}],
        "tools": [{"name": "echo"}],
    }
    assert row["response_raw"] == 'data: {"t":"Hel"}\n\ndata: {"t":"lo"}\n\n'
    assert row["usage"] == {
        "input_tokens": 1000, "output_tokens": 100, "reasoning_tokens": None,
        "cached_input_tokens": None, "cache_write_tokens": None, "cost": 0.003,
    }
    assert row["cost"] == 0.003
    assert row["error"] is None
    assert (row["prompt_tokens"], row["completion_tokens"]) == (1000, 100)
    assert row["response_text"] == "Hello"


async def test_the_raw_response_keeps_the_reasoning_the_text_leaves_out(recorder):
    chunks = [b'{"thinking":"let me check"}\n', b'{"text":"Done."}\n']
    adapter = WireAdapter(answer("Done."), chunks)

    await traced(scripted_gateway(adapter))

    row = recorder.rows[0]
    assert "let me check" in row["response_raw"]
    assert row["response_text"] == "Done."


async def test_a_step_without_wire_chunks_still_records_its_request_and_no_raw_response(recorder):
    await traced(scripted_gateway(ScriptedAdapter(text_step("hi"))))

    row = recorder.rows[0]
    assert row["response_raw"] is None and row["request_body"] == {"step": 0}


async def test_a_local_runtime_that_streams_plain_text_is_traced_like_any_other_adapter(recorder):
    chunks = ["<think>plan</think>".encode(), "The answer.".encode()]
    adapter = WireAdapter(answer("The answer."), chunks, name="native")

    await traced(scripted_gateway(adapter))

    row = recorder.rows[0]
    assert row["adapter"] == "native"
    assert row["response_raw"] == "<think>plan</think>The answer."


async def test_a_provider_error_midway_is_traced_with_what_arrived_and_the_error(recorder):
    adapter = WireAdapter([], [b"partial"], fail_after=Overloaded("busy", status=529))
    gateway = scripted_gateway(adapter)

    with pytest.raises(Overloaded):
        await traced(gateway)

    row = recorder.rows[0]
    assert row["response_raw"] == "partial"
    assert row["error"]["kind"] == "overloaded" and row["error"]["status"] == 529


async def test_an_http_error_is_traced_with_the_error_body(recorder):
    error = Overloaded("busy", status=529, detail={"body": '{"error":{"message":"busy"}}'})
    adapter = ScriptedAdapter(error)

    with pytest.raises(Overloaded):
        await traced(scripted_gateway(adapter))

    row = recorder.rows[0]
    assert row["response_raw"] == '{"error":{"message":"busy"}}'
    assert row["error"]["kind"] == "overloaded"
    assert row["cost"] is None and row["response_text"] == ""


async def test_a_stream_closed_early_is_traced_as_cancelled(recorder):
    adapter = WireAdapter([TextDelta(block=1, text="a"), TextDelta(block=1, text="b")], [b"x"])
    gateway = scripted_gateway(adapter)

    with trace_collector.activate("s1", "u1"):
        stream = gateway.stream("cfg-scripted", [user_message("hi")])
        async for _ in stream:
            break
        await stream.aclose()

    assert len(recorder.rows) == 1
    assert recorder.rows[0]["error"]["kind"] == "cancelled"


async def test_a_malformed_stream_is_traced_and_still_raises(recorder):
    adapter = WireAdapter([ToolArgsDelta(block=7, text="{}")], [b"?"])

    with pytest.raises(MalformedStream):
        await traced(scripted_gateway(adapter))

    row = recorder.rows[0]
    assert row["error"]["kind"] == "malformed_stream"
    assert row["response_raw"] == "?"


async def test_every_retry_attempt_is_a_traced_call_of_its_own(recorder):
    adapter = ScriptedAdapter(Overloaded("busy"), text_step("ok"))
    gateway = scripted_gateway(adapter)

    with trace_collector.activate("s1", "u1"):
        with pytest.raises(Overloaded):
            await gateway.complete("cfg-scripted", [user_message("hi")])
        await gateway.complete("cfg-scripted", [user_message("hi")])

    assert [row["error"] is not None for row in recorder.rows] == [True, False]


async def test_nothing_is_gathered_or_recorded_outside_a_trace_context(recorder):
    adapter = WireAdapter(answer("hi"), [b"raw"])

    await scripted_gateway(adapter).complete("cfg-scripted", [user_message("hi")])

    assert recorder.rows == []


def test_the_response_capture_is_bounded():
    capture = trace_collector.RawCapture(limit=10)

    capture.add(b"12345")
    capture.add(b"6789012345")
    capture.add(b"more")

    assert capture.text() == "1234567890"


def test_a_multibyte_character_cut_by_the_capture_limit_does_not_break_the_text():
    capture = trace_collector.RawCapture(limit=4)

    capture.add("aé€".encode())

    assert capture.text().startswith("aé")


async def test_a_completion_reports_the_priced_usage():
    gateway = scripted_gateway(ScriptedAdapter(text_step("hi")))
    gateway.profiles = PricedResolver(gateway.profiles, Prices(input=2, output=10))

    completion = await gateway.complete("cfg-scripted", [user_message("hi")])

    assert completion.usage.cost == 0.00007


async def test_a_trace_recorder_that_fails_cannot_break_the_call(recorder):
    class Broken:
        def record(self, **row: Any) -> None:
            raise RuntimeError("db down")

    trace_collector.set_recorder(Broken())
    adapter = WireAdapter(answer("hi"), [b"raw"])

    await traced(scripted_gateway(adapter))


async def test_release_calls_the_adapter_hook_and_is_a_noop_without_one():
    class Releasing(ScriptedAdapter):
        released: List[str] = []

        async def release(self, config: Any) -> None:
            self.released.append(config.id)

    releasing = Releasing()
    await scripted_gateway(releasing).release("cfg-scripted")
    await scripted_gateway(ScriptedAdapter()).release("cfg-scripted")

    assert releasing.released == ["cfg-scripted"]
