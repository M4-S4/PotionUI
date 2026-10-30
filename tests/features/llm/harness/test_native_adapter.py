import json
from types import SimpleNamespace

import pytest

from src.features.llm.harness.adapter import Adapter, RenderRequest, Settings, ToolSpec
from src.features.llm.harness.adapters.native import (
    NativeAdapter,
    NativeDecoder,
    render_chat,
    render_sampling,
    requested_thinking,
)
from src.features.llm.harness.errors import HarnessError, InvalidRequest, WireFailure
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    TextDelta,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import (
    ImagePart,
    Message,
    Origin,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from src.features.llm.harness.profile import CapabilityProfile, ProfileSource
from src.features.llm.native.runtime import NativeOutOfMemory, RunStats
from tests.features.llm.harness.history import IMAGE_B64, make_config
from tests.fixtures.llm_harness import decode_chunks

ORIGIN = Origin(adapter="native", provider="native", model="tiny")
PROFILE = CapabilityProfile(adapter="native", model="tiny", prompt_tools=True)
TOOL = ToolSpec(
    name="set_value",
    description="sets a value",
    parameters={
        "type": "object",
        "properties": {"key": {"type": "string"}, "amount": {"type": "integer"}},
        "required": ["key", "amount"],
    },
)


def user(text, *extra):
    return Message(id="u", role=Role.USER, parts=(TextPart(text), *extra))


def assistant(text, *calls):
    return Message(id="a", role=Role.ASSISTANT, parts=(TextPart(text), *calls), origin=ORIGIN)


def tool_result(call_id, name, text):
    return Message(
        id="t", role=Role.TOOL, parts=(ToolResultPart(call_id=call_id, name=name, content=(TextPart(text),)),)
    )


def request(messages, tools=(), settings=None, **config):
    return RenderRequest(
        config=make_config("native", model="tiny", **config),
        system="SYS",
        messages=tuple(messages),
        tools=tuple(tools),
        settings=settings or Settings(),
    )


def test_the_adapter_satisfies_the_adapter_protocol():
    assert isinstance(NativeAdapter(), Adapter)
    assert NativeAdapter.name == "native"


def test_render_builds_the_chat_with_the_system_message_first():
    wire = NativeAdapter().render(request([user("hi")]), PROFILE)

    assert wire.body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    assert wire.body["model"] == "tiny"
    assert json.loads(wire.content())["messages"][0]["content"] == "SYS"


def test_render_inlines_the_tools_into_the_system_message():
    wire = NativeAdapter().render(request([user("hi")], tools=[TOOL]), PROFILE)

    system = wire.body["messages"][0]["content"]
    assert system.startswith("SYS")
    assert "set_value" in system
    assert '"name": "set_value"' in system
    assert '"key"' in system and '"amount"' in system


def test_render_shows_the_model_its_own_tool_calls_and_their_results():
    call = ToolCallPart(call_id="c1", name="set_value", args={"key": "k", "amount": 2})
    history = [user("go"), assistant("", call), tool_result("c1", "set_value", "done"), user("thanks")]

    chat = render_chat("SYS", history, [TOOL])

    assert chat[2] == {
        "role": "assistant",
        "content": '<tool_call>{"name": "set_value", "arguments": {"key": "k", "amount": 2}}</tool_call>',
    }
    assert chat[3] == {"role": "user", "content": "[Tool Result: set_value]\ndone"}
    assert chat[4] == {"role": "user", "content": "thanks"}


def test_render_replays_tool_history_even_when_no_tools_are_offered():
    call = ToolCallPart(call_id="c1", name="set_value", args={})
    chat = render_chat(None, [user("go"), assistant("ok", call), tool_result("c1", "set_value", "done")], [])

    assert chat[0] == {"role": "user", "content": "go"}
    assert chat[1]["content"].startswith("ok\n<tool_call>")
    assert chat[2]["content"].startswith("[Tool Result: set_value]")


def test_render_attaches_the_selected_image_to_the_last_user_message():
    chat = render_chat("SYS", [user("look", ImagePart(media_type="image/png", data=IMAGE_B64))], [])

    assert chat[-1]["images"] == [IMAGE_B64]
    assert "images" not in chat[0]


def test_render_without_images_leaves_unresolved_references_alone():
    chat = render_chat("SYS", [user("look", ImagePart(media_type="image/png", ref="media://1"))], [], with_image=False)

    assert "images" not in chat[-1]


def test_sampling_reads_request_settings_before_config_and_options():
    config = dict(temperature=0.9, max_tokens=100, provider_options={"top_p": 0.5, "top_k": 9, "repetition_penalty": 1.2})
    settings = Settings(temperature=0.1, top_k=3, max_tokens=7)

    sampling = render_sampling(request([user("hi")], settings=settings, **config), PROFILE)

    assert sampling == {
        "temperature": 0.1,
        "top_p": 0.5,
        "top_k": 3,
        "repetition_penalty": 1.2,
        "max_new_tokens": 7,
    }


def test_sampling_drops_unset_values_and_falls_back_to_the_profile_output_limit():
    config = make_config("native", model="tiny").model_copy(update={"temperature": None, "max_tokens": None})
    unset = RenderRequest(config=config, system="", messages=(user("hi"),))

    assert render_sampling(unset, CapabilityProfile(adapter="native", model="tiny", max_output_tokens=33)) == {
        "max_new_tokens": 33
    }
    assert render_sampling(unset, PROFILE) == {}


@pytest.mark.parametrize(
    "reasoning,option,expected",
    [(None, None, None), (None, False, False), (None, True, True), (True, False, True), ("high", None, True), ("none", None, False)],
)
def test_thinking_follows_settings_then_the_provider_option(reasoning, option, expected):
    options = {} if option is None else {"thinking": option}
    config = make_config("native", provider_options=options)

    assert requested_thinking(Settings(reasoning=reasoning), config) is expected


def stats(**fields):
    return RunStats(**fields)


def test_the_decoder_splits_think_tags_into_reasoning_and_text():
    decoder = NativeDecoder(stats(prompt_tokens=5, completion_tokens=3, finish="stop"))

    events = decode_chunks(decoder, [b"<think>plan", b"</think>answer"])

    assert [type(event) for event in events if isinstance(event, (ReasoningDelta, TextDelta))] == [
        ReasoningDelta,
        TextDelta,
    ]
    assert "".join(event.text for event in events if isinstance(event, ReasoningDelta)) == "plan"
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "answer"


def test_the_decoder_keeps_a_multibyte_character_split_across_chunks():
    raw = "café".encode("utf-8")
    decoder = NativeDecoder(stats())

    events = decode_chunks(decoder, [raw[:4], raw[4:]])

    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "café"


@pytest.mark.parametrize(
    "finish,reason",
    [("stop", FinishReason.STOP), ("length", FinishReason.LENGTH), (None, FinishReason.UNKNOWN)],
)
def test_the_decoder_reports_the_finish_reason_and_the_token_counts(finish, reason):
    decoder = NativeDecoder(
        stats(prompt_tokens=5, completion_tokens=3, finish=finish, thinking_mode={"requested": True, "effective": True})
    )

    events = decode_chunks(decoder, [b"x"])

    usage = next(event for event in events if isinstance(event, UsageReported)).usage
    assert usage == Usage(
        input_tokens=5, output_tokens=3, total_tokens=8, raw={"thinking_mode": {"requested": True, "effective": True}}
    )
    assert events[-1] == Finish(reason=reason, raw_reason=finish)


def test_the_decoder_omits_usage_when_the_runtime_reported_no_counts():
    events = decode_chunks(NativeDecoder(stats()), [b"x"])

    assert not any(isinstance(event, UsageReported) for event in events)


def test_a_tool_call_in_the_text_stays_text_for_the_engine_to_parse():
    block = '<tool_call>{"name": "set_value", "arguments": {}}</tool_call>'

    events = decode_chunks(NativeDecoder(stats()), [block.encode("utf-8")])

    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == block


class FakeRuntime:
    def __init__(self, texts=(), error=None):
        self.texts = texts
        self.error = error
        self.calls = []

    async def stream(self, config, messages, sampling, enable_thinking, stats):
        self.calls.append((config.model, messages, sampling, enable_thinking))
        stats.prompt_tokens = 4
        stats.completion_tokens = 2
        stats.finish = "stop"
        for text in self.texts:
            yield text
        if self.error is not None:
            raise self.error


async def run(adapter, req):
    wire = adapter.render(req, PROFILE)
    decoder = adapter.decoder(req, PROFILE)
    events = []
    async for chunk in adapter.send(wire, req):
        events.extend(decoder.feed(chunk))
    events.extend(decoder.close())
    return events


@pytest.mark.asyncio
async def test_send_streams_the_runtime_text_and_the_decoder_sees_its_counts():
    adapter = NativeAdapter()
    adapter.runtime = FakeRuntime(texts=("he", "llo"))
    req = request([user("hi")], settings=Settings(reasoning=False))

    events = await run(adapter, req)

    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "hello"
    assert next(event for event in events if isinstance(event, UsageReported)).usage.total_tokens == 6
    assert events[-1].reason is FinishReason.STOP
    model, messages, sampling, thinking = adapter.runtime.calls[0]
    assert (model, thinking) == ("tiny", False)
    assert messages[-1] == {"role": "user", "content": "hi"}
    assert sampling["max_new_tokens"] == 512


@pytest.mark.asyncio
async def test_send_turns_out_of_memory_into_a_typed_error():
    adapter = NativeAdapter()
    adapter.runtime = FakeRuntime(error=NativeOutOfMemory("ran out of GPU memory"))

    with pytest.raises(HarnessError) as raised:
        await run(adapter, request([user("hi")]))

    assert type(raised.value) is HarnessError
    assert raised.value.code == "out_of_memory"
    assert not raised.value.retryable
    assert "ran out of GPU memory" in raised.value.message


@pytest.mark.asyncio
async def test_send_turns_a_configuration_error_into_an_invalid_request():
    adapter = NativeAdapter()
    adapter.runtime = FakeRuntime(error=ValueError("Native LLM provider: unsupported model family"))

    with pytest.raises(InvalidRequest, match="unsupported model family"):
        await run(adapter, request([user("hi")]))


def test_classify_error_names_the_cause():
    adapter = NativeAdapter()

    assert adapter.classify_error(WireFailure(status=None, cause=NativeOutOfMemory("oom"))).code == "out_of_memory"
    assert isinstance(adapter.classify_error(WireFailure(status=None, cause=ValueError("bad"))), InvalidRequest)
    other = adapter.classify_error(WireFailure(status=None, cause=KeyError("x")))
    assert other.code == "KeyError"


@pytest.mark.asyncio
async def test_discover_reports_prompt_tools_and_the_checkpoint_window(tmp_path):
    checkpoint = tmp_path / "llm" / "tiny"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "max_position_embeddings": 4096}), encoding="utf-8"
    )
    from tests.fixtures.model_roots import make_roots

    adapter = NativeAdapter(model_roots=make_roots(tmp_path, types=["llm"], home_dir=tmp_path))

    layer = await adapter.discover(make_config("native", model="tiny"))

    assert layer.source is ProfileSource.DISCOVERED
    assert layer.fields == {"prompt_tools": True, "context_window": 4096, "vision": False}


@pytest.mark.asyncio
async def test_discover_without_a_resolvable_checkpoint_still_reports_prompt_tools():
    layer = await NativeAdapter().discover(make_config("native", model="missing"))

    assert layer.fields == {"prompt_tools": True}


@pytest.mark.asyncio
async def test_list_models_lists_the_supported_checkpoints(tmp_path):
    for name, model_type in (("alpha", "qwen3"), ("beta", "totally-unknown")):
        directory = tmp_path / "llm" / name
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    from tests.fixtures.model_roots import make_roots

    adapter = NativeAdapter(model_roots=make_roots(tmp_path, types=["llm"], home_dir=tmp_path))

    models = await adapter.list_models("")

    assert [model.id for model in models] == ["alpha"]
    assert models[0].details == {"model_type": "qwen3", "vision": False}


@pytest.mark.asyncio
async def test_release_delegates_to_the_runtime():
    adapter = NativeAdapter()
    seen = []

    async def release(config):
        seen.append(config.model)
        return True

    adapter.runtime = SimpleNamespace(release=release)

    assert await adapter.release(make_config("native", model="tiny")) is True
    assert seen == ["tiny"]
