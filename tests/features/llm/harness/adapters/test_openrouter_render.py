from copy import deepcopy

import pytest

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.adapters.openrouter import ADAPTER_NAME, ATTRIBUTION_HEADERS
from src.features.llm.harness.errors import OpaqueViolation
from src.features.llm.harness.messages import Message, ReasoningPart
from tests.features.llm.harness.adapters.openrouter_support import (
    ADAPTER,
    BASE_URL,
    ENCRYPTED,
    SUMMARY,
    TEXT_BLOCK,
    lookup_tool,
    make_config,
    make_profile,
    make_request,
    reasoned_assistant,
    tool_results,
    user,
)

DETAILS = [ENCRYPTED, TEXT_BLOCK, SUMMARY]


def render(messages=(), *, config=None, settings=None, tools=(), **profile):
    request = make_request(messages, config=config, settings=settings, tools=tools)
    return ADAPTER.render(request, make_profile(**profile))


def assistant_entry(wire):
    return next(entry for entry in wire.body["messages"] if entry["role"] == "assistant")


def history(**fields):
    return [user(), reasoned_assistant(DETAILS, **fields), tool_results()]


class TestRequestShape:
    def test_the_request_goes_to_chat_completions_with_a_bearer_key_and_attribution(self):
        wire = render([user()])

        assert wire.url == f"{BASE_URL}/chat/completions"
        assert wire.headers["Authorization"] == "Bearer or-key-a"
        assert {key: wire.headers[key] for key in ATTRIBUTION_HEADERS} == ATTRIBUTION_HEADERS
        assert wire.body["stream"] is True
        assert wire.body["stream_options"] == {"include_usage": True}
        assert ADAPTER.name == ADAPTER_NAME == "openrouter"

    def test_the_output_limit_is_max_tokens(self):
        body = render([user()], settings=Settings(max_tokens=300)).body

        assert body["max_tokens"] == 300
        assert "max_completion_tokens" not in body

    def test_tools_are_sent_only_when_offered(self):
        assert "tools" not in render([user()]).body
        assert render([user()], tools=[lookup_tool()]).body["tools"][0]["function"]["name"] == "lookup"


class TestReasoningRequest:
    @pytest.mark.parametrize("setting, expected", [
        ("high", {"effort": "high"}),
        ("minimal", {"effort": "minimal"}),
        ("2048", {"max_tokens": 2048}),
        (True, {"enabled": True}),
        (False, {"enabled": False}),
    ])
    def test_the_setting_becomes_the_reasoning_object(self, setting, expected):
        assert render([user()], settings=Settings(reasoning=setting)).body["reasoning"] == expected

    def test_the_saved_effort_is_used_when_the_turn_sets_nothing(self):
        config = make_config(provider_options={"reasoning_effort": "low"})

        body = render([user()], config=config).body

        assert body["reasoning"] == {"effort": "low"}
        assert "reasoning_effort" not in body

    def test_the_turn_setting_wins_over_the_saved_effort(self):
        config = make_config(provider_options={"reasoning_effort": "low"})

        body = render([user()], config=config, settings=Settings(reasoning="high")).body

        assert body["reasoning"] == {"effort": "high"}

    def test_nothing_is_sent_without_a_setting(self):
        assert "reasoning" not in render([user()]).body


class TestReasoningReplay:
    def test_the_stored_blocks_are_sent_unmodified_and_in_order_on_a_tool_call_message(self):
        entry = assistant_entry(render(history(), tools=[lookup_tool()]))

        assert entry["reasoning_details"] == DETAILS
        assert "reasoning" not in entry
        assert [item["id"] for item in entry["reasoning_details"]] == ["rd-1", "rd-2", "rd-3"]

    def test_the_sent_blocks_are_a_copy(self):
        messages = history()
        entry = assistant_entry(render(messages, tools=[lookup_tool()]))

        entry["reasoning_details"][0]["data"] = "changed"

        assert messages[1].raw["reasoning_details"][0]["data"] == "Zm9v"

    def test_encrypted_reasoning_pins_the_provider_that_produced_it(self):
        body = render(history(provider="Google AI Studio"), tools=[lookup_tool()]).body

        assert body["provider"] == {"order": ["google-ai-studio"], "allow_fallbacks": False}

    def test_a_signature_alone_also_pins_the_provider(self):
        body = render(
            [user(), reasoned_assistant([TEXT_BLOCK]), tool_results()], tools=[lookup_tool()],
        ).body

        assert body["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}

    def test_plain_reasoning_text_does_not_pin_anything(self):
        plain = [{"type": "reasoning.text", "id": "rd-9", "index": 0, "text": "Thinking."}]

        wire = render([user(), reasoned_assistant(plain), tool_results()], tools=[lookup_tool()])

        assert "provider" not in wire.body
        assert assistant_entry(wire)["reasoning_details"] == plain

    def test_a_stored_provider_name_missing_leaves_the_routing_alone(self):
        wire = render(history(provider=None), tools=[lookup_tool()])

        assert "provider" not in wire.body
        assert assistant_entry(wire)["reasoning_details"] == DETAILS

    def test_a_message_from_another_key_sends_plain_reasoning_and_no_pin(self):
        other = make_config(api_key="or-key-b")
        stored = reasoned_assistant(DETAILS, config=make_config())

        wire = render([user(), stored, tool_results()], config=other, tools=[lookup_tool()])

        entry = assistant_entry(wire)
        assert "reasoning_details" not in entry
        assert entry["reasoning"] == "Both cities need a lookup."
        assert "provider" not in wire.body

    def test_a_message_from_another_model_sends_plain_reasoning(self):
        wire = render(history(), config=make_config(model="openai/gpt-5.6"), tools=[lookup_tool()])

        entry = assistant_entry(wire)
        assert "reasoning_details" not in entry
        assert entry["reasoning"] == "Both cities need a lookup."

    def test_a_message_without_stored_blocks_sends_plain_reasoning(self):
        stored = reasoned_assistant(DETAILS)
        bare = Message(id=stored.id, role=stored.role, parts=stored.parts, origin=stored.origin)

        entry = assistant_entry(render([user(), bare, tool_results()], tools=[lookup_tool()]))

        assert entry["reasoning"] == "Both cities need a lookup."
        assert "reasoning_details" not in entry

    def test_reasoning_is_sent_only_on_messages_with_tool_calls(self):
        wire = render([user(), reasoned_assistant(DETAILS, calls=False), user("Thanks", "user_2")])

        entry = assistant_entry(wire)
        assert "reasoning_details" not in entry
        assert "reasoning" not in entry
        assert "provider" not in wire.body

    def test_a_dropped_reasoning_part_sends_no_blocks_at_all(self):
        stored = reasoned_assistant(DETAILS)
        stripped = Message(
            id=stored.id,
            role=stored.role,
            parts=tuple(part for part in stored.parts if not isinstance(part, ReasoningPart)),
            origin=stored.origin,
            raw=stored.raw,
        )

        wire = render([user(), stripped, tool_results()], tools=[lookup_tool()])

        assert "reasoning_details" not in assistant_entry(wire)
        assert "provider" not in wire.body


class TestTamperedHistory:
    @staticmethod
    def with_raw(stored, details):
        raw = dict(stored.raw)
        raw["reasoning_details"] = details
        return Message(id=stored.id, role=stored.role, parts=stored.parts, origin=stored.origin, raw=raw)

    @pytest.mark.parametrize("label, edit", [
        ("a dropped block", lambda details: details[:1] + details[2:]),
        ("a dropped last block", lambda details: details[:-1]),
        ("reordered blocks", lambda details: [details[1], details[0], details[2]]),
        ("a reversed list", lambda details: list(reversed(details))),
        ("an edited signature", lambda details: [details[0], {**details[1], "signature": "sig-x"}, details[2]]),
        ("edited text", lambda details: [details[0], {**details[1], "text": "Different."}, details[2]]),
        ("an added block", lambda details: [*details, {"type": "reasoning.text", "index": 3, "text": "More"}]),
        ("an empty list", lambda details: []),
        ("not a list", lambda details: "oops"),
        ("a list of strings", lambda details: ["a", "b"]),
    ])
    def test_a_changed_block_list_is_refused_rather_than_sent(self, label, edit):
        stored = reasoned_assistant(DETAILS)
        tampered = self.with_raw(stored, edit(deepcopy(DETAILS)))

        with pytest.raises(OpaqueViolation):
            render([user(), tampered, tool_results()], tools=[lookup_tool()])

    def test_a_changed_list_on_a_message_without_tool_calls_is_not_sent_so_not_refused(self):
        stored = reasoned_assistant(DETAILS, calls=False)
        tampered = self.with_raw(stored, DETAILS[:1])

        wire = render([user(), tampered, user("Thanks", "user_2")])

        assert "reasoning_details" not in assistant_entry(wire)

    def test_raw_without_the_blocks_is_refused(self):
        stored = reasoned_assistant(DETAILS)
        broken = Message(id=stored.id, role=stored.role, parts=stored.parts, origin=stored.origin, raw={"provider": "Anthropic"})

        with pytest.raises(OpaqueViolation):
            render([user(), broken, tool_results()], tools=[lookup_tool()])

    def test_a_reasoning_part_that_lost_its_digest_is_refused(self):
        stored = reasoned_assistant(DETAILS)
        parts = tuple(
            ReasoningPart(text=part.text) if isinstance(part, ReasoningPart) else part for part in stored.parts
        )
        broken = Message(id=stored.id, role=stored.role, parts=parts, origin=stored.origin, raw=stored.raw)

        with pytest.raises(OpaqueViolation):
            render([user(), broken, tool_results()], tools=[lookup_tool()])
