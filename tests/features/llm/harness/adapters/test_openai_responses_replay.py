import copy
from dataclasses import replace

import pytest

from src.features.llm.harness.errors import OpaqueViolation
from src.features.llm.harness.messages import (
    LEGACY_ORIGIN,
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
)
from src.features.llm.harness.turn_state import NoticeKind
from tests.features.llm.harness.adapters.openai_responses_support import (
    ADAPTER,
    REASONING_PROFILE,
    assistant_message,
    call_item,
    make_config,
    make_profile,
    make_request,
    message_item,
    reasoning_item,
    user_message,
)

PROFILE = make_profile(**REASONING_PROFILE)


def full_output():
    return [
        reasoning_item("rs_1", "enc-1", ["Thinking."]),
        message_item("Checking.", "commentary", "msg_1"),
        call_item("call_1", "get_weather", '{"city":"Paris"}', "fc_1"),
    ]


def history(assistant, config=None):
    return make_request([user_message(), assistant, user_message("Again", "user_2")], config=config)


def rendered_input(request):
    return ADAPTER.render(request, PROFILE).body["input"]


class TestSameOriginReplay:
    def test_the_stored_output_items_are_sent_back_verbatim(self):
        output = full_output()
        assistant = assistant_message(output)

        items = rendered_input(history(assistant))

        assert items[1:4] == output
        assert items[1]["encrypted_content"] == "enc-1"
        assert items[2]["phase"] == "commentary"
        assert items[3]["id"] == "fc_1"

    def test_the_stored_payload_survives_the_wire_bytes_unchanged(self):
        assistant = assistant_message(full_output())

        first = ADAPTER.render(history(assistant), PROFILE).content()
        second = ADAPTER.render(history(assistant), PROFILE).content()

        assert first == second
        assert b"enc-1" in first

    def test_nothing_is_reported_when_nothing_was_left_out(self):
        assistant = assistant_message(full_output())

        assert ADAPTER.reasoning_notices(history(assistant), PROFILE) == ()

    def test_a_reasoning_item_with_no_answer_after_it_is_not_replayed(self):
        output = [message_item("Partial", "commentary", "msg_1"), reasoning_item("rs_9", "enc-9")]
        assistant = assistant_message(output)

        items = rendered_input(history(assistant))
        notices = ADAPTER.reasoning_notices(history(assistant), PROFILE)

        assert [item.get("type", item.get("role")) for item in items] == ["user", "message", "user"]
        assert [notice.kind for notice in notices] == [NoticeKind.REASONING_DROPPED]
        assert notices[0].detail == {"message_id": "asst_1", "reason": "incomplete"}


class TestForeignOriginReplay:
    def test_a_new_key_drops_the_reasoning_and_the_item_ids(self):
        assistant = assistant_message(full_output(), make_config(api_key="test-key-a"))
        request = history(assistant, make_config(api_key="test-key-b"))

        items = rendered_input(request)

        assert items[1:3] == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "Checking."}], "phase": "commentary"},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city":"Paris"}'},
        ]
        assert len(items) == 4
        assert all("encrypted_content" not in item and "id" not in item for item in items)

    def test_a_new_key_is_reported_to_the_user(self):
        assistant = assistant_message(full_output(), make_config(api_key="test-key-a"))
        request = history(assistant, make_config(api_key="test-key-b"))

        notices = ADAPTER.reasoning_notices(request, PROFILE)

        assert len(notices) == 1
        assert notices[0].kind is NoticeKind.REASONING_DROPPED
        assert notices[0].message == "Reasoning made with a different API key was left out."
        assert notices[0].detail == {"message_id": "asst_1", "reason": "credential"}

    def test_a_new_endpoint_counts_as_a_new_credential(self):
        assistant = assistant_message(full_output())
        request = history(assistant, make_config(base_url="https://eu.api.openai.com/v1"))

        assert ADAPTER.reasoning_notices(request, PROFILE)[0].detail["reason"] == "credential"
        assert all("encrypted_content" not in item for item in rendered_input(request))

    def test_a_new_model_drops_the_reasoning_with_its_own_message(self):
        assistant = assistant_message(full_output())
        request = history(assistant, make_config(model="gpt-5.5"))

        notices = ADAPTER.reasoning_notices(request, PROFILE)

        assert notices[0].message == "Reasoning made by a different model was left out."
        assert notices[0].detail["reason"] == "model"
        assert all("encrypted_content" not in item for item in rendered_input(request))

    def test_switching_back_restores_the_original_items(self):
        output = full_output()
        assistant = assistant_message(output, make_config(api_key="test-key-a"))

        away = rendered_input(history(assistant, make_config(api_key="test-key-b")))
        back = rendered_input(history(assistant, make_config(api_key="test-key-a")))

        assert away != back
        assert back[1:4] == output

    def test_reasoning_from_another_adapter_is_dropped_without_a_notice(self):
        other = Origin(adapter="anthropic-messages", provider="anthropic", model="claude", credential="x")
        assistant = Message(
            id="a", role=Role.ASSISTANT, origin=other,
            parts=(ReasoningPart(text="hmm", opaque={"signature": "s"}), TextPart("Answer")),
        )

        request = history(assistant)

        assert rendered_input(request)[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "Answer"}]}
        assert ADAPTER.reasoning_notices(request, PROFILE) == ()

    def test_a_converted_legacy_message_is_rebuilt_from_its_text_and_calls(self):
        assistant = Message(id="a", role=Role.ASSISTANT, origin=LEGACY_ORIGIN, parts=(
            TextPart("Looking."),
            ToolCallPart(call_id="c1", name="lookup", args={"q": "x"}, args_raw=None),
            ToolCallPart(call_id="c2", name="cut_off", args=None, args_raw='{"q"'),
        ))

        items = rendered_input(history(assistant))

        assert items[1:4] == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": '{"q":"x"}'},
            {"type": "function_call", "call_id": "c2", "name": "cut_off", "arguments": '{"q"'},
        ]

    def test_a_message_whose_raw_was_cleared_is_rebuilt_from_its_parts(self):
        assistant = replace(assistant_message(full_output()), raw=None)

        items = rendered_input(history(assistant))

        assert all("encrypted_content" not in item and "id" not in item for item in items)
        assert items[1]["phase"] == "commentary"


def drop(item, key):
    return {name: value for name, value in item.items() if name != key}


def mutate_raw(change):
    def apply(message):
        raw = copy.deepcopy(message.raw)
        change(raw)
        return replace(message, raw=raw)
    return apply


def mutate_parts(change):
    def apply(message):
        return replace(message, parts=tuple(change(part) for part in message.parts))
    return apply


def strip_part_phase(part):
    return replace(part, phase=None) if isinstance(part, TextPart) else part


def strip_part_opaque(part):
    return replace(part, opaque=None) if isinstance(part, ReasoningPart) else part


def edit_part_signature(part):
    if isinstance(part, ReasoningPart):
        return replace(part, opaque={**part.opaque, "encrypted_content": "enc-forged"})
    return part


MUTATIONS = {
    "drop encrypted_content from the stored item": mutate_raw(lambda raw: raw.__setitem__(0, drop(raw[0], "encrypted_content"))),
    "empty the encrypted_content": mutate_raw(lambda raw: raw[0].__setitem__("encrypted_content", "")),
    "drop the reasoning id": mutate_raw(lambda raw: raw.__setitem__(0, drop(raw[0], "id"))),
    "drop phase from the stored message": mutate_raw(lambda raw: raw.__setitem__(1, drop(raw[1], "phase"))),
    "change phase on the stored message": mutate_raw(lambda raw: raw[1].__setitem__("phase", "final_answer")),
    "drop encrypted_content from the parts": mutate_parts(strip_part_opaque),
    "edit encrypted_content in the parts": mutate_parts(edit_part_signature),
    "drop phase from the parts": mutate_parts(strip_part_phase),
    "reorder the stored items": mutate_raw(lambda raw: raw.insert(0, raw.pop(1))),
    "swap the reasoning and the call": mutate_raw(lambda raw: raw.insert(0, raw.pop(2))),
    "remove a stored item": mutate_raw(lambda raw: raw.pop(2)),
    "edit the call arguments": mutate_raw(lambda raw: raw[2].__setitem__("arguments", '{"city":"Rome"}')),
    "edit the message text": mutate_raw(lambda raw: raw[1]["content"][0].__setitem__("text", "Edited.")),
    "replace the payload with an object": lambda message: replace(message, raw={"output": message.raw}),
    "replace an item with a string": mutate_raw(lambda raw: raw.__setitem__(1, "message")),
}


class TestOpaqueViolations:
    def test_the_untouched_message_renders(self):
        assistant = assistant_message(full_output())

        assert ADAPTER.render(history(assistant), PROFILE).body["input"][1]["encrypted_content"] == "enc-1"

    @pytest.mark.parametrize("name", MUTATIONS)
    def test_a_damaged_same_origin_message_is_never_sent(self, name):
        damaged = MUTATIONS[name](assistant_message(full_output()))

        with pytest.raises(OpaqueViolation) as caught:
            ADAPTER.render(history(damaged), PROFILE)

        assert caught.value.detail["message_id"] == "asst_1"
        assert "enc-1" not in str(caught.value)

    def test_a_reasoning_item_that_never_carried_encrypted_content_cannot_be_replayed(self):
        assistant = assistant_message([reasoning_item("rs_1", None), message_item("Hi")])

        with pytest.raises(OpaqueViolation):
            ADAPTER.render(history(assistant), PROFILE)

    def test_damage_to_a_message_of_another_key_is_not_this_requests_business(self):
        stored = assistant_message(full_output(), make_config(api_key="test-key-a"))
        damaged = MUTATIONS["drop encrypted_content from the stored item"](stored)

        items = rendered_input(history(damaged, make_config(api_key="test-key-b")))

        assert all("encrypted_content" not in item for item in items)
