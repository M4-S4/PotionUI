import json

from src.features.chat import tool_transcript


def _metadata(confirmed: dict | None):
    transcript = [
        {
            "calls": [{"id": "c1", "name": "search_model_prompts", "arguments": {"queries": ["street"]}}],
            "results": [{"tool_call_id": "c1", "content": json.dumps({"results": [{"name": "Cyberpunk Streetscape"}]})}],
        },
        {
            "calls": [{"id": "c2", "name": "add_prompt", "arguments": {"name": "Moonlit shelf", "segments": []}}],
            "results": [{"tool_call_id": "c2", "content": "Waiting for approval."}],
        },
    ]
    executions = [
        {"tool_call_id": "c1", "tool_name": "search_model_prompts", "approval_status": None},
        {
            "tool_call_id": "c2",
            "tool_name": "add_prompt",
            "approval_status": tool_transcript.APPROVED,
            "result": {"success": True, "data": json.dumps(confirmed) if confirmed else "", "error": None},
        },
    ]
    return transcript, {"tool_transcript": transcript, "tool_executions": executions}


def test_confirmed_result_is_the_last_tool_output_after_an_earlier_search():
    confirmed = {"action": "add_prompt", "success": True, "name": "Moonlit shelf", "message": 'Saved the prompt "Moonlit shelf".'}
    transcript, metadata = _metadata(confirmed)
    tool_transcript.set_transcript_result(transcript, "c2", json.dumps(confirmed))

    messages = tool_transcript.replay_messages("", metadata)
    tool_messages = [m for m in messages if m["role"] == "tool"]

    assert [m["name"] for m in tool_messages] == ["search_model_prompts", "add_prompt"]
    assert "Moonlit shelf" in tool_messages[-1]["content"]
    assert "Cyberpunk Streetscape" not in tool_messages[-1]["content"]
    assert "Cyberpunk Streetscape" in tool_messages[0]["content"]
    assert messages[-1]["role"] == "tool"


def test_replay_falls_back_to_the_approved_execution_when_the_transcript_has_no_result():
    confirmed = {"action": "add_prompt", "success": True, "name": "Moonlit shelf"}
    transcript, metadata = _metadata(confirmed)
    transcript[1]["results"] = []

    messages = tool_transcript.replay_messages("", metadata)

    assert "Moonlit shelf" in messages[-1]["content"]


def _reasoned_metadata():
    transcript, metadata = _metadata({"action": "add_prompt", "success": True})
    transcript[0]["reasoning"] = "search before adding"
    return transcript, metadata


def test_replay_leaves_reasoning_out_unless_asked():
    _transcript, metadata = _reasoned_metadata()

    messages = tool_transcript.replay_messages("", metadata)

    assert not any("reasoning" in message for message in messages)


def test_replay_puts_each_rounds_reasoning_on_that_rounds_tool_call_message():
    _transcript, metadata = _reasoned_metadata()

    messages = tool_transcript.replay_messages("", metadata, with_reasoning=True)

    calling = [message for message in messages if message.get("tool_calls")]
    assert [message.get("reasoning") for message in calling] == ["search before adding", None]
    assert [message["role"] for message in messages] == ["assistant", "tool", "assistant", "tool"]


def test_replay_skips_an_empty_reasoning():
    transcript, metadata = _reasoned_metadata()
    transcript[0]["reasoning"] = ""

    messages = tool_transcript.replay_messages("", metadata, with_reasoning=True)

    assert not any("reasoning" in message for message in messages)


def test_replay_with_reasoning_still_places_a_paused_turns_visible_text_first():
    _transcript, metadata = _reasoned_metadata()
    metadata["tool_executions"][1]["approval_status"] = tool_transcript.PENDING

    messages = tool_transcript.replay_messages("Let me search.", metadata, with_reasoning=True)

    assert messages[0]["content"] == "Let me search."
    assert messages[0]["reasoning"] == "search before adding"
