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
