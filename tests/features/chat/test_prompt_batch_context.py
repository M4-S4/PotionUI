import pytest

from src.features.chat.context_builder import ChatContextBuilder
from src.features.llm.tools.governance import ToolGovernanceRepository
from src.features.llm.tools.tool_conditionals import render_tool_conditionals
from tests.fixtures.chat_engine import typed_history
from tests.features.chat.test_tool_availability_transparency import (
    _make_builder,
    _session,
    governance_db,
)

BATCH_ACTION_NAMES = ("add_prompt", "select_prompt", "remove_prompt", "rename_prompt", "prompt_batch_actions")


def _seg(i, content, negative=False):
    return {"index": i, "id": f"s{i}", "content": content, "name": f"Seg {i}", "type": "text", "enabled": True, **({"negative": True} if negative else {})}


def _batch():
    return {
        "active": 1,
        "prompts": [
            {"index": 0, "name": "Day", "segments": [_seg(0, "a quiet village at noon"), _seg(1, "blurry", True)]},
            {"index": 1, "name": "", "segments": [_seg(0, "a quiet village at night under stars")]},
            {"index": 2, "name": "Storm", "segments": []},
        ],
    }


def _form_state(supported=True):
    return {
        "preset": "native/SDXL",
        "mode": "image",
        "form_data": {},
        "video_director": {"active": False},
        "prompt_batch": {"supported": supported},
    }


def _state_block(metadata, offered=("get_current_segments",)):
    history = [{"role": "user", "content": "hi"}]
    typed_history(ChatContextBuilder.inject_prompt_state_block)(history, metadata, offered_tools=set(offered))
    return history[0]["content"]


def test_prompt_state_lists_three_prompts_and_marks_the_active_one():
    metadata = {
        "form_state": _form_state(),
        "segments": [_seg(0, "a quiet village at night under stars")],
        "prompt_batch": _batch(),
    }

    text = _state_block(metadata)

    assert "Prompts (3, active 1):" in text
    assert '  0 "Day" (2 segments): "a quiet village at noon"' in text
    assert '  1* "Prompt 2" (1 segment): "a quiet village at night under stars"' in text
    assert '  2 "Storm" (0 segments)' in text
    assert "get_current_segments with prompt_index" in text
    assert "belong to prompt 1" in text


def test_prompt_state_omits_overview_without_batch_support_or_with_one_prompt():
    batch = _batch()
    unsupported = {"form_state": _form_state(False), "segments": [_seg(0, "x")], "prompt_batch": batch}
    assert "Prompts (" not in _state_block(unsupported)

    single = {
        "form_state": _form_state(),
        "segments": [_seg(0, "x")],
        "prompt_batch": {"active": 0, "prompts": batch["prompts"][:1]},
    }
    assert "Prompts (" not in _state_block(single)


def test_prompt_state_overview_does_not_name_the_tool_when_it_is_not_offered():
    metadata = {"form_state": _form_state(), "segments": [_seg(0, "x")], "prompt_batch": _batch()}

    text = _state_block(metadata, offered=("get_form_state",))

    assert "Prompts (3, active 1):" in text
    assert "get_current_segments" not in text
    assert "prompt_index" not in text


def test_prompt_state_renders_with_an_empty_active_prompt():
    metadata = {"form_state": _form_state(), "segments": [], "prompt_batch": _batch()}

    assert "Prompts (3, active 1):" in _state_block(metadata)


def _prompt(builder, session, supported):
    prompt, allowed, _, _ = builder.resolve_session_prompt_and_tools(session, form_state=_form_state(supported))
    return prompt, allowed


def test_mode_prompt_offers_batch_actions_only_when_supported(governance_db):
    builder = _make_builder(ToolGovernanceRepository())

    on, _ = _prompt(builder, _session(), True)
    off, _ = _prompt(builder, _session(), False)

    for name in ("add_prompt", "select_prompt", "remove_prompt", "rename_prompt"):
        assert name in on, name
        assert name not in off, name
    assert "prompt_batch_actions" not in on
    assert "{{" not in on


@pytest.mark.parametrize("case", ["tool_withheld", "batch_unsupported", "director"])
def test_batch_action_names_never_reach_the_model_when_unavailable(governance_db, case):
    repo = ToolGovernanceRepository()
    if case == "tool_withheld":
        repo.upsert_config("cfg-a", "get_current_segments", enabled=False)
    builder = _make_builder(repo)
    form_state = _form_state(case != "batch_unsupported")
    if case == "director":
        form_state["video_director"] = {"active": True}
    session = _session()

    system_prompt, allowed, mode, deferred = builder.resolve_session_prompt_and_tools(session, form_state=form_state)
    withheld = builder.withheld_tools_for_session(session, form_state=form_state)
    offered = set(allowed or ()).union(*(deferred or {}).values())
    history = [{"role": "user", "content": "hi"}]
    metadata = {"form_state": form_state, "segments": [_seg(0, "x")], "prompt_batch": _batch()}
    typed_history(builder.inject_prompt_state_block)(history, metadata, offered_tools=offered)
    typed_history(builder.inject_tool_availability_block)(history, allowed, withheld)
    text = system_prompt + "\n" + "\n".join(m["content"] for m in history)

    for name in BATCH_ACTION_NAMES:
        assert name not in text, name


def test_capability_marker_is_dropped_when_not_granted():
    assert render_tool_conditionals("{{#if prompt_batch_actions}}x{{/if}}", ["get_current_segments"]) == ""


def test_prompt_overview_is_bounded_for_many_long_prompts():
    prompts = [
        {"index": i, "name": f"Prompt name {i} " + "n" * 80, "segments": [_seg(0, "word " * 100)]}
        for i in range(25)
    ]
    metadata = {
        "form_state": _form_state(),
        "segments": [_seg(0, "x")],
        "prompt_batch": {"active": 0, "prompts": prompts},
    }

    text = _state_block(metadata)

    lines = [line for line in text.splitlines() if line.startswith("  ") and '" (' in line]
    assert 0 < len(lines) <= 20
    assert all(len(line) < 200 for line in lines)
    assert all(line.count("…") >= 1 for line in lines)
    preview = lines[0].split(': "', 2)[-1]
    assert preview.endswith('…"')
    assert len(preview.rstrip('"').rstrip("…")) <= 60
    assert len(text) <= 4200


@pytest.mark.parametrize(
    "batch",
    [
        "garbage",
        ["a", "b"],
        {"active": "x", "prompts": "nope"},
        {"active": None, "prompts": [{"index": "1", "name": "A"}, {"index": None}, "s", {"index": 1, "name": 5, "segments": "x"}]},
        {"active": 0, "prompts": [{"index": 0, "name": ["l"], "segments": [{"content": 7}, None]}, {"index": 1}]},
    ],
)
def test_malformed_prompt_batch_never_raises(batch):
    metadata = {"form_state": _form_state(), "segments": [_seg(0, "x")], "prompt_batch": batch}

    _state_block(metadata)


@pytest.mark.parametrize("form_state", [{"prompt_batch": "z"}, {"prompt_batch": ["supported"]}])
def test_malformed_form_state_never_raises(form_state):
    metadata = {"form_state": form_state, "segments": [_seg(0, "x")], "prompt_batch": _batch()}

    assert "Prompts (" not in _state_block(metadata)
