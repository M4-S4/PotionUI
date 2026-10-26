import json
import sqlite3
from typing import Tuple

import pytest

from src.features.llm.harness import codec
from src.features.llm.harness.events import FinishReason, Usage
from src.features.llm.harness.messages import (
    LEGACY_ORIGIN,
    ImagePart,
    Origin,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from src.features.llm.harness.turn_state import NoticeKind, StepStatus, TurnState, TurnStatus
from tests.fixtures.legacy_chats import chats, env, execution, previous_template


def test_the_previous_schema_has_no_steps(env):
    database, _ = env

    with database.get_cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE name IN ('chat_turns', 'chat_steps')")
        assert cursor.fetchall() == []
        cursor.execute("PRAGMA table_info(chat_messages)")
        assert "turn_id" not in {row[1] for row in cursor.fetchall()}


def test_up_creates_the_schema_and_leaves_an_empty_database_empty(env, chats):
    _, migration = env

    migration.up()

    assert chats.count("chat_turns") == 0
    assert chats.count("chat_steps") == 0
    with chats.database.get_cursor() as cursor:
        cursor.execute("PRAGMA table_info(chat_messages)")
        assert "turn_id" in {row[1] for row in cursor.fetchall()}
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = 'chat_steps_append_only'")
        assert cursor.fetchone() is not None


def test_a_plain_exchange_becomes_one_completed_turn(env, chats):
    _, migration = env
    session = chats.session("s1")
    user = chats.row(session, "user", "hello")
    assistant = chats.row(
        session, "assistant", "hi there",
        {"prompt_tokens": 40, "completion_tokens": 7, "tokens_used": 47, "behavior_trace": {"v": 1}},
    )

    migration.up()

    steps = chats.steps(session)
    assert [(step.seq, step.message.role) for step in steps] == [(0, Role.USER), (1, Role.ASSISTANT)]
    assert steps[0].message.parts == (TextPart(text="hello"),)
    assert steps[1].message.parts == (TextPart(text="hi there"),)
    assert steps[1].message.origin == LEGACY_ORIGIN
    assert steps[1].message.raw is None
    assert steps[1].finish is FinishReason.STOP
    assert steps[1].status is StepStatus.COMPLETED
    assert (steps[1].usage.input_tokens, steps[1].usage.output_tokens, steps[1].usage.total_tokens) == (40, 7, 47)
    (turn,) = chats.turns(session)
    state = codec.loads(TurnState, turn["state_json"])
    assert turn["id"] == user and turn["status"] == "completed"
    assert state.status is TurnStatus.COMPLETED and state.first_seq == 0
    assert state.snapshot.llm_config_id == "cfg" and state.snapshot.mode == "assistant"
    assert chats.message(user)[0] == turn["id"]
    turn_id, metadata = chats.message(assistant)
    assert turn_id == turn["id"]
    assert metadata["step_range"] == {"first": 1, "last": 1}
    assert metadata["behavior_trace"] == {"v": 1}


def test_a_tool_turn_keeps_round_reasoning_calls_results_and_the_final_answer(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "find cats")
    assistant = chats.row(session, "assistant", "Found 2 cats.", {
        "prompt_tokens": 100, "completion_tokens": 20, "tokens_used": 120, "turn_rounds": 1,
        "thinking": {"segments": [
            {"text": "plan the search", "duration_ms": 900},
            {"text": "summarise results", "duration_ms": 400},
        ]},
        "tool_transcript": [{
            "calls": [{"id": "call_1", "name": "search", "arguments": {"q": "cats", "n": 2}}],
            "results": [{"tool_call_id": "call_1", "content": "2 hits"}],
            "reasoning": "plan the search",
        }],
        "tool_executions": [execution("search", "call_1", {"q": "cats", "n": 2}, data="2 hits")],
    })

    migration.up()

    steps = chats.steps(session)
    assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    asks, tool, final = steps[1], steps[2], steps[3]
    assert asks.message.parts == (
        ReasoningPart(text="plan the search", duration_ms=900),
        ToolCallPart(call_id="call_1", name="search", args={"q": "cats", "n": 2}, args_raw=None, wire_id="call_1"),
    )
    assert asks.finish is FinishReason.TOOL_CALLS and asks.usage is None
    assert tool.message.parts == (ToolResultPart(call_id="call_1", name="search", content=(TextPart(text="2 hits"),)),)
    assert tool.message.origin is None
    assert final.message.parts == (ReasoningPart(text="summarise results", duration_ms=400), TextPart(text="Found 2 cats."))
    assert final.usage.total_tokens == 120
    assert chats.message(assistant)[1]["step_range"] == {"first": 1, "last": 3}
    (turn,) = chats.turns(session)
    assert codec.loads(TurnState, turn["state_json"]).rounds_used == 1


def test_a_failed_and_a_rejected_call_are_error_results(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "do two things")
    chats.row(session, "assistant", "Tried.", {
        "tool_transcript": [{
            "calls": [
                {"id": "a", "name": "rename", "arguments": {}},
                {"id": "b", "name": "delete", "arguments": {}},
                {"id": "c", "name": "list", "arguments": {}},
            ],
            "results": [{"tool_call_id": "c", "content": "[]"}],
        }],
        "tool_executions": [
            execution("rename", "a", status="failed", success=False, error="name taken"),
            execution("delete", "b", status="rejected", reason="not now"),
            execution("list", "c", data="[]"),
        ],
    })

    migration.up()

    results = chats.steps(session)[2].message.tool_results()
    assert [(item.call_id, item.text(), item.is_error) for item in results] == [
        ("a", "Error: name taken", True),
        ("b", "User rejected this. Reason: not now", True),
        ("c", "[]", False),
    ]


def test_a_pending_approval_ends_the_turn_cancelled_with_an_expired_notice(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "rename it")
    paused = chats.row(session, "assistant", "I'll rename it.", {
        "turn_rounds": 1,
        "tool_transcript": [{
            "calls": [
                {"id": "c1", "name": "rename_prompt", "arguments": {"name": "b"}},
                {"id": "c2", "name": "search", "arguments": {}},
            ],
            "results": [{"tool_call_id": "c2", "content": "Not run: waiting for approval of rename_prompt."}],
        }],
        "tool_executions": [
            execution("rename_prompt", "c1", {"name": "b"}, status="pending", preview={"before": "a"}),
            execution("search", "c2", success=False, error="Not run", skipped=True),
        ],
    })

    migration.up()

    steps = chats.steps(session)
    assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert steps[1].message.parts[0] == TextPart(text="I'll rename it.")
    assert [type(part).__name__ for part in steps[1].message.parts] == ["TextPart", "ToolCallPart", "ToolCallPart"]
    results = steps[2].message.tool_results()
    assert [(item.call_id, item.is_error) for item in results] == [("c1", True), ("c2", True)]
    assert results[0].text() == "The approval request expired before it was decided."
    (turn,) = chats.turns(session)
    state = codec.loads(TurnState, turn["state_json"])
    assert turn["status"] == "cancelled" and state.status is TurnStatus.CANCELLED
    assert [(notice.kind, notice.step_seq) for notice in state.notices] == [(NoticeKind.APPROVAL_EXPIRED, 1)]
    assert state.pending == ()
    metadata = chats.message(paused)[1]
    assert [record["approval_status"] for record in metadata["tool_executions"]] == ["expired", None]


def test_a_continued_approval_chain_joins_one_turn(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "rename it")
    first = chats.row(session, "assistant", "Renaming.", {
        "turn_rounds": 1,
        "tool_transcript": [{
            "calls": [{"id": "c1", "name": "rename_prompt", "arguments": {"name": "b"}}],
            "results": [{"tool_call_id": "c1", "content": "Renamed."}],
        }],
        "tool_executions": [execution("rename_prompt", "c1", {"name": "b"}, data="Renamed.", status="approved")],
    })
    second = chats.row(session, "assistant", "Done, renamed.", {
        "turn_rounds": 2, "prompt_tokens": 9, "continued_from": {"message_id": first, "tool_index": 0},
    })

    migration.up()

    steps = chats.steps(session)
    assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert steps[1].message.parts[0] == TextPart(text="Renaming.")
    assert steps[2].message.tool_results()[0].text() == "Renamed."
    assert steps[3].message.parts == (TextPart(text="Done, renamed."),)
    (turn,) = chats.turns(session)
    state = codec.loads(TurnState, turn["state_json"])
    assert state.status is TurnStatus.COMPLETED and state.rounds_used == 2 and state.notices == ()
    assert {step.turn_id for step in steps} == {turn["id"]}
    assert chats.message(first)[0] == chats.message(second)[0] == turn["id"]
    assert chats.message(first)[1]["step_range"] == {"first": 1, "last": 2}
    assert chats.message(second)[1]["step_range"] == {"first": 3, "last": 3}


def test_a_user_row_without_an_answer_is_a_cancelled_turn_of_its_own(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "first, lost")
    chats.row(session, "user", "second")
    chats.row(session, "assistant", "answer")
    chats.row(session, "user", "third, lost")

    migration.up()

    assert [(turn["status"]) for turn in chats.turns(session)] == ["cancelled", "completed", "cancelled"]
    assert [step.seq for step in chats.steps(session)] == [0, 1, 2, 3]


def test_a_user_image_is_kept_as_a_reference(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "look", {"image_path": "/api/media/generations/G1/out.jpg"})
    chats.row(session, "assistant", "nice")

    migration.up()

    assert chats.steps(session)[0].message.parts == (
        TextPart(text="look"),
        ImagePart(media_type="image/jpeg", ref="/api/media/generations/G1/out.jpg"),
    )


def test_a_reasoning_only_answer_without_a_transcript_keeps_its_reasoning(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "why")
    chats.row(session, "assistant", "Because.", {"thinking": {"segments": [{"text": "hmm", "duration_ms": 10}, {"text": " ", "duration_ms": 1}]}})

    migration.up()

    assert chats.steps(session)[1].message.parts == (ReasoningPart(text="hmm", duration_ms=10), TextPart(text="Because."))


def test_stored_json_decodes_under_the_contract_and_is_byte_stable(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "zażółć 🌍", {"image_path": "a.png"})
    chats.row(session, "assistant", "ok ✓", {
        "prompt_tokens": 3,
        "tool_transcript": [{"calls": [{"id": "c", "name": "t", "arguments": {"ł": [1, 2.5, None]}}], "results": [{"tool_call_id": "c", "content": "r"}]}],
        "tool_executions": [execution("t", "c", data="r")],
    })

    migration.up()

    with chats.database.get_cursor() as cursor:
        cursor.execute("SELECT parts, origin, usage FROM chat_steps")
        for row in cursor.fetchall():
            assert codec.dumps(codec.loads(Tuple[Part, ...], row["parts"])) == row["parts"]
            if row["origin"]:
                assert codec.dumps(codec.loads(Origin, row["origin"])) == row["origin"]
            if row["usage"]:
                assert codec.dumps(codec.loads(Usage, row["usage"])) == row["usage"]
        cursor.execute("SELECT state_json FROM chat_turns")
        for row in cursor.fetchall():
            assert codec.dumps(codec.loads(TurnState, row["state_json"])) == row["state_json"]


def test_a_second_run_changes_nothing(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "hello")
    assistant = chats.row(session, "assistant", "hi", {"turn_rounds": 1})
    migration.up()
    before = (chats.count("chat_turns"), chats.count("chat_steps"), chats.steps(session), chats.message(assistant))

    migration.up()

    assert (chats.count("chat_turns"), chats.count("chat_steps"), chats.steps(session), chats.message(assistant)) == before


def test_a_session_that_already_has_steps_is_skipped(env, chats):
    _, migration = env
    done = chats.session("s1")
    chats.row(done, "user", "hello")
    chats.row(done, "assistant", "hi")
    migration.up()
    fresh = chats.session("s2")
    chats.row(fresh, "user", "later")
    chats.row(fresh, "assistant", "answer")
    steps_before = chats.steps(done)

    migration.up()

    assert chats.steps(done) == steps_before
    assert [step.seq for step in chats.steps(fresh)] == [0, 1]


def test_more_sessions_than_one_batch_are_all_converted(env, chats):
    _, migration = env
    for number in range(60):
        session = chats.session(f"s{number:03d}")
        chats.row(session, "user", f"q{number}")
        chats.row(session, "assistant", f"a{number}")
    chats.session("empty")

    migration.up()

    assert chats.count("chat_turns") == 60
    assert chats.count("chat_steps") == 120


def test_a_row_with_unreadable_metadata_still_converts(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "hello")
    broken = chats.row(session, "assistant", "hi", raw_metadata="{not json")

    migration.up()

    assert [step.message.role for step in chats.steps(session)] == [Role.USER, Role.ASSISTANT]
    assert chats.turn_of(broken) is not None
    with chats.database.get_cursor() as cursor:
        cursor.execute("SELECT metadata FROM chat_messages WHERE id = ?", (broken,))
        assert cursor.fetchone()["metadata"] == "{not json"


def test_non_finite_arguments_do_not_stop_the_conversion(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "go")
    chats.row(
        session, "assistant", "done",
        raw_metadata='{"tool_transcript": [{"calls": [{"id": "c", "name": "t", "arguments": {"x": NaN, "y": 1}}], "results": []}]}',
    )

    migration.up()

    call = chats.steps(session)[1].message.tool_calls()[0]
    assert call.args == {"x": None, "y": 1}
    assert chats.steps(session)[2].message.tool_results()[0].text() == "No result was recorded for this call."


def test_steps_cannot_be_updated(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "hello")
    migration.up()

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with chats.database.get_cursor() as cursor:
            cursor.execute("UPDATE chat_steps SET status = 'failed'")


def test_deleting_a_session_removes_its_turns_and_steps(env, chats):
    _, migration = env
    session = chats.session("s1")
    chats.row(session, "user", "hello")
    chats.row(session, "assistant", "hi")
    migration.up()

    with chats.database.get_cursor() as cursor:
        cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session,))

    assert chats.count("chat_turns") == 0
    assert chats.count("chat_steps") == 0
