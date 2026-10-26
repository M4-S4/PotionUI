import pytest

from src.features.chat import decisions
from src.features.chat.decisions import DecisionRefusal, DecisionRefused
from src.features.llm.harness.turn_state import Decision, DecisionKind, PendingCall, TurnStatus
from tests.fixtures.llm_turns import state


def waiting(*pending):
    return state(status=TurnStatus.AWAITING_DECISION, pending=pending)


def approval(call_id, name="delete_prompt"):
    return PendingCall(call_id=call_id, name=name, decision=DecisionKind.APPROVAL, preview={})


def external(call_id):
    return PendingCall(call_id=call_id, name="ask_user", decision=DecisionKind.EXTERNAL)


def refusal(*args, **kwargs):
    with pytest.raises(DecisionRefused) as caught:
        decisions.decision_for(*args, **kwargs)
    return caught.value.refusal


def test_a_pending_call_is_decided_by_the_id_stored_on_the_server():
    current = waiting(approval("c2"))

    decision = decisions.decision_for(current, ["c1", "c2"], 1, True)

    assert decision == Decision(call_id="c2", approved=True, reason=None)


def test_a_rejection_keeps_a_bounded_reason():
    current = waiting(approval("c1"))

    decision = decisions.decision_for(current, ["c1"], 0, False, "x" * 5000)

    assert decision.approved is False and decision.reason == "x" * decisions.MAX_REASON_CHARS
    assert decisions.decision_for(current, ["c1"], 0, False, "").reason is None


@pytest.mark.parametrize("index", [-1, 2, 99])
def test_an_index_outside_the_message_is_refused(index):
    assert refusal(waiting(approval("c1")), ["c1", "c2"], index, True) is DecisionRefusal.INDEX_OUT_OF_RANGE


def test_an_index_that_points_at_a_call_already_decided_is_stale():
    current = waiting(approval("c2"))

    assert refusal(current, ["c1", "c2"], 0, True) is DecisionRefusal.NOT_PENDING


def test_a_decision_made_in_another_tab_makes_the_same_index_stale():
    before = waiting(approval("c1"), approval("c2"))
    assert decisions.decision_for(before, ["c1", "c2"], 0, True).call_id == "c1"

    after = waiting(approval("c2"))

    assert refusal(after, ["c1", "c2"], 0, True) is DecisionRefusal.NOT_PENDING
    assert decisions.decision_for(after, ["c1", "c2"], 1, True).call_id == "c2"


def test_a_row_without_call_ids_cannot_be_decided():
    assert refusal(waiting(approval("c1")), [None], 0, True) is DecisionRefusal.NOT_PENDING


@pytest.mark.parametrize("status", [TurnStatus.RUNNING, TurnStatus.COMPLETED, TurnStatus.CANCELLED, TurnStatus.FAILED])
def test_a_turn_that_is_not_waiting_refuses_every_decision(status):
    assert refusal(state(status=status, pending=(approval("c1"),)), ["c1"], 0, True) is DecisionRefusal.NOT_AWAITING


def test_a_missing_turn_refuses_the_decision():
    assert refusal(None, ["c1"], 0, True) is DecisionRefusal.NOT_AWAITING


def test_only_approvals_can_be_decided_here():
    assert refusal(waiting(external("c1")), ["c1"], 0, True) is DecisionRefusal.NOT_AN_APPROVAL


def test_call_ids_come_from_the_stored_tool_cards():
    metadata = {"tool_executions": [{"tool_call_id": "c1"}, {"tool_name": "x"}, "odd"]}

    assert decisions.execution_call_ids(metadata) == ["c1", None, None]
    assert decisions.execution_call_ids({}) == []
    assert decisions.execution_call_ids(None) == []
    assert decisions.execution_call_ids({"tool_executions": "bad"}) == []


def test_a_reply_rejects_every_pending_approval_with_the_reply_as_reason():
    current = waiting(approval("c1"), external("c2"), approval("c3"))

    rejected = decisions.rejections_for_reply(current, "no, do this instead")

    assert rejected == [
        Decision(call_id="c1", approved=False, reason="no, do this instead"),
        Decision(call_id="c3", approved=False, reason="no, do this instead"),
    ]


def test_a_reply_changes_nothing_when_no_turn_is_waiting():
    assert decisions.rejections_for_reply(None, "hi") == []
    assert decisions.rejections_for_reply(state(status=TurnStatus.RUNNING, pending=(approval("c1"),)), "hi") == []
