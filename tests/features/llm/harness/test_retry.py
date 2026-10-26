import pytest

from src.features.llm.harness.errors import (
    AuthFailed,
    ConnectionFailed,
    ContextOverflow,
    HarnessError,
    InvalidRequest,
    MalformedStream,
    OpaqueViolation,
    Overloaded,
    QuotaExhausted,
    RateLimited,
    ServerFailed,
)
from src.features.llm.harness.retry import (
    BASE_DELAY,
    MAX_DELAY,
    MAX_RETRIES,
    RETRY_AFTER_CAP,
    BackoffRetry,
    retry_notice,
)
from src.features.llm.harness.events import TextDelta
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised
from tests.features.llm.harness.stepping import engine_turn, outcome_of
from tests.fixtures.llm_engine import drain, scripted_engine
from tests.fixtures.llm_harness import FailMidStream, ScriptedAdapter, text_step


def no_jitter() -> float:
    return 0.0


def full_jitter() -> float:
    return 1.0


def test_a_rate_limit_waits_exactly_as_long_as_the_server_asked():
    policy = BackoffRetry(jitter=no_jitter)
    assert policy.delay(RateLimited("slow down", retry_after=7), 0) == 7.0
    assert policy.delay(RateLimited("slow down", retry_after=7), 2) == 7.0


def test_a_rate_limit_adds_a_little_jitter_on_top_of_the_server_wait():
    assert BackoffRetry(jitter=full_jitter).delay(RateLimited("slow down", retry_after=7), 0) == pytest.approx(8.0)


def test_a_rate_limit_asking_for_longer_than_the_cap_is_not_waited_out():
    policy = BackoffRetry(jitter=no_jitter)
    assert policy.delay(RateLimited("later", retry_after=RETRY_AFTER_CAP), 0) == RETRY_AFTER_CAP
    assert policy.delay(RateLimited("later", retry_after=RETRY_AFTER_CAP + 1), 0) is None


def test_a_rate_limit_without_a_wait_backs_off_exponentially():
    policy = BackoffRetry(jitter=full_jitter)
    waits = [policy.delay(RateLimited("slow down"), attempt) for attempt in range(MAX_RETRIES)]
    assert waits == [BASE_DELAY * 2 ** step for step in range(MAX_RETRIES)]


@pytest.mark.parametrize("error_type", [Overloaded, ServerFailed, ConnectionFailed])
def test_transient_failures_back_off_with_jitter_between_half_and_the_full_ceiling(error_type):
    low = BackoffRetry(jitter=no_jitter).delay(error_type("down"), 2)
    high = BackoffRetry(jitter=full_jitter).delay(error_type("down"), 2)
    assert (low, high) == (2.0, 4.0)


def test_backoff_never_exceeds_the_delay_cap():
    policy = BackoffRetry(max_retries=50, jitter=full_jitter)
    assert policy.delay(Overloaded("down"), 40) == MAX_DELAY


def test_no_retry_once_the_retries_are_used_up():
    policy = BackoffRetry(jitter=no_jitter)
    assert policy.delay(Overloaded("down"), MAX_RETRIES - 1) is not None
    assert policy.delay(Overloaded("down"), MAX_RETRIES) is None


def test_zero_retries_means_the_first_failure_is_final():
    assert BackoffRetry(max_retries=0).delay(Overloaded("down"), 0) is None


@pytest.mark.parametrize(
    "error_type",
    [QuotaExhausted, AuthFailed, InvalidRequest, OpaqueViolation, ContextOverflow, MalformedStream, HarnessError],
)
def test_these_failures_are_never_retried(error_type):
    assert BackoffRetry(jitter=no_jitter).delay(error_type("no", retry_after=1), 0) is None


def test_a_failure_that_is_not_a_harness_error_is_not_retried():
    assert BackoffRetry().delay(RuntimeError("boom"), 0) is None


def test_the_notice_says_why_and_when_the_next_attempt_comes():
    notice = retry_notice(Overloaded("down", status=529), 1, 4.2, step_seq=3)
    assert notice.kind is NoticeKind.RETRYING
    assert notice.step_seq == 3
    assert notice.message == "The provider is overloaded. Retrying in 4 seconds."
    assert notice.detail == {"error": "overloaded", "status": 529, "attempt": 1, "delay": 4.2}
    assert retry_notice(RateLimited("x"), 0, 0.2).message == "The provider is rate limiting requests. Retrying shortly."
    assert retry_notice(ConnectionFailed("x"), 0, 1.0).message == "The connection to the provider failed. Retrying in 1 second."


async def test_the_engine_retries_each_failure_with_a_notice_and_then_succeeds():
    adapter = ScriptedAdapter(Overloaded("busy"), RateLimited("slow", retry_after=2), text_step("hello"))
    rig = scripted_engine(adapter, retry=BackoffRetry(jitter=no_jitter))
    events = await drain(rig.engine.run(engine_turn(tool_context=None)))
    notices = [event.notice for event in events if isinstance(event, NoticeRaised)]
    assert [notice.kind for notice in notices] == [NoticeKind.RETRYING, NoticeKind.RETRYING]
    assert [notice.detail["attempt"] for notice in notices] == [0, 1]
    assert rig.sleeps == [0.5, 2.0]
    outcome = outcome_of(events)
    assert outcome.content == "hello"
    assert outcome.state.notices == tuple(notices)
    assert adapter.exhausted


async def test_a_failure_that_keeps_coming_is_raised_after_the_last_retry():
    adapter = ScriptedAdapter(*[ServerFailed("500") for _ in range(MAX_RETRIES + 1)])
    rig = scripted_engine(adapter, retry=BackoffRetry(jitter=no_jitter))
    seen: list = []
    with pytest.raises(ServerFailed):
        async for event in rig.engine.run(engine_turn(tool_context=None)):
            seen.append(event)
    assert len(adapter.requests) == MAX_RETRIES + 1
    assert len(rig.sleeps) == MAX_RETRIES
    assert len([event for event in seen if isinstance(event, NoticeRaised)]) == MAX_RETRIES


@pytest.mark.parametrize("error", [AuthFailed("key"), QuotaExhausted("billing"), InvalidRequest("bad")])
async def test_a_failure_that_cannot_succeed_is_raised_at_once(error):
    adapter = ScriptedAdapter(error, text_step("never reached"))
    rig = scripted_engine(adapter, retry=BackoffRetry(jitter=no_jitter))
    with pytest.raises(type(error)):
        await drain(rig.engine.run(engine_turn(tool_context=None)))
    assert rig.sleeps == []
    assert len(adapter.requests) == 1


async def test_a_failure_after_the_stream_has_started_is_not_retried():
    adapter = ScriptedAdapter(FailMidStream([TextDelta(block=1, text="par")], Overloaded("cut")), text_step("good"))
    rig = scripted_engine(adapter, retry=BackoffRetry(jitter=no_jitter))
    seen: list = []

    with pytest.raises(Overloaded):
        async for event in rig.engine.run(engine_turn(tool_context=None)):
            seen.append(event)

    assert rig.sleeps == []
    assert len(adapter.requests) == 1
    assert [event.text for event in seen if isinstance(event, TextDelta)] == ["par"]
    assert [event for event in seen if isinstance(event, NoticeRaised)] == []
