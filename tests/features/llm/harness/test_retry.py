from typing import List

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
from src.features.llm.harness.events import Finish, FinishReason, TextDelta
from src.features.llm.harness.ports import RetryPolicy
from src.features.llm.harness.retry import (
    BASE_DELAY,
    MAX_ATTEMPTS,
    MAX_DELAY,
    RETRY_AFTER_CAP,
    BackoffRetry,
    retry_notice,
    retrying,
)
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised
from tests.features.llm.harness.stepping import request_for, stream_events
from tests.fixtures.llm_harness import ScriptedAdapter, scripted_profile, text_step


def no_jitter() -> float:
    return 0.0


def full_jitter() -> float:
    return 1.0


class Sleeps:
    def __init__(self) -> None:
        self.calls: List[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def play(adapter: ScriptedAdapter, policy: RetryPolicy, sleeps: Sleeps):
    profile = scripted_profile()
    return retrying(lambda: stream_events(adapter, profile, request_for()), policy, sleep=sleeps)


async def drain(stream) -> list:
    return [event async for event in stream]


def test_a_retryable_failure_gets_a_delay_in_seconds():
    assert isinstance(BackoffRetry().delay(RateLimited("slow down"), 1), float)


def test_a_rate_limit_waits_exactly_as_long_as_the_server_asked():
    policy = BackoffRetry(jitter=no_jitter)
    assert policy.delay(RateLimited("slow down", retry_after=7), 1) == 7.0
    assert policy.delay(RateLimited("slow down", retry_after=7), 3) == 7.0


def test_a_rate_limit_adds_a_little_jitter_on_top_of_the_server_wait():
    assert BackoffRetry(jitter=full_jitter).delay(RateLimited("slow down", retry_after=7), 1) == pytest.approx(8.0)


def test_a_rate_limit_asking_for_longer_than_the_cap_is_not_waited_out():
    policy = BackoffRetry(jitter=no_jitter)
    assert policy.delay(RateLimited("later", retry_after=RETRY_AFTER_CAP), 1) == RETRY_AFTER_CAP
    assert policy.delay(RateLimited("later", retry_after=RETRY_AFTER_CAP + 1), 1) is None


def test_a_rate_limit_without_a_wait_backs_off_exponentially():
    policy = BackoffRetry(jitter=full_jitter)
    waits = [policy.delay(RateLimited("slow down"), attempt) for attempt in range(1, MAX_ATTEMPTS)]
    assert waits == [BASE_DELAY * 2 ** step for step in range(MAX_ATTEMPTS - 1)]


@pytest.mark.parametrize("error_type", [Overloaded, ServerFailed, ConnectionFailed])
def test_transient_failures_back_off_with_jitter_between_half_and_the_full_ceiling(error_type):
    low = BackoffRetry(jitter=no_jitter).delay(error_type("down"), 3)
    high = BackoffRetry(jitter=full_jitter).delay(error_type("down"), 3)
    assert (low, high) == (2.0, 4.0)


def test_backoff_never_exceeds_the_delay_cap():
    policy = BackoffRetry(max_attempts=50, jitter=full_jitter)
    assert policy.delay(Overloaded("down"), 40) == MAX_DELAY


def test_no_retry_once_the_attempts_are_used_up():
    policy = BackoffRetry(jitter=no_jitter)
    assert policy.delay(Overloaded("down"), MAX_ATTEMPTS - 1) is not None
    assert policy.delay(Overloaded("down"), MAX_ATTEMPTS) is None


@pytest.mark.parametrize(
    "error_type",
    [QuotaExhausted, AuthFailed, InvalidRequest, OpaqueViolation, ContextOverflow, MalformedStream, HarnessError],
)
def test_these_failures_are_never_retried(error_type):
    assert BackoffRetry(jitter=no_jitter).delay(error_type("no", retry_after=1), 1) is None


def test_a_failure_that_is_not_a_harness_error_is_not_retried():
    assert BackoffRetry().delay(RuntimeError("boom"), 1) is None


def test_the_notice_says_why_and_when_the_next_attempt_comes():
    notice = retry_notice(Overloaded("down", status=529), 2, 4.2, step_seq=3)
    assert notice.kind is NoticeKind.RETRYING
    assert notice.step_seq == 3
    assert "overloaded" in notice.message and "4 seconds" in notice.message and "attempt 3" in notice.message
    assert notice.detail == {"error": "overloaded", "status": 529, "attempt": 2, "delay": 4.2}


async def test_each_retry_raises_a_notice_and_the_step_then_succeeds():
    adapter = ScriptedAdapter(Overloaded("busy"), RateLimited("slow", retry_after=2), text_step("hello"))
    sleeps = Sleeps()
    events = await drain(play(adapter, BackoffRetry(jitter=no_jitter), sleeps))
    notices = [event.notice for event in events if isinstance(event, NoticeRaised)]
    assert [notice.kind for notice in notices] == [NoticeKind.RETRYING, NoticeKind.RETRYING]
    assert [notice.detail["attempt"] for notice in notices] == [1, 2]
    assert sleeps.calls == [0.5, 2.0]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["hello"]
    assert isinstance(events[-1], Finish) and events[-1].reason is FinishReason.STOP
    assert adapter.exhausted


async def test_a_failure_that_keeps_coming_is_raised_after_the_last_attempt():
    adapter = ScriptedAdapter(*[ServerFailed("500") for _ in range(MAX_ATTEMPTS)])
    sleeps = Sleeps()
    seen: list = []
    with pytest.raises(ServerFailed):
        async for event in play(adapter, BackoffRetry(jitter=no_jitter), sleeps):
            seen.append(event)
    assert len(adapter.requests) == MAX_ATTEMPTS
    assert len(sleeps.calls) == MAX_ATTEMPTS - 1
    assert len([event for event in seen if isinstance(event, NoticeRaised)]) == MAX_ATTEMPTS - 1


@pytest.mark.parametrize("error", [AuthFailed("key"), QuotaExhausted("billing"), InvalidRequest("bad")])
async def test_a_failure_that_cannot_succeed_is_raised_at_once(error):
    adapter = ScriptedAdapter(error, text_step("never reached"))
    sleeps = Sleeps()
    with pytest.raises(type(error)):
        await drain(play(adapter, BackoffRetry(jitter=no_jitter), sleeps))
    assert sleeps.calls == []
    assert len(adapter.requests) == 1


async def test_a_stream_that_already_delivered_events_is_not_retried():
    async def half_delivered():
        yield TextDelta(block=0, text="partial")
        raise Overloaded("dropped mid-stream")

    sleeps = Sleeps()
    seen: list = []
    with pytest.raises(Overloaded):
        async for event in retrying(half_delivered, BackoffRetry(jitter=no_jitter), sleep=sleeps):
            seen.append(event)
    assert sleeps.calls == []
    assert seen == [TextDelta(block=0, text="partial")]


async def test_stopping_early_closes_the_step_stream():
    closed: List[bool] = []

    async def endless():
        try:
            yield TextDelta(block=0, text="a")
            yield TextDelta(block=0, text="b")
        finally:
            closed.append(True)

    stream = retrying(endless, BackoffRetry(), sleep=Sleeps())
    assert await anext(stream) == TextDelta(block=0, text="a")
    await stream.aclose()
    assert closed == [True]
