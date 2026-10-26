import asyncio
import random
from contextlib import aclosing
from typing import AsyncIterator, Awaitable, Callable, Optional

from src.features.llm.harness.errors import (
    ConnectionFailed,
    HarnessError,
    Overloaded,
    RateLimited,
    ServerFailed,
)
from src.features.llm.harness.ports import RetryPolicy
from src.features.llm.harness.turn_state import EngineEvent, Notice, NoticeKind, NoticeRaised

MAX_ATTEMPTS = 5
BASE_DELAY = 1.0
MAX_DELAY = 30.0
RETRY_AFTER_CAP = 60.0
RETRY_AFTER_JITTER = 1.0

REASONS = (
    (RateLimited, "The provider is rate limiting requests."),
    (Overloaded, "The provider is overloaded."),
    (ServerFailed, "The provider returned a server error."),
    (ConnectionFailed, "The connection to the provider failed."),
)


class BackoffRetry:
    def __init__(
        self,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY,
        max_delay: float = MAX_DELAY,
        retry_after_cap: float = RETRY_AFTER_CAP,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._retry_after_cap = retry_after_cap
        self._jitter = jitter

    def delay(self, error: HarnessError, attempt: int) -> Optional[float]:
        if not isinstance(error, HarnessError) or not error.retryable or attempt >= self.max_attempts:
            return None
        if error.retry_after is not None:
            if error.retry_after > self._retry_after_cap:
                return None
            return max(error.retry_after, 0.0) + self._jitter() * RETRY_AFTER_JITTER
        ceiling = min(self._max_delay, self._base_delay * 2 ** max(attempt - 1, 0))
        return ceiling / 2 + self._jitter() * ceiling / 2


def retry_notice(error: HarnessError, attempt: int, delay: float, step_seq: Optional[int] = None) -> Notice:
    reason = next((text for kind, text in REASONS if isinstance(error, kind)), "The request failed.")
    seconds = round(delay)
    when = f"in {seconds} second{'s' if seconds != 1 else ''}" if seconds >= 1 else "shortly"
    return Notice(
        kind=NoticeKind.RETRYING,
        message=f"{reason} Retrying {when} (attempt {attempt + 1}).",
        step_seq=step_seq,
        detail={"error": error.kind, "status": error.status, "attempt": attempt, "delay": round(delay, 3)},
    )


async def retrying(
    open_step: Callable[[], AsyncIterator[EngineEvent]],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    step_seq: Optional[int] = None,
) -> AsyncIterator[EngineEvent]:
    attempt = 0
    while True:
        attempt += 1
        emitted = False
        try:
            async with aclosing(open_step()) as stream:
                async for event in stream:
                    emitted = True
                    yield event
            return
        except HarnessError as error:
            wait = None if emitted else policy.delay(error, attempt)
            if wait is None:
                raise
            failure = error
        yield NoticeRaised(retry_notice(failure, attempt, wait, step_seq))
        await sleep(wait)
