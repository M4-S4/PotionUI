import asyncio
import json
import logging
from typing import Any, Callable, Dict, Hashable, Optional

from src.features.models.indexing_coordinator import RUNNING_STATES

logger = logging.getLogger(__name__)

MESSAGE_TYPE = "indexing_status"
BUSY_INTERVAL_SECONDS = 1.0
IDLE_INTERVAL_SECONDS = 2.0


def indexing_is_active(status: Dict[str, Any]) -> bool:
    if status.get("state") in RUNNING_STATES or status.get("restart_pending"):
        return True
    digests = status.get("digests")
    return bool(digests) and digests.get("queued", 0) > 0 and not digests.get("paused")


class IndexingProgressBroadcaster:
    def __init__(
        self,
        status: Callable[[], Dict[str, Any]],
        probe: Callable[[], Hashable],
        is_active: Callable[[Dict[str, Any]], bool],
        hub,
        busy_interval: float = BUSY_INTERVAL_SECONDS,
        idle_interval: float = IDLE_INTERVAL_SECONDS,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ):
        self._status = status
        self._probe = probe
        self._is_active = is_active
        self._hub = hub
        self._busy_interval = busy_interval
        self._idle_interval = idle_interval
        self._sleep = sleep
        self._task: Optional[asyncio.Task] = None
        self._last_payload: Optional[str] = None
        self._last_probe: Optional[Hashable] = None
        self._busy = False

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Pushing the indexing status failed")
            await self._sleep(self._busy_interval if self._busy else self._idle_interval)

    async def tick(self) -> bool:
        if not self._hub.active_connections:
            self._last_payload = None
            self._last_probe = None
            self._busy = False
            return False
        probe = await asyncio.to_thread(self._probe)
        if not self._busy and probe == self._last_probe and self._last_payload is not None:
            return False
        status = await asyncio.to_thread(self._status)
        self._last_probe = probe
        self._busy = self._is_active(status)
        payload = json.dumps(status, sort_keys=True, default=str)
        if payload == self._last_payload:
            return False
        self._last_payload = payload
        await self._hub.broadcast({"type": MESSAGE_TYPE, "status": json.loads(payload)})
        return True
