import asyncio

import pytest

from src.features.models.progress_broadcast import (
    MESSAGE_TYPE,
    IndexingProgressBroadcaster,
    indexing_is_active,
)


class Hub:
    def __init__(self, connected=True):
        self.active_connections = {"c1": object()} if connected else {}
        self.messages = []

    async def broadcast(self, message):
        self.messages.append(message)


class Source:
    def __init__(self, **status):
        self.current = {"state": "idle", **status}
        self.version = 0
        self.status_calls = 0

    def status(self):
        self.status_calls += 1
        return dict(self.current)

    def probe(self):
        return self.version

    def change(self, **status):
        self.current = {**self.current, **status}
        self.version += 1


def _broadcaster(source, hub, **kwargs):
    return IndexingProgressBroadcaster(source.status, source.probe, indexing_is_active, hub, **kwargs)


class TestIsActive:
    @pytest.mark.parametrize("status", [
        {"state": "scanning"},
        {"state": "indexing"},
        {"state": "idle", "restart_pending": True},
        {"state": "done", "digests": {"queued": 3, "paused": False}},
    ])
    def test_work_in_progress_is_active(self, status):
        assert indexing_is_active(status) is True

    @pytest.mark.parametrize("status", [
        {"state": "idle"},
        {"state": "done", "digests": {"queued": 0, "paused": False}},
        {"state": "done", "digests": {"queued": 3, "paused": True}},
    ])
    def test_everything_else_is_quiet(self, status):
        assert indexing_is_active(status) is False


class TestTick:
    @pytest.mark.asyncio
    async def test_the_first_tick_with_a_client_pushes_a_snapshot(self):
        source, hub = Source(), Hub()

        assert await _broadcaster(source, hub).tick() is True

        assert hub.messages == [{"type": MESSAGE_TYPE, "status": {"state": "idle"}}]

    @pytest.mark.asyncio
    async def test_nothing_is_computed_or_sent_without_a_client(self):
        source, hub = Source(), Hub(connected=False)

        assert await _broadcaster(source, hub).tick() is False

        assert (source.status_calls, hub.messages) == (0, [])

    @pytest.mark.asyncio
    async def test_a_quiet_library_costs_one_cheap_probe_per_tick(self):
        source, hub = Source(), Hub()
        broadcaster = _broadcaster(source, hub)
        await broadcaster.tick()

        assert await broadcaster.tick() is False
        assert await broadcaster.tick() is False

        assert (source.status_calls, len(hub.messages)) == (1, 1)

    @pytest.mark.asyncio
    async def test_a_changed_probe_pushes_the_new_status(self):
        source, hub = Source(), Hub()
        broadcaster = _broadcaster(source, hub)
        await broadcaster.tick()
        source.change(state="scanning")

        assert await broadcaster.tick() is True

        assert hub.messages[-1]["status"]["state"] == "scanning"

    @pytest.mark.asyncio
    async def test_progress_inside_a_busy_run_is_pushed_without_the_probe_moving(self):
        source, hub = Source(state="indexing", processed=1), Hub()
        broadcaster = _broadcaster(source, hub)
        await broadcaster.tick()
        source.current["processed"] = 2

        assert await broadcaster.tick() is True
        assert await broadcaster.tick() is False

        assert [m["status"]["processed"] for m in hub.messages] == [1, 2]

    @pytest.mark.asyncio
    async def test_the_final_idle_status_is_pushed_once_the_run_ends(self):
        source, hub = Source(state="indexing"), Hub()
        broadcaster = _broadcaster(source, hub)
        await broadcaster.tick()
        source.current["state"] = "done"

        assert await broadcaster.tick() is True
        assert await broadcaster.tick() is False

        assert [m["status"]["state"] for m in hub.messages] == ["indexing", "done"]

    @pytest.mark.asyncio
    async def test_a_client_that_connects_later_gets_a_snapshot_even_if_nothing_changed(self):
        source, hub = Source(), Hub(connected=False)
        broadcaster = _broadcaster(source, hub)
        await broadcaster.tick()
        hub.active_connections = {"c1": object()}

        assert await broadcaster.tick() is True


class TestLoop:
    @pytest.mark.asyncio
    async def test_the_loop_ticks_at_the_busy_rate_while_work_runs_and_the_idle_rate_otherwise(self):
        source, hub = Source(state="indexing"), Hub()
        sleeps = []

        async def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 2:
                source.current["state"] = "idle"
            if len(sleeps) == 4:
                raise asyncio.CancelledError()

        broadcaster = _broadcaster(source, hub, busy_interval=1.0, idle_interval=2.0, sleep=sleep)

        with pytest.raises(asyncio.CancelledError):
            await broadcaster._run()

        assert sleeps == [1.0, 1.0, 2.0, 2.0]

    @pytest.mark.asyncio
    async def test_a_failing_status_does_not_end_the_loop(self):
        hub = Hub()
        calls = []

        def status():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return {"state": "idle"}

        async def sleep(seconds):
            if len(calls) >= 2:
                raise asyncio.CancelledError()

        broadcaster = IndexingProgressBroadcaster(status, lambda: 0, indexing_is_active, hub, sleep=sleep)

        with pytest.raises(asyncio.CancelledError):
            await broadcaster._run()

        assert len(hub.messages) == 1


class TestStartStop:
    @pytest.mark.asyncio
    async def test_a_started_broadcaster_pushes_and_stops_cleanly(self):
        source, hub = Source(), Hub()
        gate = asyncio.Event()

        async def sleep(seconds):
            gate.set()
            await asyncio.Event().wait()

        broadcaster = _broadcaster(source, hub, sleep=sleep)
        broadcaster.start()
        await asyncio.wait_for(gate.wait(), 1)

        assert len(hub.messages) == 1
        task = broadcaster._task
        await broadcaster.stop()
        assert task.cancelled() or task.done()
        assert broadcaster._task is None
