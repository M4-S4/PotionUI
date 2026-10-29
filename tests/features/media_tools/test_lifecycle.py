import asyncio
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from src.features.media_tools import service as service_module
from src.features.media_tools.jobs import MediaToolJobManager, RunOutcome
from src.features.media_tools.service import MediaToolService
from src.platform.http.errors import DomainError
from src.platform.media_tools import ToolCancelled
from tests.features.media_tools.support import MediaToolsTestBase, WAIT_SECONDS, needs_ffmpeg


class Blocker:
    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Semaphore(0)

    def work(self, control):
        self.started.release()
        while not self.release.wait(0.01):
            if control.is_cancelled():
                raise RuntimeError("cancelled")
        return RunOutcome()


class TestJobManagerShutdownAndQueue(unittest.TestCase):

    def test_shutdown_cancels_queued_runs_without_starting_them(self):
        blocker = Blocker()
        jobs = MediaToolJobManager(max_concurrent=1)
        first = jobs.submit("u", blocker.work)
        queued = jobs.submit("u", blocker.work)
        self.assertTrue(blocker.started.acquire(timeout=5))

        jobs.shutdown(timeout=2)

        self.assertEqual(jobs.get("u", queued).status, "cancelled")
        self.assertEqual(jobs.get("u", first).status, "failed")
        self.assertFalse(blocker.started.acquire(timeout=0.2))

    def test_queued_runs_are_cancelled_the_moment_shutdown_returns_even_if_a_running_one_ignores_it(self):
        release = threading.Event()
        started = threading.Semaphore(0)
        jobs = MediaToolJobManager(max_concurrent=1)
        jobs.submit("u", lambda control: (started.release(), release.wait(10), RunOutcome())[2])
        self.assertTrue(started.acquire(timeout=5))
        queued = jobs.submit("u", lambda control: RunOutcome())

        jobs.shutdown(timeout=0)

        self.assertEqual(jobs.get("u", queued).status, "cancelled")
        release.set()

    def test_shutdown_sets_the_cancel_flag_of_running_runs_and_waits_for_them(self):
        seen = {}

        def work(control):
            while not control.is_cancelled():
                time.sleep(0.01)
            time.sleep(0.2)
            seen["finished"] = True
            raise ToolCancelled()

        jobs = MediaToolJobManager()
        run_id = jobs.submit("u", work)

        jobs.shutdown(timeout=5)

        self.assertTrue(seen.get("finished"))
        self.assertEqual(jobs.get("u", run_id).status, "cancelled")

    def test_shutdown_gives_up_after_its_timeout(self):
        blocker = Blocker()
        jobs = MediaToolJobManager()
        jobs.submit("u", lambda control: (blocker.started.release(), blocker.release.wait(10), RunOutcome())[2])
        self.assertTrue(blocker.started.acquire(timeout=5))
        started = time.monotonic()

        jobs.shutdown(timeout=0.3)

        self.assertLess(time.monotonic() - started, 3)
        blocker.release.set()

    def test_a_closed_manager_refuses_new_runs(self):
        jobs = MediaToolJobManager()
        jobs.shutdown(timeout=0)

        with self.assertRaises(DomainError) as raised:
            jobs.submit("u", lambda control: RunOutcome())

        self.assertEqual(raised.exception.code, "shutting_down")
        self.assertEqual(raised.exception.status_code, 503)

    def test_a_user_cannot_queue_more_than_the_cap_but_others_can(self):
        blocker = Blocker()
        jobs = MediaToolJobManager(max_concurrent=1, max_queued_per_user=2)
        jobs.submit("u", blocker.work)
        self.assertTrue(blocker.started.acquire(timeout=5))
        jobs.submit("u", blocker.work)
        jobs.submit("u", blocker.work)

        with self.assertRaises(DomainError) as raised:
            jobs.submit("u", blocker.work)

        self.assertEqual(raised.exception.code, "too_many_queued")
        self.assertEqual(raised.exception.status_code, 429)
        other = jobs.submit("v", blocker.work)
        self.assertEqual(jobs.get("v", other).status, "queued")
        blocker.release.set()
        jobs.join_all()

    def test_a_cancelled_queued_run_frees_a_queue_slot(self):
        blocker = Blocker()
        jobs = MediaToolJobManager(max_concurrent=1, max_queued_per_user=1)
        jobs.submit("u", blocker.work)
        self.assertTrue(blocker.started.acquire(timeout=5))
        waiting = jobs.submit("u", blocker.work)
        jobs.cancel("u", waiting)

        replacement = jobs.submit("u", blocker.work)

        self.assertEqual(jobs.get("u", replacement).status, "queued")
        blocker.release.set()
        jobs.join_all()

    def test_on_done_fires_when_the_run_finishes_and_at_once_when_it_already_has(self):
        jobs = MediaToolJobManager()
        fired = threading.Event()
        gate = threading.Event()
        run_id = jobs.submit("u", lambda control: (gate.wait(5), RunOutcome())[1])

        self.assertTrue(jobs.on_done("u", run_id, fired.set))
        self.assertFalse(fired.is_set())
        gate.set()
        self.assertTrue(fired.wait(5))
        late = threading.Event()
        self.assertTrue(jobs.on_done("u", run_id, late.set))
        self.assertTrue(late.is_set())

    def test_on_done_fires_when_a_queued_run_is_cancelled_and_ignores_foreign_runs(self):
        blocker = Blocker()
        jobs = MediaToolJobManager(max_concurrent=1)
        jobs.submit("u", blocker.work)
        self.assertTrue(blocker.started.acquire(timeout=5))
        waiting = jobs.submit("u", blocker.work)
        fired = threading.Event()

        self.assertFalse(jobs.on_done("intruder", waiting, fired.set))
        self.assertTrue(jobs.on_done("u", waiting, fired.set))
        jobs.cancel("u", waiting)

        self.assertTrue(fired.is_set())
        self.assertFalse(jobs.on_done("u", "unknown", fired.set))
        blocker.release.set()
        jobs.join_all()


class TestServiceShutdownAndScratch(MediaToolsTestBase):

    def user(self):
        return self.make_user(self.user_id)

    @needs_ffmpeg
    def test_shutdown_kills_a_running_encode_and_removes_its_scratch_dir(self):
        item = self.image_item()
        token = self.registry.get("longencode").impl.token
        run_id = self.start_ok([{"tool": "longencode"}], [{"item_id": item.id}])
        self.wait_for(lambda: self.encode_processes(token))
        self.assertEqual(len(self.scratch_entries()), 1)

        started = time.monotonic()
        self.service.shutdown(timeout=8)

        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual(self.jobs.get(self.user_id, run_id).status, "cancelled")
        self.assertEqual(self.encode_processes(token), [])
        self.assertEqual(self.scratch_entries(), [])
        self.assertEqual(len(self.upload_rows()), 1)

    def test_shutdown_cancels_queued_runs_and_later_starts_are_refused(self):
        item = self.image_item()
        for _ in range(2):
            self.start_ok([{"tool": "block"}], [{"item_id": item.id}])
            self.assertTrue(self.gate.entered.acquire(timeout=WAIT_SECONDS))
        queued = self.start_ok([{"tool": "block"}], [{"item_id": item.id}])

        self.service.shutdown(timeout=5)

        self.assertEqual(self.service.get_run(self.user(), queued).status, "cancelled")
        self.assertEqual(len(self.gate.seen), 2)
        refused = self.start([{"tool": "block"}], [{"item_id": item.id}])
        self.assertEqual(refused.status_code, 503)
        self.assertEqual(self.error_of(refused)["error"], "shutting_down")
        self.assertEqual(self.scratch_entries(), [])

    def test_shutdown_removes_scratch_dirs_that_a_worker_left_behind(self):
        with self.service._scratch() as path:
            (path / "partial.mp4").write_bytes(b"x")
            self.service.shutdown(timeout=0)
            self.assertFalse(path.exists())

    def test_sweep_removes_only_stale_media_tool_directories(self):
        stale = self.scratch_root / "media-tool-stale"
        fresh = self.scratch_root / "media-tool-fresh"
        foreign = self.scratch_root / "other-stale"
        plain_file = self.scratch_root / "media-tool-file"
        for directory in (stale, fresh, foreign):
            directory.mkdir()
            (directory / "x.bin").write_bytes(b"x")
        plain_file.write_bytes(b"x")
        old = time.time() - 7200
        for path in (stale, foreign, plain_file):
            os.utime(path, (old, old))

        removed = self.service.sweep_scratch(max_age=3600)

        self.assertEqual(removed, 1)
        self.assertEqual(self.scratch_entries(), ["media-tool-file", "media-tool-fresh", "other-stale"])

    def test_sweep_never_touches_a_live_run_scratch_dir(self):
        with self.service._scratch() as path:
            old = time.time() - 99999
            os.utime(path, (old, old))

            self.assertEqual(self.service.sweep_scratch(max_age=60), 0)

            self.assertTrue(path.exists())

    def test_a_new_service_sweeps_stale_scratch_dirs_at_startup(self):
        stale = self.scratch_root / "media-tool-leftover"
        stale.mkdir()
        old = time.time() - 7 * 3600
        os.utime(stale, (old, old))

        MediaToolService(
            registry=self.registry,
            params_repository=self.params_repo,
            upload_repository=self.upload_repo,
            file_repository=self.file_repo,
            media_type_resolver=self.service.publisher.media_types,
            storage_driver=self.driver,
            jobs=MediaToolJobManager(),
            scratch_root=self.scratch_root,
        )

        self.assertFalse(stale.exists())

    def test_the_default_sweep_age_is_hours_not_minutes(self):
        self.assertGreaterEqual(service_module.STALE_SCRATCH_SECONDS, 3600)

    def test_a_cap_on_outputs_fails_the_run_before_anything_is_published(self):
        item = self.image_item()

        with mock.patch.object(service_module, "MAX_OUTPUTS", 3):
            run = self.finish([{"tool": "fanmany", "params": {"n": 4}}], [{"item_id": item.id}])

        self.assertEqual(run.status, "failed")
        self.assertEqual(run.error.code, "too_many_outputs")
        self.assertIn("4", run.error.message)
        self.assertEqual(len(self.upload_rows()), 1)
        self.assertEqual(self.stored_files(), [item.filename])
        self.assertEqual(self.scratch_entries(), [])

    def test_outputs_at_the_cap_are_published(self):
        item = self.image_item()

        with mock.patch.object(service_module, "MAX_OUTPUTS", 3):
            run = self.finish([{"tool": "fanmany", "params": {"n": 3}}], [{"item_id": item.id}])

        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.items), 3)

    def test_the_output_cap_is_five_hundred(self):
        self.assertEqual(service_module.MAX_OUTPUTS, 500)


class TestRunSyncWaiting(MediaToolsTestBase):

    def user(self):
        return self.make_user(self.user_id)

    def test_run_sync_waiting_does_not_park_executor_threads(self):
        item = self.image_item()
        steps = [{"tool": "block"}]
        inputs = [{"item_id": item.id}]

        async def scenario():
            loop = asyncio.get_running_loop()
            loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
            waiters = [
                asyncio.ensure_future(self.service.run_sync(self.user(), steps, inputs, None, timeout=30))
                for _ in range(3)
            ]
            await asyncio.sleep(0.5)
            probe = await asyncio.wait_for(asyncio.to_thread(lambda: 42), 3)
            self.gate.release.set()
            results = await asyncio.wait_for(asyncio.gather(*waiters), 20)
            return probe, results

        probe, results = asyncio.run(scenario())

        self.assertEqual(probe, 42)
        self.assertEqual([result.status for result in results], ["completed"] * 3)

    def test_run_sync_times_out_and_cancels_without_a_polling_thread(self):
        item = self.image_item()

        async def scenario():
            return await self.service.run_sync(
                self.user(), [{"tool": "block"}], [{"item_id": item.id}], None, timeout=0.3
            )

        result = asyncio.run(scenario())

        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.error.code, "timeout")

    def test_cancelling_the_awaiting_task_cancels_the_run(self):
        item = self.image_item()

        async def scenario():
            task = asyncio.ensure_future(
                self.service.run_sync(self.user(), [{"tool": "block"}], [{"item_id": item.id}], None, timeout=30)
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())

        run_id = next(iter(self.jobs._runs))
        self.assertEqual(self.jobs.get(self.user_id, run_id).status, "cancelled")

    def test_run_sync_returns_the_items_of_a_finished_run(self):
        item = self.image_item()

        async def scenario():
            return await self.service.run_sync(
                self.user(), [{"tool": "shrink", "params": {"size": 9}}], [{"item_id": item.id}]
            )

        result = asyncio.run(scenario())

        self.assertEqual(result.status, "completed")
        self.assertEqual([(i.width, i.height) for i in result.items], [(9, 9)])

    def test_the_queue_cap_surfaces_through_the_route_as_429(self):
        item = self.image_item()
        self.jobs = MediaToolJobManager(max_concurrent=1, max_queued_per_user=1, clock=self.clock)
        self.service = self.make_service(self.driver)
        self.client.close()
        self.client = self.make_client(self.service)
        self.start_ok([{"tool": "block"}], [{"item_id": item.id}])
        self.assertTrue(self.gate.entered.acquire(timeout=WAIT_SECONDS))
        self.start_ok([{"tool": "block"}], [{"item_id": item.id}])

        refused = self.start([{"tool": "block"}], [{"item_id": item.id}])

        self.assertEqual(refused.status_code, 429)
        self.assertEqual(self.error_of(refused)["error"], "too_many_queued")


if __name__ == "__main__":
    unittest.main()
