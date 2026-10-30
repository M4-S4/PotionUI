import threading
import unittest

from src.features.media_tools.jobs import MediaToolJobManager, RunControl, RunOutcome
from src.features.media_tools.dto import MediaToolItem
from src.features.plans.errors import LimitExceeded
from src.platform.http.errors import NotFoundError
from src.platform.media_tools import ToolCancelled, ToolFailed, UnsupportedMedia

WAIT = 10.0


class Clock:
    def __init__(self):
        self.now = 50.0

    def __call__(self):
        return self.now


def item():
    return MediaToolItem(id="i", filename="f.png", media_type="image", url="/u")


def limit_error():
    return LimitExceeded(
        [{"message": "Full", "code": "c", "kind": "k", "label": "l", "format": "f", "used": 1,
          "limit": 1, "percent": 100, "resets_at": None}],
        "upload",
        None,
    )


class TestJobManager(unittest.TestCase):

    def setUp(self):
        self.clock = Clock()
        self.manager = MediaToolJobManager(max_concurrent=2, retention_seconds=600, clock=self.clock)
        self.addCleanup(self.manager.join_all)

    def finish(self, work, user="u1"):
        run_id = self.manager.submit(user, work)
        return run_id, self.manager.wait(user, run_id, WAIT)

    def test_a_successful_run_completes_with_its_items_and_full_progress(self):
        run_id, run = self.finish(lambda control: RunOutcome(items=[item()], replaced=True))

        self.assertEqual(run.status, "completed")
        self.assertEqual(run.progress, 1.0)
        self.assertTrue(run.replaced)
        self.assertEqual([entry.id for entry in run.items], ["i"])
        self.assertIsNone(run.error)

    def test_progress_never_goes_backwards_and_an_empty_label_keeps_the_last_one(self):
        seen = []
        ready = threading.Event()
        checkpoint = threading.Event()
        proceed = threading.Event()

        def work(control: RunControl):
            ready.wait(WAIT)
            control.report(0.5, "Encoding")
            control.report(0.2, "")
            seen.append(self.manager.get("u1", run_id))
            checkpoint.set()
            proceed.wait(WAIT)
            control.report(0.9, "Saving")
            return RunOutcome()

        run_id = self.manager.submit("u1", work)
        ready.set()
        self.assertTrue(checkpoint.wait(WAIT))
        midway = seen[0]
        proceed.set()
        final = self.manager.wait("u1", run_id, WAIT)

        self.assertEqual((midway.progress, midway.label), (0.5, "Encoding"))
        self.assertEqual(final.status, "completed")

    def test_progress_is_clamped_to_the_unit_interval(self):
        seen = []
        ready = threading.Event()

        def work(control):
            ready.wait(WAIT)
            control.report(7, "x")
            seen.append(self.manager.get("u1", run_id).progress)
            control.report(-3, "")
            return RunOutcome()

        run_id = self.manager.submit("u1", work)
        ready.set()
        self.manager.wait("u1", run_id, WAIT)

        self.assertEqual(seen, [1.0])

    def test_a_media_tool_error_fails_the_run_with_its_own_code_and_message(self):
        def fail(control):
            raise UnsupportedMedia("Needs sound")

        _, run = self.finish(fail)

        self.assertEqual(run.status, "failed")
        self.assertEqual((run.error.code, run.error.message), ("unsupported_media", "Needs sound"))
        self.assertEqual(run.items, [])

    def test_a_tool_failure_keeps_its_code(self):
        def fail(control):
            raise ToolFailed("Failed to encode the media")

        _, run = self.finish(fail)

        self.assertEqual((run.status, run.error.code), ("failed", "tool_failed"))

    def test_a_domain_error_keeps_its_code(self):
        def fail(control):
            raise NotFoundError("not_found", "Media not found")

        _, run = self.finish(fail)

        self.assertEqual((run.status, run.error.code, run.error.message), ("failed", "not_found", "Media not found"))

    def test_a_quota_refusal_reports_limit_exceeded(self):
        def fail(control):
            raise limit_error()

        _, run = self.finish(fail)

        self.assertEqual((run.status, run.error.code, run.error.message), ("failed", "limit_exceeded", "Full"))

    def test_an_unexpected_error_fails_the_run_without_leaking_its_text(self):
        def fail(control):
            raise RuntimeError("/srv/private/token=abc")

        _, run = self.finish(fail)

        self.assertEqual((run.status, run.error.code), ("failed", "tool_failed"))
        self.assertNotIn("/srv", run.error.message)
        self.assertNotIn("abc", run.error.message)

    def test_a_cancelled_exception_ends_the_run_as_cancelled(self):
        def cancelled(control):
            raise ToolCancelled()

        _, run = self.finish(cancelled)

        self.assertEqual(run.status, "cancelled")
        self.assertIsNone(run.error)

    def test_cancel_sets_the_flag_the_work_sees(self):
        entered = threading.Event()

        def work(control):
            entered.set()
            while not control.is_cancelled():
                threading.Event().wait(0.005)
            raise ToolCancelled()

        run_id = self.manager.submit("u1", work)
        self.assertTrue(entered.wait(WAIT))

        run = self.manager.cancel("u1", run_id)

        self.assertEqual(run.status, "cancelled")

    def test_a_cancelled_run_that_has_not_started_never_runs_its_work(self):
        release = threading.Event()
        started = []
        entered = threading.Semaphore(0)

        def blocker(control):
            started.append("blocker")
            entered.release()
            release.wait(WAIT)
            return RunOutcome()

        def victim(control):
            started.append("victim")
            return RunOutcome()

        first = self.manager.submit("u1", blocker)
        second = self.manager.submit("u1", blocker)
        third = self.manager.submit("u1", victim)
        self.assertTrue(entered.acquire(timeout=WAIT))
        self.assertTrue(entered.acquire(timeout=WAIT))
        self.assertEqual(self.manager.get("u1", third).status, "queued")

        cancelled = self.manager.cancel("u1", third)
        release.set()
        self.manager.wait("u1", first, WAIT)
        self.manager.wait("u1", second, WAIT)

        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(started, ["blocker", "blocker"])
        self.assertEqual(self.manager.get("u1", third).status, "cancelled")

    def test_the_cap_holds_and_queued_runs_start_as_slots_free_up(self):
        release = threading.Event()
        entered = threading.Semaphore(0)
        started = []

        def work(control):
            started.append(1)
            entered.release()
            release.wait(WAIT)
            return RunOutcome()

        ids = [self.manager.submit("u1", work) for _ in range(4)]
        self.assertTrue(entered.acquire(timeout=WAIT))
        self.assertTrue(entered.acquire(timeout=WAIT))

        self.assertEqual(self.manager.counts(), {"queued": 2, "running": 2, "finished": 0})
        self.assertEqual(len(started), 2)
        release.set()
        for run_id in ids:
            self.assertEqual(self.manager.wait("u1", run_id, WAIT).status, "completed")

        self.assertEqual(len(started), 4)
        self.assertEqual(self.manager.counts(), {"queued": 0, "running": 0, "finished": 4})

    def test_queued_runs_start_in_submission_order(self):
        manager = MediaToolJobManager(max_concurrent=1, clock=self.clock)
        self.addCleanup(manager.join_all)
        order = []

        def make(name):
            def work(control):
                order.append(name)
                return RunOutcome()
            return work

        ids = [manager.submit("u1", make(name)) for name in ("a", "b", "c", "d")]
        for run_id in ids:
            manager.wait("u1", run_id, WAIT)

        self.assertEqual(order, ["a", "b", "c", "d"])

    def test_a_failure_frees_its_slot_for_the_next_run(self):
        manager = MediaToolJobManager(max_concurrent=1, clock=self.clock)
        self.addCleanup(manager.join_all)

        def boom(control):
            raise RuntimeError("x")

        first = manager.submit("u1", boom)
        second = manager.submit("u1", lambda control: RunOutcome())

        self.assertEqual(manager.wait("u1", first, WAIT).status, "failed")
        self.assertEqual(manager.wait("u1", second, WAIT).status, "completed")

    def test_other_users_see_nothing_of_a_run(self):
        run_id, _ = self.finish(lambda control: RunOutcome())

        self.assertIsNone(self.manager.get("u2", run_id))
        self.assertIsNone(self.manager.cancel("u2", run_id))
        self.assertIsNone(self.manager.wait("u2", run_id, 0.01))
        self.assertEqual(self.manager.get("u1", run_id).status, "completed")

    def test_finished_runs_expire_after_the_retention_window(self):
        run_id, _ = self.finish(lambda control: RunOutcome())

        self.clock.now += 599.9
        self.assertIsNotNone(self.manager.get("u1", run_id))
        self.clock.now += 0.2

        self.assertIsNone(self.manager.get("u1", run_id))
        self.assertEqual(self.manager.counts()["finished"], 0)

    def test_expiry_also_happens_when_a_new_run_is_submitted(self):
        run_id, _ = self.finish(lambda control: RunOutcome())
        self.clock.now += 700

        self.finish(lambda control: RunOutcome())

        self.assertEqual(self.manager.counts()["finished"], 1)
        self.assertIsNone(self.manager.get("u1", run_id))

    def test_expiry_counts_from_when_the_run_finished(self):
        release = threading.Event()
        entered = threading.Event()

        def slow(control):
            entered.set()
            release.wait(WAIT)
            return RunOutcome()

        run_id = self.manager.submit("u1", slow)
        self.assertTrue(entered.wait(WAIT))
        self.clock.now += 1000
        release.set()
        self.manager.wait("u1", run_id, WAIT)

        self.clock.now += 599
        self.assertIsNotNone(self.manager.get("u1", run_id))

    def test_cancelling_a_finished_run_changes_nothing(self):
        run_id, run = self.finish(lambda control: RunOutcome(items=[item()]))

        again = self.manager.cancel("u1", run_id)

        self.assertEqual(again.status, "completed")
        self.assertEqual(len(again.items), 1)

    def test_reports_after_the_run_ended_are_ignored(self):
        controls = []

        def work(control):
            controls.append(control)
            return RunOutcome()

        run_id, _ = self.finish(work)
        controls[0].report(0.1, "late")

        run = self.manager.get("u1", run_id)
        self.assertEqual((run.progress, run.label), (1.0, ""))


if __name__ == "__main__":
    unittest.main()
