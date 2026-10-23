import pytest

from src.features.models.digest_queue_repository import (
    MAX_ATTEMPTS,
    PRIORITY_BACKGROUND,
    PRIORITY_CHOSEN,
    PRIORITY_DEMAND,
    PRIORITY_TWIN,
    DigestQueueRepository,
)
from src.platform.database import get_cursor
from tests.features.models.digest_support import DigestLibrary

NOW = 1_000.0
STAMP = "2026-01-01T00:00:00"


@pytest.fixture
def library(tmp_path, mock_db):
    return DigestLibrary(tmp_path)


@pytest.fixture
def queue(mock_db):
    return DigestQueueRepository()


def _queue_rows():
    with get_cursor() as cursor:
        cursor.execute("SELECT * FROM model_digest_queue ORDER BY location_id")
        return [dict(row) for row in cursor.fetchall()]


class TestEnqueue:
    def test_only_present_locations_without_a_digest_are_queued(self, library, queue):
        pending = library.add("pending.safetensors", b"a" * 10)
        ready = library.add("ready.safetensors", b"b" * 10, sha256="x" * 64)
        missing = library.add("missing.safetensors", b"c" * 10, status="missing")

        added = queue.enqueue([pending.location_id, ready.location_id, missing.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        assert added == 1
        assert [row["location_id"] for row in _queue_rows()] == [pending.location_id]

    def test_the_row_records_size_priority_and_reason(self, library, queue):
        item = library.add("a.safetensors", b"a" * 123)

        queue.enqueue([item.location_id], PRIORITY_CHOSEN, "pick", STAMP)

        row = _queue_rows()[0]
        assert (row["size"], row["priority"], row["reason"], row["attempts"], row["queued_at"]) == (123, 1, "pick", 0, STAMP)

    def test_a_higher_priority_raises_the_row_and_a_lower_one_never_lowers_it(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        queue.enqueue([item.location_id], PRIORITY_DEMAND, "demand", STAMP)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        row = _queue_rows()[0]
        assert (row["priority"], row["reason"]) == (PRIORITY_DEMAND, "demand")

    def test_raising_the_priority_of_a_backed_off_job_makes_it_runnable_again(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        queue.record_failure(item.location_id, "busy", NOW, (600.0,))

        queue.enqueue([item.location_id], PRIORITY_DEMAND, "demand", STAMP)

        assert queue.next_jobs(1, NOW)[0]["location_id"] == item.location_id

    def test_re_enqueueing_keeps_the_failure_count(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        queue.record_failure(item.location_id, "busy", NOW, ())

        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        assert _queue_rows()[0]["attempts"] == 1

    def test_enqueue_all_pending_skips_what_is_already_queued_or_ready(self, library, queue):
        first = library.add("a.safetensors", b"a" * 10)
        library.add("b.safetensors", b"b" * 20)
        library.add("c.safetensors", b"c" * 30, sha256="y" * 64)
        queue.enqueue([first.location_id], PRIORITY_DEMAND, "demand", STAMP)

        added = queue.enqueue_all_pending(PRIORITY_BACKGROUND, "scan", STAMP)

        assert added == 1
        rows = {row["location_id"]: row for row in _queue_rows()}
        assert len(rows) == 2
        assert rows[first.location_id]["priority"] == PRIORITY_DEMAND

    def test_unknown_ids_are_ignored(self, queue):
        assert queue.enqueue(["nope", ""], PRIORITY_BACKGROUND, "scan", STAMP) == 0


class TestNextJobs:
    def _queue_three(self, library, queue):
        big = library.add("big.safetensors", b"b" * 300)
        small = library.add("small.safetensors", b"s" * 10)
        mid = library.add("mid.safetensors", b"m" * 100)
        queue.enqueue([big.location_id, small.location_id, mid.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        return big, small, mid

    def test_smaller_files_come_first_within_a_priority(self, library, queue):
        big, small, mid = self._queue_three(library, queue)

        order = [row["location_id"] for row in queue.next_jobs(10, NOW)]

        assert order == [small.location_id, mid.location_id, big.location_id]

    def test_a_higher_priority_beats_a_smaller_size(self, library, queue):
        big, small, mid = self._queue_three(library, queue)
        queue.enqueue([big.location_id], PRIORITY_TWIN, "twin", STAMP)
        queue.enqueue([mid.location_id], PRIORITY_DEMAND, "demand", STAMP)

        order = [row["location_id"] for row in queue.next_jobs(10, NOW)]

        assert order == [mid.location_id, big.location_id, small.location_id]

    def test_a_job_carries_what_a_worker_needs(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_CHOSEN, "pick", STAMP)

        job = queue.next_jobs(1, NOW)[0]

        assert (job["model_id"], job["root_id"], job["model_type"], job["rel_path"]) == (
            item.model.id, "r_home", "lora", "a.safetensors",
        )
        assert (job["filename"], job["size"], job["priority"], job["attempts"]) == ("a.safetensors", 10, 1, 0)
        assert job["binding_id"] == library.binding.binding_id
        assert job["mtime_ns"] == item.path.stat().st_mtime_ns

    def test_excluded_ids_are_skipped_and_the_limit_applies(self, library, queue):
        big, small, mid = self._queue_three(library, queue)

        jobs = queue.next_jobs(1, NOW, exclude=[small.location_id])

        assert [row["location_id"] for row in jobs] == [mid.location_id]

    def test_a_backed_off_job_waits_for_its_time(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        queue.record_failure(item.location_id, "busy", NOW, (60.0,))

        assert queue.next_jobs(1, NOW) == []
        assert len(queue.next_jobs(1, NOW + 60)) == 1

    def test_a_job_that_failed_too_often_is_not_offered(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            queue.record_failure(item.location_id, "unreadable", NOW, ())

        assert queue.next_jobs(1, NOW) == []

    def test_rows_for_locations_that_gained_a_digest_or_went_missing_are_not_offered(self, library, queue):
        done = library.add("done.safetensors", b"d" * 10)
        gone = library.add("gone.safetensors", b"g" * 10)
        live = library.add("live.safetensors", b"l" * 10)
        queue.enqueue([done.location_id, gone.location_id, live.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        library.set_digest(done, "z" * 64, "q" * 64)
        library.set_status(gone, "missing")

        assert [row["location_id"] for row in queue.next_jobs(10, NOW)] == [live.location_id]


class TestFailures:
    def test_record_failure_counts_attempts_and_keeps_the_last_error(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        assert queue.record_failure(item.location_id, "first", NOW, (60.0, 600.0)) == 1
        first = queue.get(item.location_id)
        assert queue.record_failure(item.location_id, "second", NOW, (60.0, 600.0)) == 2
        second = queue.get(item.location_id)
        assert queue.record_failure(item.location_id, "third", NOW, (60.0, 600.0)) == 3
        third = queue.get(item.location_id)

        assert (first["attempts"], first["last_error"], first["not_before"]) == (1, "first", NOW + 60)
        assert (second["attempts"], second["last_error"], second["not_before"]) == (2, "second", NOW + 600)
        assert (third["attempts"], third["last_error"], third["not_before"]) == (3, "third", None)

    def test_a_failure_for_a_location_that_is_not_queued_changes_nothing(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)

        assert queue.record_failure(item.location_id, "gone", NOW, (60.0,)) == 0
        assert _queue_rows() == []

    def test_a_long_error_is_cut_to_five_hundred_characters(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        queue.record_failure(item.location_id, "x" * 900, NOW, ())

        assert len(queue.get(item.location_id)["last_error"]) == 500

    def test_failed_rows_are_listed_with_their_error_and_can_be_retried(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        other = library.add("b.safetensors", b"b" * 10)
        queue.enqueue([item.location_id, other.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            queue.record_failure(item.location_id, "unreadable", NOW, ())

        failed = queue.list_failed()
        assert [(row["location_id"], row["last_error"]) for row in failed] == [(item.location_id, "unreadable")]
        assert queue.retry_failed([other.location_id]) == 0
        assert queue.retry_failed() == 1
        assert queue.get(item.location_id)["attempts"] == 0
        assert queue.get(item.location_id)["last_error"] is None
        assert queue.list_failed() == []

    def test_defer_postpones_a_job_without_counting_an_attempt(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        queue.defer(item.location_id, NOW + 60)

        assert queue.get(item.location_id)["attempts"] == 0
        assert queue.next_jobs(1, NOW) == []


class TestPendingListing:
    def _library(self, library, queue):
        self.alpha = library.add("Alpha-Lora.safetensors", b"a" * 30)
        self.beta = library.add("beta_vae.safetensors", b"b" * 10)
        self.gamma = library.add("gamma.ckpt", b"c" * 20)
        ready = library.add("ready.safetensors", b"d" * 5, sha256="z" * 64)
        gone = library.add("gone.safetensors", b"e" * 5, status="missing")
        with get_cursor() as cursor:
            cursor.execute("UPDATE models SET model_type = 'vae' WHERE id = ?", (self.beta.model.id,))
            cursor.execute("UPDATE models SET model_type = 'checkpoint' WHERE id = ?", (self.gamma.model.id,))
        queue.enqueue([self.beta.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        return ready, gone

    def test_every_pending_present_file_is_listed_by_name_with_its_queue_flag(self, library, queue):
        self._library(library, queue)

        rows = queue.list_pending()

        assert [(row["filename"], row["queued"]) for row in rows] == [
            ("Alpha-Lora.safetensors", False), ("beta_vae.safetensors", True), ("gamma.ckpt", False),
        ]
        assert "failed" not in rows[0]
        first = rows[0]
        assert set(first) == {
            "location_id", "model_id", "filename", "model_type", "size", "root_label", "rel_path", "queued",
        }
        assert (first["location_id"], first["model_id"], first["size"], first["root_label"], first["rel_path"]) == (
            self.alpha.location_id, self.alpha.model.id, 30, "r_home", "Alpha-Lora.safetensors",
        )

    def test_the_query_is_a_case_insensitive_substring_of_the_filename(self, library, queue):
        self._library(library, queue)

        assert [r["filename"] for r in queue.list_pending(query="LORA")] == ["Alpha-Lora.safetensors"]
        assert [r["filename"] for r in queue.list_pending(query=" a")] == [
            "Alpha-Lora.safetensors", "beta_vae.safetensors", "gamma.ckpt",
        ]
        assert queue.list_pending(query="nothing") == []

    @pytest.mark.parametrize(
        ("query", "expected"),
        [("a_b", ["a_b.safetensors"]), ("50%", ["50%off.safetensors"]), ("c\\d", ["c\\d.safetensors"])],
        ids=["underscore", "percent", "backslash"],
    )
    def test_wildcards_and_the_escape_character_in_a_query_are_taken_literally(self, library, queue, query, expected):
        for name in ("a_b.safetensors", "axb.safetensors", "50%off.safetensors", "50 off.safetensors",
                     "c\\d.safetensors", "cd.safetensors", "cxd.safetensors"):
            item = library.add(name, name.encode().ljust(12, b"."))
            queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        assert [r["filename"] for r in queue.list_pending(query=query)] == expected
        assert queue.count_pending(query)["total"] == len(expected)

    def test_a_file_that_failed_out_is_listed_but_not_as_queued(self, library, queue):
        self._library(library, queue)
        for _ in range(MAX_ATTEMPTS):
            queue.record_failure(self.beta.location_id, "boom", 0.0, ())

        rows = {row["filename"]: row["queued"] for row in queue.list_pending()}

        assert rows == {"Alpha-Lora.safetensors": False, "beta_vae.safetensors": False, "gamma.ckpt": False}

    def test_the_model_type_and_root_filters_apply(self, library, queue):
        self._library(library, queue)

        assert [r["filename"] for r in queue.list_pending(model_type="vae")] == ["beta_vae.safetensors"]
        assert queue.list_pending(root_id="r_home") != []
        assert queue.list_pending(root_id="elsewhere") == []

    def test_paging_slices_the_ordered_list(self, library, queue):
        self._library(library, queue)

        assert [r["filename"] for r in queue.list_pending(offset=1, limit=1)] == ["beta_vae.safetensors"]
        assert queue.list_pending(offset=3, limit=5) == []

    def test_the_counts_cover_the_whole_filtered_set_not_a_page(self, library, queue):
        self._library(library, queue)

        assert queue.count_pending() == {"total": 3, "total_bytes": 60}
        assert queue.count_pending(query="a") == {"total": 3, "total_bytes": 60}
        assert queue.count_pending(model_type="checkpoint") == {"total": 1, "total_bytes": 20}
        assert queue.count_pending(query="zzz") == {"total": 0, "total_bytes": 0}

    def test_by_type_ignores_the_query_and_the_type_filter_so_chip_counts_stay_put(self, library, queue):
        self._library(library, queue)

        assert queue.pending_by_type() == {"lora": 1, "vae": 1, "checkpoint": 1}
        assert queue.pending_by_type(root_id="elsewhere") == {}

    def test_the_order_is_by_name_ignoring_case_not_by_when_the_files_were_added(self, library, queue):
        library.add("Zulu.safetensors", b"z" * 10)
        library.add("alpha.safetensors", b"a" * 10)
        library.add("Beta.safetensors", b"b" * 10)

        assert [r["filename"] for r in queue.list_pending()] == ["alpha.safetensors", "Beta.safetensors", "Zulu.safetensors"]

    def test_a_ready_or_missing_file_is_never_listed_or_counted(self, library, queue):
        ready, gone = self._library(library, queue)

        listed = {r["location_id"] for r in queue.list_pending()}

        assert ready.location_id not in listed and gone.location_id not in listed
        assert queue.count_pending()["total"] == 3


class TestSelectingPendingLocations:
    def test_the_ids_of_every_match_are_returned_minus_the_excluded_models(self, library, queue):
        first = library.add("a.safetensors", b"a" * 10)
        second = library.add("b.safetensors", b"b" * 10)
        third = library.add("c.safetensors", b"c" * 10)

        everything = queue.pending_location_ids()
        without = queue.pending_location_ids(exclude_model_ids=[second.model.id, ""])
        filtered = queue.pending_location_ids(query="b", exclude_model_ids=[third.model.id])

        assert sorted(everything) == sorted([first.location_id, second.location_id, third.location_id])
        assert sorted(without) == sorted([first.location_id, third.location_id])
        assert filtered == [second.location_id]

    def test_the_locations_of_chosen_models_include_every_pending_copy_and_skip_ready_ones(self, library, queue):
        first = library.add("a.safetensors", b"a" * 10)
        copy = library.add("a.safetensors", b"a" * 10, rel_path="copies/a.safetensors", model=first.model)
        ready = library.add("a.safetensors", b"a" * 10, rel_path="done/a.safetensors", model=first.model, sha256="z" * 64)
        other = library.add("b.safetensors", b"b" * 10)

        ids = queue.pending_location_ids_for_models([first.model.id, "unknown", ""])

        assert sorted(ids) == sorted([first.location_id, copy.location_id])
        assert ready.location_id not in ids and other.location_id not in ids
        assert queue.pending_location_ids_for_models([]) == []


class TestClearingTheBackground:
    def test_only_background_rows_go_and_picked_twin_and_demand_rows_stay(self, library, queue):
        rows = {}
        for index, priority in enumerate((PRIORITY_BACKGROUND, PRIORITY_CHOSEN, PRIORITY_TWIN, PRIORITY_DEMAND)):
            item = library.add(f"m{index}.safetensors", b"x" * 10)
            queue.enqueue([item.location_id], priority, "test", STAMP)
            rows[priority] = item

        removed = queue.clear_background()

        assert removed == 1
        assert {row["location_id"] for row in _queue_rows()} == {
            rows[PRIORITY_CHOSEN].location_id, rows[PRIORITY_TWIN].location_id, rows[PRIORITY_DEMAND].location_id,
        }


class TestFailedBytes:
    def test_the_size_of_files_that_gave_up_is_reported_separately(self, library, queue):
        failed = library.add("f.safetensors", b"f" * 40)
        queued = library.add("q.safetensors", b"q" * 25)
        queue.enqueue([failed.location_id, queued.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            queue.record_failure(failed.location_id, "unreadable", NOW, ())

        assert queue.queue_stats() == {"queued": 1, "queued_bytes": 25, "failed": 1, "failed_bytes": 40}


class TestWaitingAmong:
    def test_only_queued_pending_locations_that_can_still_run_are_waiting(self, library, queue):
        waiting = library.add("waiting.safetensors", b"a" * 10)
        done = library.add("done.safetensors", b"b" * 10)
        gone = library.add("gone.safetensors", b"c" * 10)
        failed = library.add("failed.safetensors", b"d" * 10)
        unqueued = library.add("unqueued.safetensors", b"e" * 10)
        queue.enqueue(
            [waiting.location_id, done.location_id, gone.location_id, failed.location_id],
            PRIORITY_BACKGROUND, "scan", STAMP,
        )
        library.set_digest(done, "z" * 64)
        library.set_status(gone, "missing")
        for _ in range(MAX_ATTEMPTS):
            queue.record_failure(failed.location_id, "unreadable", NOW, ())
        ids = [waiting, done, gone, failed, unqueued]

        assert queue.waiting_among([item.location_id for item in ids]) == [waiting.location_id]

    def test_a_row_that_is_backing_off_is_waiting_only_when_the_clock_is_not_asked(self, library, queue):
        ready = library.add("ready.safetensors", b"a" * 10)
        backing = library.add("backing.safetensors", b"b" * 10)
        due = library.add("due.safetensors", b"c" * 10)
        queue.enqueue([ready.location_id, backing.location_id, due.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        queue.record_failure(backing.location_id, "unreadable", NOW, (60.0,))
        queue.record_failure(due.location_id, "unreadable", NOW, (60.0,))
        ids = [ready.location_id, backing.location_id, due.location_id]

        assert sorted(queue.waiting_among(ids)) == sorted(ids)
        assert sorted(queue.waiting_among(ids, NOW + 30)) == [ready.location_id]
        assert sorted(queue.waiting_among(ids, NOW + 60)) == sorted(ids)

    def test_nothing_waits_when_nothing_is_asked(self, queue):
        assert queue.waiting_among([]) == []


class TestStaleRows:
    def test_delete_stale_removes_rows_whose_location_changed_state(self, library, queue):
        done = library.add("done.safetensors", b"d" * 10)
        gone = library.add("gone.safetensors", b"g" * 10)
        live = library.add("live.safetensors", b"l" * 10)
        queue.enqueue([done.location_id, gone.location_id, live.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        library.set_digest(done, "z" * 64)
        library.set_status(gone, "missing")

        stale = queue.delete_stale()

        assert sorted(stale) == sorted([done.location_id, gone.location_id])
        assert [row["location_id"] for row in _queue_rows()] == [live.location_id]

    def test_removing_a_location_removes_its_queue_row(self, library, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        library.locations.delete(item.location_id)

        assert _queue_rows() == []

    def test_remove_reports_how_many_rows_went(self, library, queue):
        first = library.add("a.safetensors", b"a" * 10)
        second = library.add("b.safetensors", b"b" * 10)
        queue.enqueue([first.location_id, second.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        assert queue.remove([first.location_id, "nope"]) == 1


class TestStats:
    def test_queue_and_digest_totals(self, library, queue):
        queued = library.add("a.safetensors", b"a" * 100)
        library.add("b.safetensors", b"b" * 50)
        library.add("c.safetensors", b"c" * 25, sha256="c" * 64)
        failed = library.add("d.safetensors", b"d" * 5)
        library.add("e.safetensors", b"e" * 7, status="missing")
        queue.enqueue([queued.location_id, failed.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            queue.record_failure(failed.location_id, "unreadable", NOW, ())

        assert queue.queue_stats() == {"queued": 1, "queued_bytes": 100, "failed": 1, "failed_bytes": 5}
        assert queue.digest_stats() == {"pending": 3, "pending_bytes": 155, "ready": 1, "ready_bytes": 25}

    def test_an_empty_library_reports_zeros(self, queue):
        assert queue.queue_stats() == {"queued": 0, "queued_bytes": 0, "failed": 0, "failed_bytes": 0}
        assert queue.digest_stats() == {"pending": 0, "pending_bytes": 0, "ready": 0, "ready_bytes": 0}


class TestLocationAndModelWrites:
    def test_pending_locations_for_a_model_lists_only_unhashed_present_copies(self, library, queue):
        first = library.add("a.safetensors", b"a" * 10)
        library.add("a.safetensors", b"a" * 10, rel_path="copies/a.safetensors", model=first.model, sha256="s" * 64)
        library.add("a.safetensors", b"a" * 10, rel_path="more/a.safetensors", model=first.model, status="missing")

        rows = queue.pending_locations_for_model(first.model.id)

        assert [row["location_id"] for row in rows] == [first.location_id]
        assert rows[0]["filename"] == "a.safetensors"
