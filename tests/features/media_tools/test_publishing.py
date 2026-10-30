import hashlib
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from src.features.generation.thumbnail_profile import SIZE_ORDER, load_thumbnail_profile, profile_hash
from src.features.media.media_types import MediaTypeResolver
from src.features.media.store import MediaStore
from src.features.media.upload_thumbnails import render_image_thumbnails
from src.features.media_tools.publisher import MediaToolPublisher
from src.features.media_tools.service import _safe_stem
from src.platform.http.errors import ConflictError
from src.platform.media_tools import MediaTool, ToolMedia
from src.platform.media_tools.ops import register_core_tools
from tests.features.media_tools.support import MediaToolsTestBase, WAIT_SECONDS, needs_ffmpeg
from tests.features.media_tools.test_service import FailingRepository
from tests.platform.media_tools.media_factory import make_image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PublishingBase(MediaToolsTestBase):

    def setUp(self):
        self.pending = []
        super().setUp()
        register_core_tools(self.registry)

    def make_service(self, driver):
        service = super().make_service(driver)
        service.publisher.thumbnail_runner = self.pending.append
        return service

    def profile(self):
        return load_thumbnail_profile(self.service.publisher.settings)

    def row(self, item):
        return self.upload_repo.get_by_id(item.id, self.user_id)

    def thumbnail_paths(self, row):
        return [path for path in (row.thumbnail_small, row.thumbnail_medium, row.thumbnail_large) if path]

    def assert_profile_sizes(self, row):
        present = {size for size in SIZE_ORDER if getattr(row, f"thumbnail_{size}")}
        self.assertEqual(present, set(self.profile().sizes))
        self.assertTrue(present)

    def thumbnail_exists(self, relative):
        return (self.uploads_dir / relative).is_file()

    def give_image_thumbnails(self, item):
        with Image.open(self.uploads_dir / item.filename) as image:
            image.load()
            paths = render_image_thumbnails(self.driver, self.profile(), image, Path(item.filename).stem)
        self.upload_repo.set_thumbnail_paths(
            item.id, paths.get("small"), paths.get("medium"), paths.get("large"), profile_hash(self.profile())
        )
        return self.row(item)

    def give_fake_video_thumbnails(self, item):
        directory = self.uploads_dir / "thumbnails"
        directory.mkdir(exist_ok=True)
        stem = Path(item.filename).stem
        for suffix in ("small", "medium", "large", "small_animated", "medium_animated", "large_animated"):
            (directory / f"{stem}_{suffix}.webp").write_bytes(b"old thumbnail")
        self.upload_repo.set_thumbnail_paths(
            item.id,
            f"thumbnails/{stem}_small.webp",
            f"thumbnails/{stem}_medium.webp",
            f"thumbnails/{stem}_large.webp",
            "old-profile",
        )
        return self.row(item)

    def run_pending(self):
        workers, self.pending[:] = list(self.pending), []
        for worker in workers:
            worker()


class TestNewResultsLookLikeUploads(PublishingBase):

    def test_a_new_image_result_has_all_three_thumbnails_stored(self):
        item = self.image_item(size=(80, 60))

        run = self.finish([{"tool": "shrink", "params": {"size": 20}}], [{"item_id": item.id}])

        row = self.upload_repo.get_by_id(run.items[0].id, self.user_id)
        self.assert_profile_sizes(row)
        self.assertTrue(all(self.thumbnail_exists(path) for path in self.thumbnail_paths(row)))
        self.assertEqual(row.thumbnail_profile, profile_hash(self.profile()))
        for size in self.profile().sizes:
            self.assertEqual(
                Path(getattr(row, f"thumbnail_{size}")).stem, f"{Path(row.filename).stem}_{size}"
            )

    def test_run_items_carry_thumbnail_urls_for_the_sizes_that_exist(self):
        item = self.image_item(size=(80, 60))

        run = self.finish([{"tool": "shrink", "params": {"size": 20}}], [{"item_id": item.id}])

        entry = run.items[0]
        row = self.upload_repo.get_by_id(entry.id, self.user_id)
        for size in SIZE_ORDER:
            expected = f"/api/media/uploads/{row.filename}?size={size}" if getattr(row, f"thumbnail_{size}") else None
            self.assertEqual(getattr(entry, f"thumbnail_{size}"), expected)
        self.assertTrue(any(getattr(entry, f"thumbnail_{size}") for size in SIZE_ORDER))

    def test_a_replaced_items_run_item_points_at_the_new_thumbnails(self):
        item = self.image_item(size=(80, 60))
        self.give_image_thumbnails(item)

        run = self.finish([{"tool": "shrink", "params": {"size": 20}}], [{"item_id": item.id}], mode="replace")

        entry = run.items[0]
        after = self.row(item)
        self.assertEqual(entry.filename, after.filename)
        self.assertIn(after.filename, next(url for url in (entry.thumbnail_small, entry.thumbnail_medium) if url))

    @needs_ffmpeg
    def test_a_video_run_item_has_no_thumbnail_urls_until_the_async_path_has_run(self):
        item = self.video_item(seconds=1.0)

        run = self.finish([{"tool": "rotate", "params": {"degrees": 90}}], [{"item_id": item.id}])

        entry = run.items[0]
        self.assertEqual([entry.thumbnail_small, entry.thumbnail_medium, entry.thumbnail_large], [None, None, None])

    def test_a_new_result_records_the_content_hash_and_the_upload_purpose(self):
        item = self.image_item()

        run = self.finish([{"tool": "shrink", "params": {"size": 12}}], [{"item_id": item.id}])

        row = self.upload_repo.get_by_id(run.items[0].id, self.user_id)
        self.assertEqual(row.content_hash, digest(self.uploads_dir / row.filename))
        self.assertEqual(row.purpose, "user_upload")

    def test_a_result_with_a_thumbnail_less_image_still_publishes(self):
        item = self.image_item()
        publisher = self.service.publisher
        publisher._image_thumbnails = lambda profile, media, stem: {}

        run = self.finish([{"tool": "shrink", "params": {"size": 12}}], [{"item_id": item.id}])

        row = self.upload_repo.get_by_id(run.items[0].id, self.user_id)
        self.assertEqual(self.thumbnail_paths(row), [])
        self.assertIsNone(row.thumbnail_profile)

    @needs_ffmpeg
    def test_a_new_video_result_gets_its_thumbnails_through_the_async_path(self):
        item = self.video_item(seconds=1.0)

        run = self.finish([{"tool": "rotate", "params": {"degrees": 90}}], [{"item_id": item.id}])

        row = self.upload_repo.get_by_id(run.items[0].id, self.user_id)
        self.assertEqual(self.thumbnail_paths(row), [])
        self.assertEqual(len(self.pending), 1)
        self.run_pending()
        row = self.upload_repo.get_by_id(run.items[0].id, self.user_id)
        self.assert_profile_sizes(row)
        self.assertTrue(all(self.thumbnail_exists(path) for path in self.thumbnail_paths(row)))
        self.assertEqual(row.thumbnail_profile, profile_hash(self.profile()))
        self.assertEqual(row.content_hash, digest(self.uploads_dir / row.filename))

    @needs_ffmpeg
    def test_a_new_audio_result_has_no_thumbnails_and_schedules_nothing(self):
        item = self.audio_item(seconds=1.0)

        run = self.finish([{"tool": "speed", "params": {"speed": 2}}], [{"item_id": item.id}])

        row = self.upload_repo.get_by_id(run.items[0].id, self.user_id)
        self.assertEqual(self.thumbnail_paths(row), [])
        self.assertEqual(self.pending, [])
        self.assertEqual(row.content_hash, digest(self.uploads_dir / row.filename))

    def test_a_row_failure_part_way_removes_the_thumbnails_already_stored(self):
        publisher = MediaToolPublisher(
            FailingRepository(self.upload_repo, fail_on=2), MediaTypeResolver(), self.driver, self.guard
        )
        directory = Path(self.temp_dir) / "outputs"
        directory.mkdir(exist_ok=True)
        medias = [
            ToolMedia(path=make_image(directory / f"o{i}.png", (30, 20)), kind="image", width=30, height=20)
            for i in range(3)
        ]

        with self.assertRaises(RuntimeError):
            publisher.publish_new(self.user_id, medias, ["a.png", "b.png", "c.png"])

        self.assertEqual(self.stored_files(), [])
        self.assertEqual(self.thumbnail_files(), [])
        self.assertEqual(len(self.upload_rows()), 0)


class TestReplaceKeepsThumbnailsInSync(PublishingBase):

    def test_replacing_an_image_swaps_the_thumbnails_and_deletes_the_old_files(self):
        item = self.image_item(size=(80, 60))
        before = self.give_image_thumbnails(item)
        old_paths = self.thumbnail_paths(before)
        self.assertTrue(all(self.thumbnail_exists(path) for path in old_paths))

        run = self.finish([{"tool": "shrink", "params": {"size": 20}}], [{"item_id": item.id}], mode="replace")

        after = self.row(item)
        self.assertEqual(run.items[0].id, item.id)
        new_paths = self.thumbnail_paths(after)
        self.assert_profile_sizes(after)
        self.assertTrue(set(new_paths).isdisjoint(old_paths))
        self.assertTrue(all(self.thumbnail_exists(path) for path in new_paths))
        self.assertFalse(any(self.thumbnail_exists(path) for path in old_paths))
        self.assertEqual(after.thumbnail_profile, profile_hash(self.profile()))
        self.assertEqual(after.content_hash, digest(self.uploads_dir / after.filename))
        self.assertNotEqual(after.filename, item.filename)
        self.assertFalse((self.uploads_dir / item.filename).exists())

    def test_replace_refreshes_the_content_hash_so_old_bytes_no_longer_dedupe_to_the_row(self):
        item = self.image_item()
        original_hash = digest(self.uploads_dir / item.filename)
        self.upload_repo.set_thumbnail_paths(item.id, None, None, None, None)
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE uploads SET content_hash = ? WHERE id = ?", (original_hash, item.id))

        self.finish([{"tool": "shrink", "params": {"size": 9}}], [{"item_id": item.id}], mode="replace")

        self.assertIsNone(self.upload_repo.find_by_hash(self.user_id, original_hash, "user_upload"))
        self.assertEqual(self.row(item).content_hash, digest(self.uploads_dir / self.row(item).filename))

    @needs_ffmpeg
    def test_replacing_a_video_clears_the_columns_removes_old_files_then_the_async_path_fills_them(self):
        item = self.video_item(seconds=1.0)
        before = self.give_fake_video_thumbnails(item)
        old_paths = self.thumbnail_paths(before)

        self.finish([{"tool": "rotate", "params": {"degrees": 180}}], [{"item_id": item.id}], mode="replace")

        cleared = self.row(item)
        self.assertEqual(self.thumbnail_paths(cleared), [])
        self.assertIsNone(cleared.thumbnail_profile)
        self.assertEqual(self.thumbnail_files(), [])
        self.assertEqual(len(self.pending), 1)
        self.run_pending()
        filled = self.row(item)
        self.assert_profile_sizes(filled)
        self.assertTrue(all(self.thumbnail_exists(path) for path in self.thumbnail_paths(filled)))
        self.assertTrue(set(self.thumbnail_paths(filled)).isdisjoint(old_paths))
        self.assertEqual(filled.thumbnail_profile, profile_hash(self.profile()))

    def test_the_replaced_items_new_filename_serves_its_medium_thumbnail(self):
        item = self.image_item(size=(80, 60))
        self.give_image_thumbnails(item)
        self.finish([{"tool": "shrink", "params": {"size": 20}}], [{"item_id": item.id}], mode="replace")
        after = self.row(item)
        store = MediaStore(
            file_resolver=Mock(),
            image_processor=Mock(),
            media_type_resolver=MediaTypeResolver(),
            file_repository=Mock(),
            generation_repository=Mock(),
            settings=Mock(),
            file_service=Mock(),
            plugin_registry=Mock(),
            upload_repository=self.upload_repo,
            storage_driver=self.driver,
        )

        served = store.get_uploaded_media(after.filename, user_id=self.user_id, size="medium")

        self.assertEqual(Path(served.file_path).name, Path(after.thumbnail_medium).name)
        self.assertTrue(Path(served.file_path).is_file())
        self.assertEqual(served.media_type, "image/webp")
        with self.assertRaises(ValueError):
            store.get_uploaded_media(item.filename, user_id=self.user_id, size="medium")


class TestReplaceConflicts(PublishingBase):

    def publisher(self):
        return self.service.publisher

    def output(self, name="new.png", size=(9, 9)):
        directory = Path(self.temp_dir) / "outputs"
        directory.mkdir(exist_ok=True)
        path = make_image(directory / name, size)
        return ToolMedia(path=path, kind="image", width=size[0], height=size[1])

    def test_a_stale_replace_fails_with_a_conflict_and_cleans_up_without_admitting_quota(self):
        item = self.image_item(size=(40, 30))
        stale = self.row(item)
        first = self.publisher().publish_replace(self.user_id, stale, f"uploads/{item.filename}", self.output("a.png"))
        winner_files = self.stored_files()
        winner_thumbs = self.thumbnail_files()
        admits = len(self.guard.admits)

        with self.assertRaises(ConflictError) as raised:
            self.publisher().publish_replace(
                self.user_id, stale, f"uploads/{item.filename}", self.output("b.png", (11, 11))
            )

        self.assertEqual(raised.exception.code, "conflict")
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(len(self.guard.admits), admits)
        self.assertEqual(self.stored_files(), winner_files)
        self.assertEqual(self.thumbnail_files(), winner_thumbs)
        current = self.row(item)
        self.assertEqual(current.filename, first.filename)
        self.assertTrue((self.uploads_dir / current.filename).is_file())
        self.assertEqual((current.width, current.height), (9, 9))

    def test_a_row_changed_between_the_check_and_the_update_is_still_a_conflict(self):
        item = self.image_item(size=(40, 30))
        stale = self.row(item)
        publisher = self.publisher()
        real = self.upload_repo.get_by_id
        calls = {"n": 0}

        def check_then_change(upload_id, user_id):
            calls["n"] += 1
            found = real(upload_id, user_id)
            if calls["n"] == 1:
                with self.db.get_cursor() as cursor:
                    cursor.execute("UPDATE uploads SET filename = ? WHERE id = ?", ("rival.png", upload_id))
            return found

        publisher.upload_repo = Mock(wraps=self.upload_repo)
        publisher.upload_repo.get_by_id = check_then_change

        with self.assertRaises(ConflictError):
            publisher.publish_replace(self.user_id, stale, f"uploads/{item.filename}", self.output())

        self.assertEqual(self.stored_files(), [item.filename])
        self.assertEqual(self.thumbnail_files(), [])
        self.assertEqual(self.row(item).filename, "rival.png")

    def test_two_runs_replacing_one_item_lose_nothing_and_leak_nothing(self):
        item = self.image_item(size=(40, 30))
        first = self.start_ok([{"tool": "block"}], [{"item_id": item.id}], mode="replace")
        second = self.start_ok([{"tool": "block"}], [{"item_id": item.id}], mode="replace")
        self.assertTrue(self.gate.entered.acquire(timeout=WAIT_SECONDS))
        self.assertTrue(self.gate.entered.acquire(timeout=WAIT_SECONDS))

        self.gate.release.set()
        runs = [self.wait_run(first), self.wait_run(second)]

        statuses = sorted(run.status for run in runs)
        self.assertEqual(statuses, ["completed", "failed"])
        failed = next(run for run in runs if run.status == "failed")
        self.assertEqual(failed.error.code, "conflict")
        self.assertEqual(len(self.upload_rows()), 1)
        current = self.row(item)
        self.assertEqual(self.stored_files(), [current.filename])
        self.assertEqual(self.scratch_entries(), [])

    def test_a_replace_whose_row_vanished_is_not_found_not_a_conflict(self):
        item = self.image_item()
        stale = self.row(item)
        self.upload_repo.delete(item.filename, self.user_id)

        with self.assertRaises(Exception) as raised:
            self.publisher().publish_replace(self.user_id, stale, f"uploads/{item.filename}", self.output())

        self.assertEqual(raised.exception.code, "not_found")


class TestMessagesNamesAndLimits(PublishingBase):

    def test_safe_stem_strips_separators_and_extensions(self):
        self.assertEqual(_safe_stem("take 1/2.wav"), "take 1-2")
        self.assertEqual(_safe_stem("a\\b\\c.mp4"), "a-b-c")
        self.assertEqual(_safe_stem("clip — part 2/5.wav"), "clip — part 2-5")
        self.assertEqual(_safe_stem("plain.mov"), "plain")
        self.assertEqual(_safe_stem(""), "media")
        self.assertEqual(_safe_stem(None), "media")
        self.assertEqual(_safe_stem("///"), "---")

    def test_split_parts_are_named_part_n_of_m_with_a_clean_stem(self):
        item = self.audio_item()
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE uploads SET original_filename = ? WHERE id = ?", ("take 1/2\\x.wav", item.id)
            )

        run = self.finish([{"tool": "splittwo"}], [{"item_id": item.id}])

        names = [entry.original_filename for entry in run.items]
        self.assertEqual(names, ["take 1-2-x — part 1 of 2.wav", "take 1-2-x — part 2 of 2.wav"])
        self.assertTrue(all("/" not in name and "\\" not in name for name in names))

    def test_names_derived_from_a_slashed_source_never_lose_their_stem(self):
        item = self.image_item()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE uploads SET original_filename = ? WHERE id = ?", ("holiday 3/4.png", item.id))

        run = self.finish([{"tool": "shrink", "params": {"size": 9}}], [{"item_id": item.id}])

        self.assertEqual(run.items[0].original_filename, "holiday 3-4-shrink.png")

    def test_a_corrupt_stored_image_fails_with_a_message_that_leaks_no_path(self):
        path = self.uploads_dir / "broken.png"
        path.write_bytes(b"this is not a png")
        item = self.record(path, "image")

        run = self.finish([{"tool": "shrink", "params": {"size": 9}}], [{"item_id": item.id}])

        self.assertEqual(run.status, "failed")
        message = run.error.message
        for private in (str(self.temp_dir), str(self.storage_dir), "broken.png", "/tmp", "identify"):
            self.assertNotIn(private, message)
        self.assertTrue(message)
        self.assertEqual(self.scratch_entries(), [])

    def test_a_dimension_over_the_declared_max_is_refused_before_a_run_starts(self):
        item = self.image_item()

        for tool, params in (
            ("fit_canvas", {"width": 9000, "height": 100}),
            ("fit_canvas", {"width": 100, "height": 9000}),
            ("resize", {"width": 9000}),
            ("resize", {"height": 9000}),
        ):
            response = self.start([{"tool": tool, "params": params}], [{"item_id": item.id}])
            self.assertEqual(response.status_code, 400, (tool, params))
            self.assertEqual(self.error_of(response)["error"], "invalid_params")
            self.assertIn(self.error_of(response)["field"], ("width", "height"))
            self.assertEqual(self.error_of(response)["step_index"], 0)
        self.assertEqual(self.jobs.counts(), {"queued": 0, "running": 0, "finished": 0})

    def test_the_catalog_exposes_the_max_of_the_dimension_params(self):
        tools = self.client.get("/api/media-tools").json()["data"]["tools"]
        by_id = {tool["id"]: tool for tool in tools}

        for tool_id in ("fit_canvas", "resize"):
            limits = {p["name"]: p.get("max") for p in by_id[tool_id]["params"] if p["name"] in ("width", "height")}
            self.assertEqual(limits, {"width": 8192, "height": 8192})


if __name__ == "__main__":
    unittest.main()
