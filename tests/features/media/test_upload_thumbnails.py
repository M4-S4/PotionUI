import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from src.features.generation.thumbnail_profile import DEFAULT_PROFILE, profile_hash
from src.features.media import upload_thumbnails as module


def record(small=None, medium=None, large=None):
    return SimpleNamespace(thumbnail_small=small, thumbnail_medium=medium, thumbnail_large=large)


class TestThumbnailKeys(unittest.TestCase):

    def test_image_rows_list_only_their_own_thumbnail_files(self):
        keys = module.thumbnail_keys(record(small="thumbnails/a_small.webp", large="thumbnails/a_large.webp"), False)

        self.assertEqual(keys, {"uploads/thumbnails/a_small.webp", "uploads/thumbnails/a_large.webp"})

    def test_video_rows_also_list_the_animated_sibling_of_each_size(self):
        keys = module.thumbnail_keys(record(medium="thumbnails/v_medium.webp"), True)

        self.assertEqual(
            keys, {"uploads/thumbnails/v_medium.webp", "uploads/thumbnails/v_medium_animated.webp"}
        )

    def test_a_row_without_thumbnails_has_no_keys(self):
        self.assertEqual(module.thumbnail_keys(record(), True), set())

    def test_paths_to_keys_mirrors_the_row_form(self):
        paths = {"small": "thumbnails/n_small.webp"}

        self.assertEqual(module.paths_to_keys(paths, False), {"uploads/thumbnails/n_small.webp"})
        self.assertEqual(
            module.paths_to_keys(paths, True),
            {"uploads/thumbnails/n_small.webp", "uploads/thumbnails/n_small_animated.webp"},
        )

    def test_delete_keys_removes_each_key_and_survives_a_failing_delete(self):
        driver = Mock()
        driver.delete.side_effect = [RuntimeError("gone"), None]

        module.delete_keys(driver, {"uploads/a", "uploads/b"})

        self.assertEqual(driver.delete.call_count, 2)


class TestRendering(unittest.TestCase):

    def test_image_thumbnails_go_through_the_shared_generator_under_the_uploads_base(self):
        image = Image.new("RGB", (4, 4))
        driver = Mock()

        with patch.object(module, "generate_thumbnails", return_value={"small": "thumbnails/s_small.webp"}) as gen:
            result = module.render_image_thumbnails(driver, DEFAULT_PROFILE, image, "stem")

        gen.assert_called_once_with(image, driver, "uploads", "stem", DEFAULT_PROFILE)
        self.assertEqual(result, {"small": "thumbnails/s_small.webp"})

    def test_video_thumbnails_are_stored_on_the_row_with_the_profile_hash(self):
        driver = Mock()
        driver.local_path.return_value = "/clip.mp4"
        repo = Mock()
        paths = {"small": "thumbnails/s_small.webp", "medium": "thumbnails/s_medium.webp"}

        with patch.object(module, "generate_video_thumbnails", return_value=paths):
            module.render_video_thumbnails(driver, repo, DEFAULT_PROFILE, "uploads/s.mp4", ".mp4", "row-1", "s")

        repo.set_thumbnail_paths.assert_called_once_with(
            "row-1", "thumbnails/s_small.webp", "thumbnails/s_medium.webp", None, profile_hash(DEFAULT_PROFILE)
        )

    def test_a_video_that_yields_no_thumbnails_leaves_the_row_alone(self):
        driver = Mock()
        driver.local_path.return_value = "/clip.mp4"
        repo = Mock()

        with patch.object(module, "generate_video_thumbnails", return_value={}):
            module.render_video_thumbnails(driver, repo, DEFAULT_PROFILE, "uploads/s.mp4", ".mp4", "row-1", "s")

        repo.set_thumbnail_paths.assert_not_called()

    def test_a_failing_video_render_is_swallowed(self):
        driver = Mock()
        driver.local_path.side_effect = RuntimeError("boom")
        repo = Mock()

        module.render_video_thumbnails(driver, repo, DEFAULT_PROFILE, "uploads/s.mp4", ".mp4", "row-1", "s")

        repo.set_thumbnail_paths.assert_not_called()

    def test_run_in_background_runs_the_worker_on_another_thread(self):
        done = threading.Event()
        seen = []

        def worker():
            seen.append(threading.current_thread() is not threading.main_thread())
            done.set()

        module.run_in_background(worker)

        self.assertTrue(done.wait(5))
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()
