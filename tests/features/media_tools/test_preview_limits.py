import io
import threading
import time
import unittest

from PIL import Image

from src.features.media_tools import service as service_module
from src.features.media_tools.service import _scale_pixel_params
from src.platform.http.errors import InvalidInputError
from src.platform.media_tools import MediaTool, ToolSpec
from tests.features.media_tools.support import MediaToolsTestBase, needs_ffmpeg


class SlowPad(MediaTool):
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def image(self, image, params):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.15)
        with self.lock:
            self.active -= 1
        return image


class TestPreviewLimits(MediaToolsTestBase):

    def preview(self, tool, params, source, **extra):
        return self.client.post(
            "/api/media-tools/preview", json={"tool": tool, "params": params, "input": source, **extra}
        )

    def pad_calls(self):
        return self.registry.get("padpx").impl.seen

    def decode(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        return Image.open(io.BytesIO(response.content))

    def test_a_huge_pixel_param_is_scaled_down_before_the_tool_runs(self):
        item = self.image_item(size=(300, 200))

        image = self.decode(self.preview("padpx", {"width": 8192, "height": 8192}, {"item_id": item.id}))

        self.assertEqual([(w, h) for w, h, _ in self.pad_calls()], [(512, 512)])
        self.assertLessEqual(max(image.size), 512)

    def test_pixel_params_keep_their_proportion_when_scaled(self):
        item = self.image_item()

        self.decode(self.preview("padpx", {"width": 8192, "height": 4096}, {"item_id": item.id}))

        self.assertEqual([(w, h) for w, h, _ in self.pad_calls()], [(512, 256)])

    def test_pixel_params_within_the_preview_side_are_untouched(self):
        item = self.image_item()

        self.decode(self.preview("padpx", {"width": 100, "height": 50}, {"item_id": item.id}))

        self.assertEqual([(w, h) for w, h, _ in self.pad_calls()], [(100, 50)])

    def test_a_scaled_pixel_param_never_drops_below_its_minimum(self):
        item = self.image_item()

        self.decode(self.preview("padpx", {"width": 8192, "height": 2}, {"item_id": item.id}))

        self.assertEqual([(w, h) for w, h, _ in self.pad_calls()], [(512, 1)])

    def test_a_value_over_the_declared_maximum_is_refused_before_any_work(self):
        item = self.image_item()

        response = self.preview("padpx", {"width": 8193, "height": 10}, {"item_id": item.id})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.error_of(response)["error"], "invalid_params")
        self.assertEqual(self.error_of(response)["field"], "width")
        self.assertEqual(self.pad_calls(), [])

    def test_the_scale_helper_only_touches_numeric_pixel_params(self):
        spec = self.registry.get("padpx")

        scaled = _scale_pixel_params(spec, {"width": 4096, "height": 1024, "other": 9999}, 512)

        self.assertEqual(scaled, {"width": 512, "height": 128, "other": 9999})
        self.assertEqual(_scale_pixel_params(spec, {"width": None, "height": 10}, 512), {"width": None, "height": 10})
        self.assertEqual(_scale_pixel_params(spec, {"width": True, "height": 10}, 512), {"width": True, "height": 10})

    def test_the_source_is_decoded_at_preview_size_not_full_size(self):
        item = self.image_item(name="wide.png", size=(3000, 2000))

        self.decode(self.preview("padpx", {"width": 64, "height": 64}, {"item_id": item.id}))

        _, _, source_size = self.pad_calls()[0]
        self.assertLessEqual(max(source_size), service_module.PREVIEW_SOURCE_SIDE)

    def test_previews_beyond_the_slot_count_wait_and_then_get_a_busy_answer(self):
        item = self.image_item()
        self.service._preview_wait = 0.05
        held = [self.service._preview_slots.acquire() for _ in range(service_module.PREVIEW_CONCURRENCY)]
        self.assertTrue(all(held))

        response = self.preview("padpx", {"width": 8, "height": 8}, {"item_id": item.id})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.error_of(response)["error"], "preview_busy")
        for _ in held:
            self.service._preview_slots.release()
        self.decode(self.preview("padpx", {"width": 8, "height": 8}, {"item_id": item.id}))

    def test_a_failed_preview_releases_its_slot(self):
        item = self.image_item()
        for _ in range(service_module.PREVIEW_CONCURRENCY + 2):
            response = self.preview("padpx", {"width": 8193, "height": 8}, {"item_id": item.id})
            self.assertEqual(response.status_code, 400)
        self.service._preview_wait = 0.2
        for _ in range(service_module.PREVIEW_CONCURRENCY + 2):
            response = self.preview("rot", {}, {"generation_file_id": "missing"})
            self.assertEqual(response.status_code, 404)
        self.decode(self.preview("padpx", {"width": 8, "height": 8}, {"item_id": item.id}))

    def test_concurrent_previews_never_exceed_the_slot_count(self):
        item = self.image_item()
        slow = SlowPad()
        self.registry.register(ToolSpec(id="slowpad", label="Slow", accepts=("image",), preview=True, impl=slow))
        self.service._preview_wait = 20
        results = []

        def call():
            results.append(self.preview("slowpad", {}, {"item_id": item.id}).status_code)

        threads = [threading.Thread(target=call) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(results, [200] * 6)
        self.assertLessEqual(slow.peak, service_module.PREVIEW_CONCURRENCY)
        self.assertGreaterEqual(slow.peak, 1)


@needs_ffmpeg
class TestPreviewTime(MediaToolsTestBase):

    def preview(self, source, **extra):
        return self.client.post(
            "/api/media-tools/preview", json={"tool": "rot", "params": {}, "input": source, **extra}
        )

    def test_a_non_finite_time_is_refused_as_invalid_params_on_time_seconds(self):
        item = self.video_item(seconds=1.0)

        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidInputError) as raised:
                self.service.preview(self.user_id, "rot", {}, {"item_id": item.id}, time_seconds=bad)
            self.assertEqual(raised.exception.code, "invalid_params")
            self.assertEqual(raised.exception.extra["field"], "time_seconds")

    def test_a_negative_time_is_a_400_on_time_seconds(self):
        item = self.video_item(seconds=1.0)

        response = self.preview({"item_id": item.id}, time_seconds=-1)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.error_of(response)["error"], "invalid_params")
        self.assertEqual(self.error_of(response)["field"], "time_seconds")

    def test_a_time_past_the_end_is_a_400_on_time_seconds(self):
        item = self.video_item(seconds=1.0)

        response = self.preview({"item_id": item.id}, time_seconds=5)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.error_of(response)["error"], "invalid_params")
        self.assertEqual(self.error_of(response)["field"], "time_seconds")
        self.assertEqual(self.scratch_entries(), [])

    def test_a_time_at_the_very_end_still_returns_the_last_frame(self):
        item = self.video_item(seconds=1.0)

        response = self.preview({"item_id": item.id}, time_seconds=1.0)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(Image.open(io.BytesIO(response.content)).size, (48, 64))

    def test_a_time_inside_the_clip_still_works_and_a_missing_time_means_the_start(self):
        item = self.video_item(seconds=1.0)

        self.assertEqual(self.preview({"item_id": item.id}, time_seconds=0.5).status_code, 200)
        self.assertEqual(self.preview({"item_id": item.id}).status_code, 200)

    def test_an_image_preview_ignores_the_time(self):
        item = self.image_item()

        self.assertEqual(self.preview({"item_id": item.id}, time_seconds=999).status_code, 200)


if __name__ == "__main__":
    unittest.main()
