"""Tests for GenerationNotifier.

Regression coverage for the doubled-notification bug: a failed
generation must raise exactly one "Generation failed" notification, not two.
`_handle_generation_output` calls `notify_failure` when the
`ErrorGenerationOutput` arrives; the backend's `finally: emit(None)` then
always delivers a completion sentinel afterward (on both success and
failure), which used to trigger a second, redundant notification out of
`notify_completion`'s FAILED branch.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.features.generation.records import Generation

from src.features.generation.failure import GenerationFailure
from src.features.generation.notifier import GenerationNotifier
from src.features.generation.status_tracker import GenerationState


def _record(state: GenerationState, error: str = None, preset_id: str = "preset-1", grid_id: str = None):
    record = Mock()
    record.state = state
    record.error = error
    record.preset_id = preset_id
    record.grid_id = grid_id
    return record


def _generation(user_id: str = "user-1"):
    generation = Mock()
    generation.user_id = user_id
    return generation


class TestNotifyCompletion:
    def test_completed_state_raises_one_notification(self):
        manager = Mock()
        notifier = GenerationNotifier(notification_manager=manager)

        notifier.notify_completion(
            "gen-1", _record(GenerationState.COMPLETED), _generation(), duration=1.0
        )

        manager.assert_called_once()
        kwargs = manager.call_args.kwargs
        assert kwargs["title"] == "Generation completed"
        assert kwargs["type"] == "generation.completed"

    def test_failed_state_does_not_notify(self):
        """The FAILED branch was removed: notify_failure already covers it."""
        manager = Mock()
        notifier = GenerationNotifier(notification_manager=manager)

        notifier.notify_completion(
            "gen-1", _record(GenerationState.FAILED, error="boom"), _generation(), duration=1.0
        )

        manager.assert_not_called()

    def test_cancelled_state_does_not_notify(self):
        manager = Mock()
        notifier = GenerationNotifier(notification_manager=manager)

        notifier.notify_completion(
            "gen-1", _record(GenerationState.CANCELLED), _generation(), duration=1.0
        )

        manager.assert_not_called()

    def test_no_notification_manager_is_a_noop(self):
        notifier = GenerationNotifier(notification_manager=None)
        # Should not raise even though there's nothing to call notify() on.
        notifier.notify_completion(
            "gen-1", _record(GenerationState.COMPLETED), _generation(), duration=1.0
        )

    def test_no_user_id_is_a_noop(self):
        manager = Mock()
        notifier = GenerationNotifier(notification_manager=manager)

        notifier.notify_completion(
            "gen-1", _record(GenerationState.COMPLETED), _generation(user_id=None), duration=1.0
        )

        manager.assert_not_called()

    def test_notify_exception_is_swallowed(self):
        manager = Mock()
        manager.side_effect = RuntimeError("ws down")
        notifier = GenerationNotifier(notification_manager=manager)

        # Must not raise: a notification failure can't break generation completion.
        notifier.notify_completion(
            "gen-1", _record(GenerationState.COMPLETED), _generation(), duration=1.0
        )


def _failure():
    return GenerationFailure(
        error_code="disk_full",
        message="The disk is full.",
        hints=("Free up space",),
        raw_error="OSError: [Errno 28] No space left on device: '/srv/outputs/private/x.png'",
        detail="Traceback (most recent call last):\n  File \"/srv/app/src/handler.py\", line 9",
        failed_pipe_id="saver",
        failed_pipe_name="image_saver",
        failed_at_step="Saving 1/1",
    )


class TestNotifyFailure:
    def test_regular_owner_gets_the_safe_message_without_detail(self):
        global_manager = Mock()
        notifier = GenerationNotifier(notification_manager=Mock())

        with patch(
            "src.platform.plugins.runtime_registries.get_global_notification_manager",
            return_value=global_manager,
        ):
            notifier.notify_failure("gen-1", "user-1", _failure())

        global_manager.assert_called_once()
        kwargs = global_manager.call_args.kwargs
        assert kwargs["title"] == "Generation failed"
        assert kwargs["type"] == "generation.failed"
        assert kwargs["user_id"] == "user-1"
        assert kwargs["message"] == "The server has run out of storage space."
        metadata = kwargs["metadata"]
        assert metadata["generation_id"] == "gen-1"
        assert metadata["error_id"] == "gen-1"
        assert metadata["error_code"] == "disk_full"
        assert metadata["hint"] == "- Send the error ID to your administrator"
        assert "detail" not in metadata
        assert "failed_pipe_id" not in metadata
        rendered = str(kwargs)
        assert "/srv/" not in rendered
        assert "Traceback" not in rendered
        assert "Errno" not in rendered

    def test_admin_owner_gets_the_detail(self):
        global_manager = Mock()
        notifier = GenerationNotifier(notification_manager=Mock())

        with patch(
            "src.platform.plugins.runtime_registries.get_global_notification_manager",
            return_value=global_manager,
        ):
            notifier.notify_failure("gen-1", "admin-1", _failure(), include_detail=True)

        metadata = global_manager.call_args.kwargs["metadata"]
        assert "No space left on device: '/srv/outputs/private/x.png'" in metadata["detail"]
        assert "Traceback" in metadata["detail"]
        assert metadata["failed_pipe_id"] == "saver"
        assert metadata["failed_pipe_name"] == "image_saver"
        assert metadata["failed_at_step"] == "Saving 1/1"

    def test_exception_is_swallowed(self):
        notifier = GenerationNotifier(notification_manager=Mock())

        with patch(
            "src.platform.plugins.runtime_registries.get_global_notification_manager",
            side_effect=RuntimeError("not ready"),
        ):
            notifier.notify_failure("gen-1", "user-1", _failure())

    def test_only_one_notification_for_a_failed_generation_end_to_end(self):
        shared_manager = Mock()
        notifier = GenerationNotifier(notification_manager=shared_manager)

        with patch(
            "src.platform.plugins.runtime_registries.get_global_notification_manager",
            return_value=shared_manager,
        ):
            notifier.notify_failure("gen-1", "user-1", _failure())

        notifier.notify_completion(
            "gen-1", _record(GenerationState.FAILED, error="boom"), _generation(), duration=1.0
        )

        assert shared_manager.call_count == 1


def _cell(gen_id: str, x: int, status: str) -> Generation:
    return Generation(id=gen_id, user_id="user-1", preset_id="preset-1", form_data={}, status=status, grid_id="grid-1", grid_x=x, grid_y=0)


class _Grids:
    def __init__(self, cell_count: int, cells):
        self.cell_count = cell_count
        self.rows = list(cells)

    def get(self, grid_id):
        return SimpleNamespace(id=grid_id, cell_count=self.cell_count)

    def cells(self, grid_id):
        return list(self.rows)


class TestGridCompletion:
    def _finish(self, notifier, gen_id, state=GenerationState.COMPLETED):
        notifier.notify_completion(gen_id, _record(state, grid_id="grid-1"), _generation(), duration=1.0)

    def test_grid_raises_one_notification_after_the_last_cell(self):
        manager = Mock()
        grids = _Grids(3, [_cell("a", 0, "completed"), _cell("b", 1, "running"), _cell("c", 2, "pending")])
        notifier = GenerationNotifier(notification_manager=manager, grids=grids)

        self._finish(notifier, "a")
        grids.rows[1] = _cell("b", 1, "completed")
        self._finish(notifier, "b")
        manager.assert_not_called()

        grids.rows[2] = _cell("c", 2, "completed")
        self._finish(notifier, "c")
        self._finish(notifier, "c")

        manager.assert_called_once()
        kwargs = manager.call_args.kwargs
        assert kwargs["type"] == "generation.completed"
        assert kwargs["level"] == "success"
        assert kwargs["message"] == "3 of 3 cells done."
        assert kwargs["metadata"]["grid_id"] == "grid-1"
        assert "generation_id" not in kwargs["metadata"]

    def test_failed_last_cell_still_closes_the_grid(self):
        manager = Mock()
        grids = _Grids(2, [_cell("a", 0, "completed"), _cell("b", 1, "failed")])
        notifier = GenerationNotifier(notification_manager=manager, grids=grids)

        self._finish(notifier, "b", GenerationState.FAILED)

        kwargs = manager.call_args.kwargs
        assert kwargs["level"] == "warning"
        assert kwargs["message"] == "1 of 2 cells done, 1 failed."
        assert kwargs["metadata"]["failed"] == 1

    def test_unsubmitted_cells_keep_the_grid_open(self):
        manager = Mock()
        notifier = GenerationNotifier(notification_manager=manager, grids=_Grids(3, [_cell("a", 0, "completed")]))

        self._finish(notifier, "a")

        manager.assert_not_called()

    def test_retried_cell_uses_its_latest_row_and_announces_again(self):
        manager = Mock()
        grids = _Grids(2, [_cell("01A", 0, "completed"), _cell("01B", 1, "failed")])
        notifier = GenerationNotifier(notification_manager=manager, grids=grids)
        self._finish(notifier, "01B", GenerationState.FAILED)

        grids.rows.append(_cell("01C", 1, "running"))
        self._finish(notifier, "01A")
        assert manager.call_count == 1

        grids.rows[2] = _cell("01C", 1, "completed")
        self._finish(notifier, "01C")

        assert manager.call_count == 2
        assert manager.call_args.kwargs["message"] == "2 of 2 cells done."

    def test_all_failed_grid_reports_the_failure_once(self):
        manager = Mock()
        cells = [_cell(f"0{x}", x, "failed") for x in range(4)]
        notifier = GenerationNotifier(notification_manager=manager, grids=_Grids(4, cells))

        self._finish(notifier, "03", GenerationState.FAILED)
        self._finish(notifier, "03", GenerationState.FAILED)

        manager.assert_called_once()
        kwargs = manager.call_args.kwargs
        assert kwargs["level"] == "error"
        assert kwargs["type"] == "generation.failed"
        assert kwargs["message"] == "0 of 4 cells done, 4 failed."

    def test_all_cancelled_grid_stays_silent(self):
        manager = Mock()
        notifier = GenerationNotifier(notification_manager=manager, grids=_Grids(1, [_cell("a", 0, "cancelled")]))

        self._finish(notifier, "a")

        manager.assert_not_called()

    def test_grid_cell_failure_sends_no_per_cell_notification_and_closes_the_grid(self):
        manager = Mock()
        global_manager = Mock()
        grids = _Grids(2, [_cell("a", 0, "completed"), _cell("b", 1, "failed")])
        notifier = GenerationNotifier(notification_manager=manager, grids=grids)

        with patch(
            "src.platform.plugins.runtime_registries.get_global_notification_manager",
            return_value=global_manager,
        ):
            notifier.notify_failure("b", "user-1", _failure(), grid_id="grid-1", preset_id="preset-1")

        global_manager.assert_not_called()
        manager.assert_called_once()
        assert manager.call_args.kwargs["message"] == "1 of 2 cells done, 1 failed."

    def test_announced_runs_are_bounded(self):
        from src.features.generation.notifier import ANNOUNCED_GRIDS_CAP
        manager = Mock()
        grids = _Grids(1, [])
        notifier = GenerationNotifier(notification_manager=manager, grids=grids)

        for n in range(ANNOUNCED_GRIDS_CAP + 5):
            grids.rows = [_cell(f"{n:04d}", 0, "completed")]
            self._finish(notifier, f"{n:04d}")

        assert manager.call_count == ANNOUNCED_GRIDS_CAP + 5
        assert len(notifier._announced_grids) == ANNOUNCED_GRIDS_CAP
        assert ("grid-1", ("0000",)) not in notifier._announced_grids
