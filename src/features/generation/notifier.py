"""Generation completion/failure notifications.

Two distinct paths, both best-effort so a notification failure never breaks
generation handling: failures fan out through the *global* notification manager
(toast + persistent bell entry), while successful completions go through the
*injected* notification manager. Grouped here so the orchestrator carries no
notification wiring.
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple

from src.features.generation.failure import GenerationFailure
from src.features.generation.grids.repository import GridRepository
from src.features.generation.status_tracker import GenerationState

logger = logging.getLogger(__name__)

ANNOUNCED_GRIDS_CAP = 256


class GenerationNotifier:
    """Emits generation completed/failed notifications for the owning user."""

    def __init__(
        self,
        notification_manager: Optional[Callable[..., Any]] = None,
        grids: Optional[GridRepository] = None,
    ):
        """Initialize the notifier.

        Args:
            notification_manager: Optional bound notify callable
                (`functools.partial(operations.notify, collaborators)`) used
                for completion notifications. Failure notifications use the
                global one instead.
        """
        self.notification_manager = notification_manager
        self._grids = grids
        self._announced_grids: "OrderedDict[Tuple[str, Tuple[str, ...]], None]" = OrderedDict()
        self._grid_lock = threading.Lock()

    def notify_failure(
        self,
        generation_id: str,
        user_id: Optional[str],
        failure: GenerationFailure,
        include_detail: bool = False,
        video: bool = False,
        grid_id: Optional[str] = None,
        preset_id: Optional[str] = None,
    ) -> None:
        if grid_id:
            if user_id:
                self._notify_grid_completion(grid_id, preset_id, user_id)
            return
        try:
            from src.platform.plugins.runtime_registries import get_global_notification_manager

            metadata = {
                "generation_id": generation_id,
                **failure.viewer_payload(generation_id, include_detail, video),
                "contact_admin": not include_detail,
            }
            if include_detail:
                metadata.update(failure.admin_payload())

            get_global_notification_manager()(
                level="error",
                title="Generation failed",
                message=failure.viewer_message(include_detail),
                category="generation",
                type="generation.failed",
                user_id=user_id,
                metadata=metadata,
                show_toast=True,
            )
        except Exception as e:
            logger.error(f"Failed to raise generation-failure notification for {generation_id}: {e}")

    def notify_completion(
        self,
        generation_id: str,
        record,
        generation,
        duration: float,
    ) -> None:
        """
        Notify the owning user of a successful completion (best-effort - a
        notification failure must never break generation completion).

        FAILED is intentionally not handled here: `_handle_generation_output`
        already calls `notify_failure` the moment the `ErrorGenerationOutput`
        arrives, before the backend's completion sentinel reaches this method
        (see `in_process_backend.py`'s `finally: emit(None)`, which runs
        unconditionally on both success and failure). Raising here too would
        fire a second "Generation failed" notification for the same event.
        CANCELLED is user-initiated and intentionally not notified either.
        """
        if not (self.notification_manager and generation and generation.user_id):
            return
        if getattr(record, 'grid_id', None):
            if record.state in (GenerationState.COMPLETED, GenerationState.FAILED):
                self._notify_grid_completion(record.grid_id, record.preset_id, generation.user_id)
            return
        if record.state != GenerationState.COMPLETED:
            return
        try:
            self.notification_manager(
                level='success',
                title='Generation completed',
                category='generation',
                user_id=generation.user_id,
                type='generation.completed',
                metadata={
                    'generation_id': generation_id,
                    'preset_id': record.preset_id,
                    'duration': duration,
                }
            )
        except Exception as e:
            logger.error(f"Failed to send generation notification for {generation_id}: {e}")

    def _notify_grid_completion(self, grid_id: str, preset_id: Optional[str], user_id: str) -> None:
        grids = self._grids
        if grids is None or self.notification_manager is None:
            return
        try:
            grid = grids.get(grid_id)
            if grid is None:
                return
            latest = {}
            for row in grids.cells(grid_id):
                if row.grid_x is None or row.grid_y is None:
                    continue
                key = (row.grid_x, row.grid_y)
                if key not in latest or row.id > latest[key].id:
                    latest[key] = row
            if len(latest) < grid.cell_count or any(row.is_active() for row in latest.values()):
                return
            done = sum(1 for row in latest.values() if row.status == 'completed')
            failed = sum(1 for row in latest.values() if row.status == 'failed')
            if done == 0 and failed == 0:
                return
            run_key = (grid_id, tuple(sorted(row.id for row in latest.values())))
            with self._grid_lock:
                if run_key in self._announced_grids:
                    return
                self._announced_grids[run_key] = None
                while len(self._announced_grids) > ANNOUNCED_GRIDS_CAP:
                    self._announced_grids.popitem(last=False)
            total = len(latest)
            message = f"{done} of {total} cells done, {failed} failed." if failed else f"{done} of {total} cells done."
            self.notification_manager(
                level='error' if done == 0 else 'warning' if failed else 'success',
                title='Comparison failed' if done == 0 else 'Comparison finished',
                message=message,
                category='generation',
                user_id=user_id,
                type='generation.failed' if done == 0 else 'generation.completed',
                metadata={
                    'grid_id': grid_id,
                    'preset_id': preset_id,
                    'cells': total,
                    'completed': done,
                    'failed': failed,
                },
            )
        except Exception as e:
            logger.error(f"Failed to send grid notification for {grid_id}: {e}")
