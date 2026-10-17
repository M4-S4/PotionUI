"""Plan limits: contributing a limit kind, recording its events, and reading
where a user stands against one (`limit_status`, for a plugin that admits its
own operations against a kind it measures itself).

`LimitEvents.admit(user_id, kind, units, ref_id)` checks the user's limit for
`kind` and records the units in one step, so two concurrent admissions cannot
both pass on the same remaining room; it raises `LimitExceeded` (with `limit`,
`used` and `remaining`) and records nothing when the units do not fit.
`LimitEvents.refund(ref_id, kind=None)` refunds the events recorded with
`ref_id`, only those of `kind` when it is given.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.features.plans.errors import LimitExceeded
from src.platform.database.rows import now_utc
from src.platform.plugins.limit_kinds import (
    AdmissionRequest,
    LimitEventsView,
    LimitKind,
    MeasureContext,
    Refusal,
    RefusalContext,
    Usage,
)
from src.platform.plugins.runtime_registries import get_container

__all__ = [
    "AdmissionRequest",
    "LimitEvents",
    "LimitEventsView",
    "LimitExceeded",
    "LimitKind",
    "LimitStatus",
    "MeasureContext",
    "Refusal",
    "RefusalContext",
    "Usage",
    "limit_status",
]


class LimitEvents:

    @staticmethod
    def record(user_id: str, kind: str, units: float = 1, ref_id: Optional[str] = None) -> str:
        return get_container().plans.events.record(user_id, kind, units, ref_id, now_utc())

    @staticmethod
    def admit(user_id: str, kind: str, units: float = 1, ref_id: Optional[str] = None) -> str:
        """Record `units` of `kind` for the user if they fit the user's limit; returns the event id."""
        return get_container().plans.guard.admit_units(user_id, kind, units, ref_id)

    @staticmethod
    def refund(ref_id: str, kind: Optional[str] = None) -> int:
        """Refund the events recorded with `ref_id` (only those of `kind` when given); returns how many."""
        return get_container().plans.guard.refund(ref_id, kind)


@dataclass(frozen=True)
class LimitStatus:
    """Where a user stands against one limit kind. `limit` is None when nothing
    limits the user for `kind`; `exempt` is True for an admin the plan settings
    exempt (`limit` is still reported, `used` is not measured)."""

    kind: str
    limit: Optional[float] = None
    used: Optional[float] = None
    exempt: bool = False
    resets_at: Optional[datetime] = None
    contact_line: str = ""

    @property
    def limited(self) -> bool:
        return self.limit is not None and not self.exempt

    @property
    def remaining(self) -> Optional[float]:
        """What the user may still use in this window; None when unlimited or exempt."""
        if not self.limited:
            return None
        return max(0.0, float(self.limit) - float(self.used or 0.0))


def limit_status(user_id: str, kind: str) -> LimitStatus:
    """Read the limit `kind` as it applies to `user_id` right now (see `LimitStatus`)."""
    container = get_container()
    guard = container.plans.guard
    resolution = guard.resolution(user_id)
    if resolution is None:
        return LimitStatus(kind=kind)
    limit = resolution.value(kind)
    if limit is None:
        return LimitStatus(kind=kind)
    settings = guard.settings.read()
    contact_line = str(getattr(settings, "contact_line", "") or "")
    if guard.is_exempt(resolution.subject, settings):
        return LimitStatus(kind=kind, limit=limit, exempt=True, contact_line=contact_line)
    limit_kind = container.limit_kind_registry.get(kind)
    if limit_kind is None:
        return LimitStatus(kind=kind)
    measured = guard.measure(limit_kind, user_id, settings.day_timezone)
    return LimitStatus(
        kind=kind,
        limit=limit,
        used=measured.used,
        exempt=False,
        resets_at=measured.resets_at,
        contact_line=contact_line,
    )
