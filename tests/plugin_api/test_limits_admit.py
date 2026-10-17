import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.features.plans.components import build_plans
from src.features.user_groups.constants import ALL_USERS_GROUP_ID
from src.platform.plugins.limit_kinds import LimitKind, LimitKindRegistry, Usage
from src.platform.settings.repository import SettingRepository
from src.platform.settings.settings import Settings
from src.plugin_api.limits import LimitEvents, LimitExceeded
from tests.features.plans.conftest import Clock, Seeder, make_plan

KIND = "producer.gpu_minutes"
OTHER = "producer.exports"


def slow_ledger(context):
    time.sleep(0.05)
    return Usage(used=context.events.total())


@pytest.fixture
def plans(mock_db):
    kinds = LimitKindRegistry()
    for key, measure in ((KIND, slow_ledger), (OTHER, None)):
        kinds.register(LimitKind(
            key=key, label=key, value_type="count", window="day", ledger=True,
            measure=measure, applies=lambda request: False, source="producer",
        ))
    clock = Clock(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc))
    return build_plans(kinds, Settings(SettingRepository()), clock=clock)


@pytest.fixture
def seed(mock_db):
    return Seeder(mock_db)


@pytest.fixture
def container(plans):
    with patch("src.plugin_api.limits.get_container", return_value=SimpleNamespace(plans=plans)):
        yield plans


def limit(plans, **limits):
    plan = make_plan(plans, "Producer", **limits)
    plans.guard.plans.set_group_plan(ALL_USERS_GROUP_ID, plan.id)


def ledger(plans, user_id="u1", kind=KIND):
    return plans.guard.events.total(user_id, kind, None, None)


def test_units_that_fit_are_recorded_and_return_the_event_id(seed, container):
    seed.user("u1")
    limit(container, **{KIND: 5})

    event_id = LimitEvents.admit("u1", KIND, 3, ref_id="take-1")

    assert isinstance(event_id, str) and event_id
    assert ledger(container) == 3


def test_units_that_do_not_fit_are_refused_and_not_recorded(seed, container):
    seed.user("u1")
    limit(container, **{KIND: 5})
    LimitEvents.admit("u1", KIND, 4, ref_id="take-1")

    with pytest.raises(LimitExceeded) as refused:
        LimitEvents.admit("u1", KIND, 2, ref_id="take-2")

    assert (refused.value.limit, refused.value.used, refused.value.remaining) == (5, 4, 1)
    assert refused.value.payload()["needed"] == 2
    assert refused.value.kinds == [KIND]
    assert ledger(container) == 4


def test_two_concurrent_admissions_cannot_share_the_same_room(seed, container):
    seed.user("u1")
    limit(container, **{KIND: 5})
    barrier = threading.Barrier(2)
    outcomes = []

    def admit(ref_id):
        barrier.wait()
        try:
            outcomes.append(LimitEvents.admit("u1", KIND, 3, ref_id=ref_id))
        except LimitExceeded:
            outcomes.append("refused")

    threads = [threading.Thread(target=admit, args=(f"take-{n}",)) for n in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(outcome == "refused" for outcome in outcomes) == [False, True]
    assert ledger(container) == 3


def test_users_nothing_limits_are_recorded_without_a_check(seed, container):
    seed.user("u1")
    seed.user("admin", admin=True)
    limit(container, **{KIND: 1})

    LimitEvents.admit("admin", KIND, 10)
    LimitEvents.admit("u1", OTHER, 10)
    LimitEvents.admit("u1", "producer.unregistered", 10)

    assert ledger(container, "admin") == 10
    assert ledger(container, "u1", OTHER) == 10
    assert ledger(container, "u1", "producer.unregistered") == 10


def test_refund_is_limited_to_a_kind_when_one_is_given(seed, container):
    seed.user("u1")
    LimitEvents.admit("u1", KIND, 3, ref_id="take-1")
    LimitEvents.admit("u1", OTHER, 1, ref_id="take-1")

    assert LimitEvents.refund("take-1", kind=OTHER) == 1
    assert (ledger(container), ledger(container, kind=OTHER)) == (3, 0)
    assert LimitEvents.refund("take-1") == 1
    assert ledger(container) == 0
