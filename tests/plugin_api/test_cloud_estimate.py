from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.features.cloud.director import DirectorEstimateError, estimate_shots
from src.features.cloud.dto import DirectorEstimateRequest
from src.features.cloud.routes import CloudCapabilitiesController
from src.features.models.exceptions import ModelNotFoundException
from src.platform.plugins import runtime_registries
from src.platform.security.user import AccountType, User
from src.plugin_api import cloud as api
from src.plugin_api.cloud_testing import fake_director_specs

FULL = fake_director_specs()[0]
FILM = [
    {"task": "txt2video", "params": {"duration_s": 2}},
    {"task": "img2video", "params": {"duration_s": 4}},
]


def _user(account_type):
    return User(id="u1", username="u1", email="u1@example.test", password_hash="h", account_type=account_type)


class FakeCapabilities:
    def __init__(self):
        self.calls = []

    def estimate(self, model_id, shots, driver=None):
        self.calls.append((model_id, shots, driver))
        if model_id != "m1":
            raise ModelNotFoundException(f"Model '{model_id}' is not an enabled cloud model")
        return {"model_id": model_id, "slug": "fake-director-1", **estimate_shots(FULL, shots)}


@pytest.fixture
def capabilities(monkeypatch):
    fake = FakeCapabilities()
    monkeypatch.setattr(runtime_registries, "_container", SimpleNamespace(cloud_capabilities=fake))
    return fake


async def test_an_admin_gets_exactly_what_the_estimate_route_returns(capabilities):
    route = await CloudCapabilitiesController(capabilities).estimate(
        "m1", DirectorEstimateRequest.model_validate({"shots": FILM})
    )

    estimate = api.estimate_cloud_cost(_user(AccountType.ADMIN), "m1", FILM)

    assert estimate == route.data
    assert estimate["known"] is True and Decimal(estimate["total_usd"]) == Decimal("0.60")
    assert capabilities.calls[0] == capabilities.calls[1]


def test_shots_are_completed_like_the_routes_body(capabilities):
    api.estimate_cloud_cost(_user(AccountType.ADMIN), "m1", [{"params": {"duration_s": 3}}])

    assert capabilities.calls == [("m1", [{"task": "txt2video", "params": {"duration_s": 3}}], None)]


def test_the_driver_is_passed_on(capabilities):
    api.estimate_cloud_cost(_user(AccountType.ADMIN), "m1", FILM, driver="cloud.fake")

    assert capabilities.calls[0][2] == "cloud.fake"


@pytest.mark.parametrize("user", [None, _user(AccountType.USER)])
def test_only_an_admin_sees_a_cost(capabilities, user):
    with pytest.raises(PermissionError):
        api.estimate_cloud_cost(user, "m1", FILM)

    assert capabilities.calls == []


def test_a_shot_the_estimator_refuses_raises_its_error(capabilities):
    with pytest.raises(DirectorEstimateError, match="unknown task 'dance'") as raised:
        api.estimate_cloud_cost(_user(AccountType.ADMIN), "m1", [{"task": "dance"}])

    assert isinstance(raised.value, ValueError)


def test_an_empty_film_is_refused_before_anything_is_estimated(capabilities):
    with pytest.raises(ValidationError) as raised:
        api.estimate_cloud_cost(_user(AccountType.ADMIN), "m1", [])

    assert isinstance(raised.value, ValueError)
    assert capabilities.calls == []


def test_a_model_that_is_not_an_enabled_cloud_model_is_not_found(capabilities):
    with pytest.raises(ModelNotFoundException):
        api.estimate_cloud_cost(_user(AccountType.ADMIN), "nope", FILM)


def test_estimate_cloud_cost_is_importable_from_the_package_root():
    import src.plugin_api as root

    assert root.estimate_cloud_cost is api.estimate_cloud_cost
    assert "estimate_cloud_cost" in root.__all__
    assert "estimate_cloud_cost" in api.__all__
