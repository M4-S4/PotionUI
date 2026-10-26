from types import SimpleNamespace
from typing import Dict, Optional
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.llm import routes as llm_mod
from src.features.llm.harness.profiles import LayeredProfileResolver
from src.features.llm.repository import LLMConfig
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User
from tests.fixtures.llm_profiles import FakeClock, ProbeAdapter, adapter_for, make_config, tiny_seed


class FakeRepository:
    default_provider = None

    def __init__(self, *configs: LLMConfig) -> None:
        self.configs: Dict[str, LLMConfig] = {config.id: config for config in configs}
        self.writes = []

    def get_configuration(self, config_id: str) -> Optional[LLMConfig]:
        return self.configs.get(config_id)

    def set_profile_overrides(self, config_id: str, overrides) -> bool:
        if config_id not in self.configs:
            return False
        self.writes.append((config_id, overrides))
        self.configs[config_id] = self.configs[config_id].model_copy(update={"profile_overrides": overrides or None})
        return True


def _user(account_type: AccountType) -> User:
    return User(username="u", email="u@example.com", password_hash="x", account_type=account_type, id="user-1")


class Harness:
    def __init__(self, resolver: Optional[LayeredProfileResolver], adapter: ProbeAdapter) -> None:
        self.repository = FakeRepository(make_config(id="cfg-1", model="qwen3.8:14b"))
        self.adapter = adapter
        controller = llm_mod.LLMController(
            llm_repository=self.repository,
            llm_service=Mock(),
            settings=Mock(),
            plugin_registry=Mock(),
            profile_resolver=resolver,
        )
        self.controller = controller

    def client(self, account_type: AccountType = AccountType.ADMIN) -> TestClient:
        app = FastAPI()
        app.include_router(llm_mod.build_router(SimpleNamespace(llm_controller=self.controller)))
        app.dependency_overrides[get_current_active_user] = lambda: _user(account_type)
        return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def harness():
    adapter = ProbeAdapter(fields={"context_window": 65536})
    seed = tiny_seed({"id": "all", "fields": {"context_window": 32768, "replay": "send_all"}})
    resolver = LayeredProfileResolver(adapter_for(adapter), seed, clock=FakeClock())
    return Harness(resolver, adapter)


def test_the_schema_route_lists_the_fields(harness):
    response = harness.client().get("/api/llm/profile/schema")

    assert response.status_code == 200
    names = [item["name"] for item in response.json()["data"]["fields"]]
    assert "replay" in names and "context_window" in names and "verified" in names


def test_the_profile_route_reports_each_layer_and_discovers_once(harness):
    client = harness.client()

    first = client.get("/api/llm/configurations/cfg-1/profile").json()["data"]
    second = client.get("/api/llm/configurations/cfg-1/profile").json()["data"]

    assert first == second
    assert harness.adapter.calls == ["cfg-1"]
    assert first["config_id"] == "cfg-1" and first["adapter"] == "probe" and first["model"] == "qwen3.8:14b"
    assert first["discovery_error"] is None and first["discovered_at"]
    assert first["fields"]["context_window"] == {
        "value": 65536, "source": "discovered", "seed": 32768, "discovered": 65536, "override": None,
    }
    assert first["verified"] is None


def test_putting_overrides_stores_them_and_reports_the_new_profile(harness):
    response = harness.client().put(
        "/api/llm/configurations/cfg-1/profile", json={"overrides": {"replay": "preserve", "context_window": 4096}}
    )

    data = response.json()["data"]
    assert response.status_code == 200
    assert harness.repository.configs["cfg-1"].profile_overrides == {"context_window": 4096, "replay": "preserve"}
    assert data["fields"]["replay"] == {
        "value": "preserve", "source": "override", "seed": "send_all", "discovered": None, "override": "preserve",
    }
    assert data["fields"]["context_window"]["value"] == 4096


def test_null_clears_an_override_and_other_overrides_stay(harness):
    client = harness.client()
    client.put("/api/llm/configurations/cfg-1/profile", json={"overrides": {"replay": "preserve", "context_window": 4096}})

    response = client.put("/api/llm/configurations/cfg-1/profile", json={"overrides": {"context_window": None}})

    assert harness.repository.configs["cfg-1"].profile_overrides == {"replay": "preserve"}
    assert response.json()["data"]["fields"]["context_window"]["source"] == "discovered"


def test_clearing_the_last_override_leaves_none_stored(harness):
    client = harness.client()
    client.put("/api/llm/configurations/cfg-1/profile", json={"overrides": {"replay": "never"}})

    client.put("/api/llm/configurations/cfg-1/profile", json={"overrides": {"replay": None}})

    assert harness.repository.configs["cfg-1"].profile_overrides is None


@pytest.mark.parametrize(
    "overrides",
    [{"replay": "sometimes"}, {"bogus": 1}, {"context_window": 10}, {"verified": {"suite": "s", "passed_at": "p"}}],
)
def test_invalid_overrides_are_rejected_before_anything_is_stored(harness, overrides):
    response = harness.client().put("/api/llm/configurations/cfg-1/profile", json={"overrides": overrides})

    assert response.status_code == 422
    assert harness.repository.writes == []


def test_refresh_discovers_again_and_returns_the_profile(harness):
    client = harness.client()
    client.get("/api/llm/configurations/cfg-1/profile")
    harness.adapter.fields = {"context_window": 131072}

    response = client.post("/api/llm/configurations/cfg-1/profile/refresh")

    assert harness.adapter.calls == ["cfg-1", "cfg-1"]
    assert response.json()["data"]["fields"]["context_window"]["value"] == 131072


def test_a_failed_refresh_still_answers_with_the_error_in_the_profile(harness):
    from src.features.llm.model_listing import ModelListingError

    harness.adapter.error = ModelListingError("Couldn't reach it: connection refused.")

    response = harness.client().post("/api/llm/configurations/cfg-1/profile/refresh")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["discovery_error"] == "Couldn't reach it: connection refused."
    assert data["fields"]["context_window"]["source"] == "seed"


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("get", "/api/llm/configurations/missing/profile", None),
        ("put", "/api/llm/configurations/missing/profile", {"overrides": {"replay": "never"}}),
        ("post", "/api/llm/configurations/missing/profile/refresh", None),
    ],
)
def test_an_unknown_configuration_is_reported_not_found(harness, method, path, body):
    response = getattr(harness.client(), method)(path, **({"json": body} if body else {}))

    assert response.json()["success"] is False
    assert response.json()["error"] == "configuration_not_found"
    assert harness.repository.writes == []


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("get", "/api/llm/profile/schema", None),
        ("get", "/api/llm/configurations/cfg-1/profile", None),
        ("put", "/api/llm/configurations/cfg-1/profile", {"overrides": {"replay": "never"}}),
        ("post", "/api/llm/configurations/cfg-1/profile/refresh", None),
    ],
)
def test_only_admins_reach_the_profile_routes(harness, method, path, body):
    response = getattr(harness.client(AccountType.USER), method)(path, **({"json": body} if body else {}))

    assert response.status_code == 403
    assert harness.adapter.calls == [] and harness.repository.writes == []


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("get", "/api/llm/configurations/cfg-1/profile", None),
        ("put", "/api/llm/configurations/cfg-1/profile", {"overrides": {"replay": "never"}}),
        ("post", "/api/llm/configurations/cfg-1/profile/refresh", None),
    ],
)
def test_without_a_resolver_the_routes_say_profiles_are_unavailable(method, path, body):
    harness = Harness(None, ProbeAdapter())

    response = getattr(harness.client(), method)(path, **({"json": body} if body else {}))

    assert response.json()["success"] is False
    assert response.json()["error"] == "profiles_unavailable"
    assert harness.repository.writes == []


def test_the_recommended_sampling_is_served_in_the_same_per_field_shape_as_every_other_field():
    from src.features.llm.harness.profile_seed import default_seed

    adapter = ProbeAdapter(name="ollama-chat")
    resolver = LayeredProfileResolver(adapter_for(adapter), default_seed(), clock=FakeClock())
    harness = Harness(resolver, adapter)
    harness.repository.configs["cfg-1"] = make_config(id="cfg-1", model="qwen3.6:27b", type="ollama")

    field = harness.client().get("/api/llm/configurations/cfg-1/profile").json()["data"]["fields"]["recommended_sampling"]

    assert field == {
        "value": {"temperature": 1.0, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0},
        "source": "seed",
        "seed": {"temperature": 1.0, "top_p": 0.95, "top_k": 20.0, "min_p": 0.0},
        "discovered": None,
        "override": None,
    }


def test_an_override_of_the_recommended_sampling_is_reported_as_the_override(harness):
    response = harness.client().put(
        "/api/llm/configurations/cfg-1/profile", json={"overrides": {"recommended_sampling": {"temperature": 0.3}}}
    )

    field = response.json()["data"]["fields"]["recommended_sampling"]
    assert field["value"] == {"temperature": 0.3} and field["source"] == "override" and field["override"] == {"temperature": 0.3}


def test_the_requested_context_size_is_only_ever_an_override(harness):
    client = harness.client()
    discovered = client.get("/api/llm/configurations/cfg-1/profile").json()["data"]["fields"]
    assert discovered["context_window"]["source"] == "discovered"
    assert discovered["context_request"] == {"value": None, "source": "seed", "seed": None, "discovered": None, "override": None}

    overridden = client.put(
        "/api/llm/configurations/cfg-1/profile", json={"overrides": {"context_request": 16384}}
    ).json()["data"]["fields"]["context_request"]

    assert overridden == {"value": 16384, "source": "override", "seed": None, "discovered": None, "override": 16384}
