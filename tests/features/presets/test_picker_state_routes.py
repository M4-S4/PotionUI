import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import tests.conftest as ct
from src.features.presets.operations import picker_state
from src.features.presets.routes import PresetController, build_router
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User
from src.platform.settings.repository import SettingRepository
from src.platform.settings.settings import Settings

_MIGRATIONS = Path("src/platform/database/migrations")


def _load(stem: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _MIGRATIONS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _user(user_id: str) -> User:
    from src.platform.database.database import db
    with db.get_cursor() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, username, email, password_hash, account_type) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_id, f"{user_id}@x.com", "h", "USER"),
        )
    return User(id=user_id, username=user_id, email=f"{user_id}@x.com", password_hash="h", account_type=AccountType.USER)


@pytest.fixture
def harness(monkeypatch):
    test_database = ct.TestDatabase()
    with patch("src.platform.database.database.db", test_database):
        _load("001_baseline", f"m001_{id(test_database)}").up()
        _load("063_preset_picker_state", f"m063_{id(test_database)}").up()
        users = {"ada": _user("ada"), "bo": _user("bo")}
        visible = {
            "ada": ["sdxl", "krea", "wan", "qwen", *[f"p{i}" for i in range(60)]],
            "bo": ["sdxl", "krea"],
        }
        monkeypatch.setattr(
            picker_state, "list_presets", lambda collaborators, user: [{"id": preset_id} for preset_id in visible[user.id]]
        )
        controller = PresetController(SimpleNamespace(settings=Settings(SettingRepository())), Mock())
        app = FastAPI()
        app.include_router(build_router(SimpleNamespace(preset_controller=controller)))
        current = {"user": users["ada"]}
        app.dependency_overrides[get_current_active_user] = lambda: current["user"]

        def as_user(name: str):
            current["user"] = users[name]

        yield SimpleNamespace(client=TestClient(app), as_user=as_user, visible=visible)
    test_database.close()


def _state(response):
    body = response.json()
    assert body["success"] is True, body
    return body["data"]


def test_a_new_user_starts_with_nothing_pinned_or_recent(harness):
    assert _state(harness.client.get("/api/presets/picker-state")) == {"pinned": [], "recent": []}


def test_pinning_is_idempotent_and_unpinning_removes_it(harness):
    client = harness.client
    assert _state(client.put("/api/presets/picker-state/pinned/krea"))["pinned"] == ["krea"]
    assert _state(client.put("/api/presets/picker-state/pinned/krea"))["pinned"] == ["krea"]
    assert _state(client.put("/api/presets/picker-state/pinned/wan"))["pinned"] == ["krea", "wan"]
    assert _state(client.delete("/api/presets/picker-state/pinned/krea"))["pinned"] == ["wan"]
    assert _state(client.delete("/api/presets/picker-state/pinned/krea"))["pinned"] == ["wan"]
    assert _state(client.get("/api/presets/picker-state"))["pinned"] == ["wan"]


def test_recent_moves_the_chosen_preset_to_the_front_without_duplicates(harness):
    client = harness.client
    for preset_id in ["sdxl", "krea", "wan", "krea"]:
        state = _state(client.post(f"/api/presets/picker-state/recent/{preset_id}"))
    assert state["recent"] == ["krea", "wan", "sdxl"]
    assert _state(client.get("/api/presets/picker-state"))["recent"] == ["krea", "wan", "sdxl"]


def test_recent_keeps_the_last_eight(harness):
    client = harness.client
    for i in range(12):
        state = _state(client.post(f"/api/presets/picker-state/recent/p{i}"))
    assert state["recent"] == [f"p{i}" for i in range(11, 3, -1)]


def test_pinned_keeps_the_newest_fifty(harness):
    client = harness.client
    for i in range(55):
        state = _state(client.put(f"/api/presets/picker-state/pinned/p{i}"))
    assert len(state["pinned"]) == 50
    assert state["pinned"][0] == "p5"
    assert state["pinned"][-1] == "p54"


def test_each_user_keeps_their_own_lists(harness):
    client = harness.client
    client.put("/api/presets/picker-state/pinned/krea")
    client.post("/api/presets/picker-state/recent/wan")
    harness.as_user("bo")
    assert _state(client.get("/api/presets/picker-state")) == {"pinned": [], "recent": []}
    client.put("/api/presets/picker-state/pinned/sdxl")
    harness.as_user("ada")
    assert _state(client.get("/api/presets/picker-state")) == {"pinned": ["krea"], "recent": ["wan"]}


def test_presets_the_user_can_no_longer_see_drop_out(harness):
    client = harness.client
    client.put("/api/presets/picker-state/pinned/qwen")
    client.post("/api/presets/picker-state/recent/qwen")
    client.post("/api/presets/picker-state/recent/sdxl")
    harness.visible["ada"].remove("qwen")
    assert _state(client.get("/api/presets/picker-state")) == {"pinned": [], "recent": ["sdxl"]}


def test_an_unknown_or_hidden_preset_is_refused(harness):
    client = harness.client
    for response in (
        client.put("/api/presets/picker-state/pinned/nope"),
        client.post("/api/presets/picker-state/recent/nope"),
    ):
        body = response.json()
        assert body["success"] is False
        assert body["error"] == "preset_not_found"
    harness.as_user("bo")
    assert client.put("/api/presets/picker-state/pinned/wan").json()["error"] == "preset_not_found"
    assert _state(client.get("/api/presets/picker-state")) == {"pinned": [], "recent": []}


def test_pinning_an_already_pinned_preset_never_pushes_others_out(harness):
    client = harness.client
    for i in range(49):
        client.put(f"/api/presets/picker-state/pinned/p{i}")
    client.put("/api/presets/picker-state/pinned/krea")
    for _ in range(5):
        state = _state(client.put("/api/presets/picker-state/pinned/krea"))
    assert len(state["pinned"]) == 50
    assert state["pinned"][0] == "p0"
