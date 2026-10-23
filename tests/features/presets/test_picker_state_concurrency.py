import importlib.util
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tests.conftest as ct
from src.features.presets.operations import picker_state
from src.platform.security.user import AccountType, User
from src.platform.settings.repository import SettingRepository
from src.platform.settings.settings import Settings

_MIGRATIONS = Path("src/platform/database/migrations")
_IDS = [f"p{i}" for i in range(60)]


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


class _PausingSettings(Settings):
    def __init__(self, repository: SettingRepository):
        super().__init__(repository)
        self.pause = lambda user_id: time.sleep(0.005)

    def get_setting(self, key, default=None, user_id=None):
        value = super().get_setting(key, default=default, user_id=user_id)
        if key in (picker_state.PINNED_KEY, picker_state.RECENT_KEY):
            self.pause(user_id)
        return value


@pytest.fixture
def picker(monkeypatch):
    test_database = ct.TestDatabase()
    with patch("src.platform.database.database.db", test_database):
        _load("001_baseline", f"m001_{id(test_database)}").up()
        _load("063_preset_picker_state", f"m063_{id(test_database)}").up()
        monkeypatch.setattr(picker_state, "list_presets", lambda collaborators, user: [{"id": preset_id} for preset_id in _IDS])
        settings = _PausingSettings(SettingRepository())
        yield SimpleNamespace(
            collaborators=SimpleNamespace(settings=settings),
            settings=settings,
            ada=_user("ada"),
            bo=_user("bo"),
        )
    test_database.close()


def _all_at_once(operation, ids):
    start = threading.Barrier(len(ids))

    def run(preset_id):
        start.wait(timeout=5)
        return operation(preset_id)

    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        list(pool.map(run, ids))


def test_concurrent_pins_unpins_and_recent_pushes_for_one_user_all_land_within_the_caps(picker):
    collaborators, ada = picker.collaborators, picker.ada

    _all_at_once(lambda preset_id: picker_state.pin_preset(collaborators, ada, preset_id), _IDS[:12])
    assert sorted(picker_state.get_picker_state(collaborators, ada)["pinned"]) == sorted(_IDS[:12])

    _all_at_once(lambda preset_id: picker_state.unpin_preset(collaborators, ada, preset_id), _IDS[:12])
    assert picker_state.get_picker_state(collaborators, ada)["pinned"] == []

    _all_at_once(lambda preset_id: picker_state.pin_preset(collaborators, ada, preset_id), _IDS[:52])
    pinned = picker_state.get_picker_state(collaborators, ada)["pinned"]
    assert len(pinned) == picker_state.PINNED_LIMIT
    assert len(set(pinned)) == picker_state.PINNED_LIMIT

    _all_at_once(lambda preset_id: picker_state.push_recent_preset(collaborators, ada, preset_id), _IDS[:12])
    recent = picker_state.get_picker_state(collaborators, ada)["recent"]
    assert len(recent) == picker_state.RECENT_LIMIT
    assert len(set(recent)) == picker_state.RECENT_LIMIT


def test_one_user_holding_the_write_never_blocks_another_user(picker):
    collaborators, ada, bo = picker.collaborators, picker.ada, picker.bo
    ada_inside = threading.Event()
    release = threading.Event()

    def pause(user_id):
        if user_id == ada.id:
            ada_inside.set()
            release.wait(timeout=5)

    picker.settings.pause = pause
    with ThreadPoolExecutor(max_workers=2) as pool:
        ada_pin = pool.submit(picker_state.pin_preset, collaborators, ada, "p1")
        assert ada_inside.wait(timeout=5)
        bo_pin = pool.submit(picker_state.pin_preset, collaborators, bo, "p2")
        try:
            assert bo_pin.result(timeout=2)["pinned"] == ["p2"]
            assert not ada_pin.done()
        finally:
            release.set()
        assert ada_pin.result(timeout=5)["pinned"] == ["p1"]
