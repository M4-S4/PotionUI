from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.presets import operations
from src.features.presets.routes import PresetController, build_router
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType


def _client(monkeypatch, presets, backends):
    monkeypatch.setattr(operations, "list_presets", lambda collaborators, user, include_uninstalled=False: [dict(p) for p in presets])
    registry = Mock()
    registry.get_backends_for_engine.side_effect = lambda engine: backends.get(engine, [])
    registry.engine_supports_prompt_batch.return_value = False
    controller = PresetController(SimpleNamespace(), registry)
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(preset_controller=controller)))
    user = SimpleNamespace(id="u1", account_type=AccountType.USER)
    app.dependency_overrides[get_current_active_user] = lambda: user
    return TestClient(app), registry


def test_the_list_marks_presets_whose_engine_has_no_backend(monkeypatch):
    presets = [
        {"id": "sdxl", "engine": "native"},
        {"id": "flux1", "engine": "comfyui"},
        {"id": "krea", "engine": "native"},
        {"id": "loose"},
    ]
    client, registry = _client(monkeypatch, presets, {"native": [object()]})
    body = client.get("/api/presets").json()
    assert body["success"] is True
    assert {p["id"]: p["engine_available"] for p in body["data"]} == {
        "sdxl": True,
        "flux1": False,
        "krea": True,
        "loose": False,
    }
    assert sorted(call.args[0] for call in registry.get_backends_for_engine.call_args_list) == ["comfyui", "native"]
