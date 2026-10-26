"""GET /api/setup/machine: admin-only host facts, never a 500 on a failed probe."""

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.setup import machine
from src.features.setup.routes import build_router
from src.platform.runtime.gpu_profile import GpuProfile, build_gpu_profile
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User

GIB = 1024 ** 3
Usage = namedtuple("Usage", "total used free")


def _user(account_type):
    return User(
        username="u", email="u@example.com", password_hash="x",
        account_type=account_type, id="user-1",
    )


def _roots(path="/models/home"):
    roots = Mock()
    roots.home_dir.return_value = Path(path)
    return roots


def _client(user, roots=None):
    container = SimpleNamespace(
        instance_claim_repository=Mock(),
        claim_token_store=Mock(),
        settings=Mock(),
        model_roots=roots if roots is not None else _roots(),
    )
    app = FastAPI()
    app.include_router(build_router(container))
    app.dependency_overrides[get_current_active_user] = lambda: user
    return TestClient(app)


@pytest.fixture
def probes(monkeypatch):
    monkeypatch.setattr(machine, "detect_gpu_profile", lambda: build_gpu_profile((12, 0), 32.0, "NVIDIA GeForce RTX 5090"))
    monkeypatch.setattr(machine, "get_system_memory", lambda: SimpleNamespace(total=64 * GIB, available=32 * GIB))
    monkeypatch.setattr(machine.shutil, "disk_usage", lambda path: Usage(1000 * GIB, 600 * GIB, 400 * GIB))


def test_full_payload(probes):
    body = _client(_user(AccountType.ADMIN)).get("/api/setup/machine").json()
    assert body == {
        "gpu": {
            "name": "NVIDIA GeForce RTX 5090",
            "vram_bytes": 32 * GIB,
            "compute_capability": "12.0",
            "supported": True,
        },
        "ram_total_bytes": 64 * GIB,
        "models_disk": {"path": str(Path("/models/home")), "free_bytes": 400 * GIB, "total_bytes": 1000 * GIB},
    }


def test_non_admin_gets_404(probes):
    assert _client(_user(AccountType.USER)).get("/api/setup/machine").status_code == 404


@pytest.mark.parametrize(
    "capability,supported",
    [((7, 5), True), ((7, 0), False), ((6, 1), False), ((8, 6), True)],
)
def test_supported_is_turing_or_newer(monkeypatch, probes, capability, supported):
    monkeypatch.setattr(machine, "detect_gpu_profile", lambda: build_gpu_profile(capability, 8.0, "Card"))
    gpu = _client(_user(AccountType.ADMIN)).get("/api/setup/machine").json()["gpu"]
    assert gpu["supported"] is supported


def test_no_gpu_is_null(monkeypatch, probes):
    monkeypatch.setattr(machine, "detect_gpu_profile", lambda: GpuProfile())
    body = _client(_user(AccountType.ADMIN)).get("/api/setup/machine").json()
    assert body["gpu"] is None
    assert body["ram_total_bytes"] == 64 * GIB


def test_each_probe_failure_is_null_not_500(monkeypatch, probes):
    def boom(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(machine, "detect_gpu_profile", boom)
    monkeypatch.setattr(machine, "get_system_memory", boom)
    monkeypatch.setattr(machine.shutil, "disk_usage", boom)
    response = _client(_user(AccountType.ADMIN)).get("/api/setup/machine")
    assert response.status_code == 200
    assert response.json() == {"gpu": None, "ram_total_bytes": None, "models_disk": None}


def test_unresolvable_models_home_is_null(probes):
    roots = Mock()
    roots.home_dir.side_effect = RuntimeError("missing")
    body = _client(_user(AccountType.ADMIN), roots=roots).get("/api/setup/machine").json()
    assert body["models_disk"] is None
    assert body["gpu"] is not None
