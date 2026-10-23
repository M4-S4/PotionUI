import re
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.features.models import digests as digests_module
from src.features.models import indexer as indexer_module
from src.features.models.digest_queue_repository import (
    MAX_ATTEMPTS,
    PRIORITY_BACKGROUND,
    PRIORITY_CHOSEN,
    PRIORITY_DEMAND,
)
from src.features.models.indexing_coordinator import ModelIndexingCoordinator
from src.features.models.repository import model_repo
from src.features.models.routes import ModelController, build_router
from src.platform.database import get_cursor
from src.platform.security.current_user import get_current_active_user, get_current_admin_user
from src.platform.security.user import AccountType, User
from tests.features.models.fakes import OpenPlugins
from tests.features.models.rows_first_support import SMALL_LIMIT, Harness, _blob
from tests.features.models.test_indexer_roots import _models

TYPES_FILE = Path(__file__).resolve().parents[3] / "frontend/src/lib/services/api/models.ts"


def _admin():
    user = Mock(spec=User)
    user.id = "admin-1"
    user.account_type = AccountType.ADMIN
    return user


@pytest.fixture
def stack(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
    harness = Harness(tmp_path)
    reconciler = MagicMock()
    reconciler.reconcile = AsyncMock()
    coordinator = ModelIndexingCoordinator(
        model_repository=model_repo, plugin_registry=OpenPlugins(), scanner=harness.scanner,
        native_availability_projector=reconciler, spawn=lambda target: target(),
    )
    harness.manager._on_applied = coordinator.on_digests_applied
    harness.coordinator = coordinator
    return harness


@pytest.fixture
def app(stack):
    collaborators = SimpleNamespace(indexing=stack.coordinator, digests=stack.manager)
    container = Mock()
    container.model_controller = ModelController(collaborators, Mock(), Mock())
    fastapi_app = FastAPI()
    fastapi_app.include_router(build_router(container))
    fastapi_app.dependency_overrides[get_current_admin_user] = _admin
    return fastapi_app


def client_for(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _fill(stack, count, prefix="m", size=200):
    return [stack.put("home", f"{prefix}{index:03d}.safetensors", _blob(index, size)) for index in range(count)]


def _scan(stack):
    stack.coordinator.start_indexing(trigger="manual")


def _data(response):
    assert response.status_code == 200, response.text
    return response.json()["data"]


class TestManualRescan:
    @pytest.mark.asyncio
    async def test_the_index_route_runs_a_scan_and_lists_the_new_files(self, app, stack):
        _fill(stack, 2)

        async with client_for(app) as client:
            data = _data(await client.post("/api/models/index"))

        assert (data["state"], data["indexed"], data["trigger"]) == ("done", 2, "manual")
        assert data["digests"]["pending_total"] == 2

    @pytest.mark.asyncio
    async def test_the_index_route_during_a_running_scan_asks_for_one_more_pass_after_it(self, app, stack):
        _fill(stack, 1)
        with stack.coordinator._lock:
            stack.coordinator._state = "scanning"
            stack.coordinator._worker_active = True

        async with client_for(app) as client:
            data = _data(await client.post("/api/models/index"))

        assert data["restart_pending"] is True


class TestTheStatusRouteDigestsBlock:
    @pytest.mark.asyncio
    async def test_the_status_route_carries_the_digests_block(self, app, stack):
        _fill(stack, 3)
        _scan(stack)

        async with client_for(app) as client:
            data = _data(await client.get("/api/models/indexing/status"))

        digests = data["digests"]
        assert (digests["policy"], digests["level"], digests["pending_total"], digests["queued"]) == (
            "background", "low", 3, 3,
        )
        assert digests["plan"] == {
            "awaiting_choice": False, "files": 0, "bytes": 0, "min_files": 20, "min_bytes": 64 * 1024 ** 3,
        }
        assert data["state"] == "done"

    @pytest.mark.asyncio
    async def test_a_big_library_waits_for_the_choice_and_says_so(self, app, stack):
        _fill(stack, 21)
        _scan(stack)

        async with client_for(app) as client:
            digests = _data(await client.get("/api/models/indexing/status"))["digests"]

        assert (digests["policy"], digests["queued"], digests["deferred"]) == ("unset", 0, 21)
        assert digests["plan"]["awaiting_choice"] is True
        assert (digests["plan"]["files"], digests["plan"]["bytes"]) == (21, 21 * 200)


class TestSettings:
    @pytest.mark.asyncio
    async def test_the_level_and_the_policy_come_back_together(self, app, stack):
        async with client_for(app) as client:
            both = _data(await client.put("/api/models/indexing/settings", json={"level": "normal", "policy": "on_demand"}))
            neither = _data(await client.put("/api/models/indexing/settings", json={}))

        assert both == {"level": "normal", "policy": "on_demand"}
        assert neither == {"level": "normal", "policy": "on_demand"}
        assert stack.manager.policy == "on_demand"

    @pytest.mark.asyncio
    async def test_choosing_background_starts_the_work(self, app, stack):
        _fill(stack, 21)
        _scan(stack)

        async with client_for(app) as client:
            _data(await client.put("/api/models/indexing/settings", json={"policy": "background"}))
            digests = _data(await client.get("/api/models/indexing/status"))["digests"]

        assert (digests["policy"], digests["queued"], digests["deferred"]) == ("background", 21, 0)
        assert digests["plan"]["awaiting_choice"] is False

    @pytest.mark.asyncio
    async def test_an_unknown_policy_is_unprocessable_and_changes_nothing(self, app, stack):
        async with client_for(app) as client:
            response = await client.put("/api/models/indexing/settings", json={"policy": "sometimes"})

        assert response.status_code == 422
        assert stack.manager.policy == "unset"


class TestPauseAndResume:
    @pytest.mark.asyncio
    async def test_pausing_is_remembered_and_the_status_follows(self, app, stack):
        async with client_for(app) as client:
            paused = _data(await client.post("/api/models/indexing/pause"))
            resumed = _data(await client.post("/api/models/indexing/resume"))

        assert (paused["paused"], paused["pause_reason"]) == (True, None)
        assert resumed["paused"] is False
        assert stack.manager.paused is False

    @pytest.mark.asyncio
    async def test_the_pause_survives_a_restart_of_the_manager(self, app, stack):
        async with client_for(app) as client:
            await client.post("/api/models/indexing/pause")

        reborn = digests_module.DigestManager(
            stack.queue, stack.governor, stack.scanner.location_path,
            setting_repository=stack.manager._settings,
        )

        assert reborn.paused is True


class TestPending:
    @pytest.mark.asyncio
    async def test_the_list_has_the_page_the_totals_of_the_match_and_the_type_counts(self, app, stack):
        _fill(stack, 4, prefix="lora_")
        stack.put("home", "vae_x.safetensors", _blob(9, 300))
        _scan(stack)
        with get_cursor() as cursor:
            cursor.execute("UPDATE models SET model_type = 'vae' WHERE filename = 'vae_x.safetensors'")

        async with client_for(app) as client:
            data = _data(await client.get(
                "/api/models/indexing/pending",
                params={"query": "LORA", "model_type": "lora", "limit": 2, "offset": 1},
            ))

        assert [item["filename"] for item in data["items"]] == ["lora_001.safetensors", "lora_002.safetensors"]
        assert (data["total"], data["total_bytes"]) == (4, 800)
        assert data["by_type"] == {"lora": 4, "vae": 1}
        assert set(data["items"][0]) == {
            "model_id", "location_id", "filename", "model_type", "size", "root_label", "rel_path", "queued",
        }
        assert data["items"][0]["queued"] is True

    @pytest.mark.asyncio
    async def test_the_filters_and_bounds_are_validated(self, app, stack):
        async with client_for(app) as client:
            low = await client.get("/api/models/indexing/pending", params={"limit": 0})
            high = await client.get("/api/models/indexing/pending", params={"limit": 201})
            negative = await client.get("/api/models/indexing/pending", params={"offset": -1})
            by_root = await client.get("/api/models/indexing/pending", params={"root_id": "elsewhere"})

        assert (low.status_code, high.status_code, negative.status_code) == (422, 422, 422)
        assert _data(by_root)["total"] == 0


class TestQueueing:
    @pytest.mark.asyncio
    async def test_all_queues_every_pending_file_in_the_background(self, app, stack):
        _fill(stack, 21)
        _scan(stack)

        async with client_for(app) as client:
            data = _data(await client.post("/api/models/indexing/digests", json={"mode": "all"}))

        assert data["queued"] == 21
        assert data["digests"]["queued"] == 21
        assert {row["priority"] for row in stack.queue.next_jobs(100, 0)} == {PRIORITY_BACKGROUND}

    @pytest.mark.asyncio
    async def test_a_failed_file_is_listed_as_not_queued_and_an_explicit_pick_re_arms_it(self, app, stack):
        _fill(stack, 2)
        _scan(stack)
        model = _models()[0]
        location_id = stack.locations(model)[0]["id"]
        for _ in range(MAX_ATTEMPTS):
            stack.queue.record_failure(location_id, "boom", 0.0, ())

        async with client_for(app) as client:
            before = _data(await client.get("/api/models/indexing/pending"))
            picked = _data(await client.post(
                "/api/models/indexing/digests", json={"mode": "selected", "model_ids": [model.id]}
            ))
            after = _data(await client.get("/api/models/indexing/pending"))

        flags = {item["model_id"]: item["queued"] for item in before["items"]}
        assert flags[model.id] is False
        assert picked["queued"] == 1 and picked["digests"]["failed"] == []
        assert {item["model_id"]: item["queued"] for item in after["items"]}[model.id] is True

    @pytest.mark.asyncio
    async def test_selected_models_are_queued_ahead_of_everything_else(self, app, stack):
        _fill(stack, 21)
        _scan(stack)
        chosen = [m for m in _models() if m.filename in ("m003.safetensors", "m004.safetensors")]

        async with client_for(app) as client:
            data = _data(await client.post(
                "/api/models/indexing/digests", json={"mode": "selected", "model_ids": [m.id for m in chosen]}
            ))

        assert data["queued"] == 2
        assert {row["filename"] for row in stack.queue.next_jobs(10, 0)} == {"m003.safetensors", "m004.safetensors"}
        assert {row["priority"] for row in stack.queue.next_jobs(10, 0)} == {PRIORITY_CHOSEN}
        assert stack.manager.policy == "unset"

    @pytest.mark.asyncio
    async def test_matching_with_an_unticked_model_queues_the_rest_of_the_match(self, app, stack):
        _fill(stack, 21, prefix="lora_")
        stack.put("home", "vae_x.safetensors", _blob(9, 300))
        _scan(stack)
        unticked = [m for m in _models() if m.filename == "lora_005.safetensors"][0]

        async with client_for(app) as client:
            data = _data(await client.post(
                "/api/models/indexing/digests",
                json={"mode": "matching", "query": "lora_", "exclude_model_ids": [unticked.id]},
            ))

        assert data["queued"] == 20
        queued = {row["filename"] for row in stack.queue.next_jobs(100, 0)}
        assert "lora_005.safetensors" not in queued and "vae_x.safetensors" not in queued
        assert {row["priority"] for row in stack.queue.next_jobs(100, 0)} == {PRIORITY_CHOSEN}

    @pytest.mark.asyncio
    async def test_matching_by_type_queues_only_that_type(self, app, stack):
        _fill(stack, 2)
        _scan(stack)
        with get_cursor() as cursor:
            cursor.execute("UPDATE models SET model_type = 'vae' WHERE filename = 'm000.safetensors'")

        async with client_for(app) as client:
            data = _data(await client.post(
                "/api/models/indexing/digests", json={"mode": "matching", "model_type": "vae"}
            ))

        assert data["queued"] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "body", [{"mode": "everything"}, {"mode": "selected"}, {}, {"mode": "selected", "model_ids": "m1"}]
    )
    async def test_a_malformed_request_is_unprocessable_and_queues_nothing(self, app, stack, body):
        _fill(stack, 2)
        _scan(stack)
        stack.queue.clear_background()

        async with client_for(app) as client:
            response = await client.post("/api/models/indexing/digests", json=body)

        assert response.status_code == 422
        assert stack.queue.next_jobs(10, 0) == []

    @pytest.mark.asyncio
    async def test_selecting_no_models_queues_nothing(self, app, stack):
        async with client_for(app) as client:
            data = _data(await client.post("/api/models/indexing/digests", json={"mode": "selected", "model_ids": []}))

        assert data["queued"] == 0


class TestModelDigest:
    @pytest.mark.asyncio
    async def test_a_model_without_a_digest_is_queued_with_an_estimate_and_asking_again_changes_nothing(
        self, app, stack, monkeypatch
    ):
        _fill(stack, 1)
        _scan(stack)
        model = _models()[0]
        release = threading.Event()
        started = []
        monkeypatch.setattr(
            stack.manager, "ensure_digest", lambda model_id, timeout=None: started.append(model_id) or release.wait(5)
        )

        async with client_for(app) as client:
            first = _data(await client.post(f"/api/models/{model.id}/digest"))
            second = _data(await client.post(f"/api/models/{model.id}/digest"))
            third = _data(await client.post(f"/api/models/{model.id}/digest"))
        release.set()

        assert first["state"] == "queued" and isinstance(first["eta_seconds"], int)
        assert (second["state"], third["state"]) == ("queued", "queued")
        assert set(first) == {"state", "eta_seconds"}
        assert len(stack.queue.next_jobs(10, 0)) == 1
        assert stack.queue.next_jobs(10, 0)[0]["priority"] == PRIORITY_DEMAND
        assert started == [model.id]

    @pytest.mark.asyncio
    async def test_a_file_that_failed_out_answers_failed_until_a_retry_is_asked_for(self, app, stack, monkeypatch):
        _fill(stack, 1)
        _scan(stack)
        model = _models()[0]
        location_id = stack.locations(model)[0]["id"]
        for _ in range(MAX_ATTEMPTS):
            stack.queue.record_failure(location_id, "boom", 0.0, ())
        release = threading.Event()
        started = []
        monkeypatch.setattr(
            stack.manager, "ensure_digest", lambda model_id, timeout=None: started.append(model_id) or release.wait(5)
        )

        async with client_for(app) as client:
            failed = _data(await client.post(f"/api/models/{model.id}/digest"))
            still_failed = _data(await client.post(f"/api/models/{model.id}/digest?retry=false"))
            retried = _data(await client.post(f"/api/models/{model.id}/digest?retry=true"))
        release.set()

        assert failed == still_failed == {"state": "failed", "eta_seconds": None}
        assert retried["state"] == "queued" and started == [model.id]

    @pytest.mark.asyncio
    async def test_a_model_that_already_has_its_digest_is_ready(self, app, stack):
        stack.put("home", "tiny.safetensors", b"tiny")
        _scan(stack)
        model = _models()[0]

        async with client_for(app) as client:
            data = _data(await client.post(f"/api/models/{model.id}/digest"))

        assert data == {"state": "ready", "eta_seconds": 0}

    @pytest.mark.asyncio
    async def test_an_unknown_model_is_not_found(self, app, stack):
        async with client_for(app) as client:
            response = await client.post("/api/models/nope/digest")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_a_model_with_no_readable_file_cannot_be_hashed(self, app, stack):
        _fill(stack, 1)
        _scan(stack)
        model = _models()[0]
        with get_cursor() as cursor:
            cursor.execute("UPDATE model_locations SET status = 'missing'")

        async with client_for(app) as client:
            response = await client.post(f"/api/models/{model.id}/digest")

        assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/api/models/indexing/status", None),
        ("put", "/api/models/indexing/settings", {"level": "max"}),
        ("post", "/api/models/indexing/pause", None),
        ("post", "/api/models/indexing/resume", None),
        ("get", "/api/models/indexing/pending", None),
        ("post", "/api/models/indexing/digests", {"mode": "all"}),
        ("post", "/api/models/some-model/digest", None),
    ],
)
async def test_every_indexing_route_is_for_admins_only(app, stack, method, path, body):
    regular = Mock(spec=User)
    regular.id = "user-1"
    regular.account_type = AccountType.USER
    app.dependency_overrides.clear()
    app.dependency_overrides[get_current_active_user] = lambda: regular

    async with client_for(app) as client:
        response = await getattr(client, method)(path, **({"json": body} if body is not None else {}))

    assert response.status_code == 403
    assert stack.manager.policy == "unset" and stack.manager.paused is False


def _ts_fields(name, optional=True):
    source = TYPES_FILE.read_text(encoding="utf-8")
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    assert match, f"{name} is not declared in {TYPES_FILE.name}"
    pattern = r"^\s*(\w+)\??:" if optional else r"^\s*(\w+):"
    return {field for field in re.findall(pattern, match.group(1), re.M)}


@pytest.mark.skipif(not TYPES_FILE.exists(), reason="the frontend sources are not part of this checkout")
class TestTheContractTheFrontendIsBuiltAgainst:
    @pytest.mark.asyncio
    async def test_the_digests_block_has_exactly_the_fields_the_ui_declares(self, app, stack):
        _fill(stack, 2)
        _scan(stack)
        failed = [m for m in _models()][0]
        location = stack.scanner.locations.list_for_model(failed.id)[0]
        for _ in range(3):
            stack.queue.record_failure(location["id"], "unreadable", 0, ())
        worker = digests_module._Worker()
        worker.job = digests_module._Job(
            key="k", kind="location", path=None, size=10, mtime_ns=None, priority=0, filename="x.safetensors",
            model_id=failed.id, done=4,
        )
        stack.manager._workers = [worker]

        async with client_for(app) as client:
            digests = _data(await client.get("/api/models/indexing/status"))["digests"]

        assert set(digests) == _ts_fields("IndexingDigests")
        assert set(digests["levels"][0]) == _ts_fields("IndexingLevelProfile")
        assert set(digests["plan"]) == _ts_fields("IndexingDigestPlan")
        assert set(digests["current"][0]) == _ts_fields("IndexingDigestCurrent")
        assert set(digests["failed"][0]) == _ts_fields("IndexingDigestFailure")
        assert [level["id"] for level in digests["levels"]] == ["low", "normal", "max"]

    @pytest.mark.asyncio
    async def test_the_pending_list_and_the_settings_and_digest_answers_match_the_ui_types(self, app, stack, monkeypatch):
        _fill(stack, 1)
        _scan(stack)
        model = _models()[0]
        monkeypatch.setattr(stack.manager, "ensure_digest", lambda model_id, timeout=None: None)

        async with client_for(app) as client:
            pending = _data(await client.get("/api/models/indexing/pending"))
            settings = _data(await client.put("/api/models/indexing/settings", json={"level": "low", "policy": "background"}))
            digest = _data(await client.post(f"/api/models/{model.id}/digest"))

        assert set(pending) == _ts_fields("IndexingPendingPage")
        assert set(pending["items"][0]) == _ts_fields("IndexingPendingItem")
        assert set(settings) == _ts_fields("IndexingSettingsResult")
        assert _ts_fields("ModelDigestRequestResult", optional=False) == set(digest)
        assert set(digest) <= _ts_fields("ModelDigestRequestResult")

    def test_the_request_types_name_the_same_modes_and_filters_as_the_api(self):
        source = TYPES_FILE.read_text(encoding="utf-8")

        assert {"all", "selected", "matching"} == set(re.findall(r"mode: '(\w+)'", source))
        assert {"query", "model_type", "offset", "limit"} <= _ts_fields("IndexingPendingParams")
        assert {"level", "policy"} == _ts_fields("IndexingSettingsPayload")
