import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.generation.history_facade import GenerationHistoryFacade
from src.features.generation.output_broadcaster import GenerationOutputBroadcaster
from src.features.generation.records import Generation
from src.features.generation.repository import GenerationRepository
from src.features.generation.routes import GenerationController, build_admin_router, build_router
from src.features.generation.run_report_recorder import RunReportRecorder
from src.features.generation.run_report_repository import GenerationRunReportRepository
from src.features.generation.status_tracker import GenerationStatusTracker
from src.pipelines.outputs import DiffTextGenerationOutput, ModelGenerationOutput, ModelsGenerationOutput
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User
from src.platform.util.ids import generate_ulid
from src.platform.websocket.connection_hub import ConnectionHub
from tests.fixtures.persistence_base import PersistenceTestBase

MODEL_PATH = "/srv/potion/models/diffusion_models/secret_dit_fp16.safetensors"


class _Socket:
    def __init__(self):
        self.sent = []

    async def accept(self):
        return None

    async def send_text(self, text):
        self.sent.append(json.loads(text))


def _user(user_id, account_type):
    return User(
        id=user_id,
        username=user_id,
        email=f"{user_id}@example.com",
        password_hash="$2b$12$x",
        account_type=account_type,
        created_at=datetime.utcnow(),
        last_login=None,
    )


class TestArtifactsAreAdminOnly(PersistenceTestBase):

    def setUp(self):
        super().setUp()
        self.generation_repo = GenerationRepository()
        self.recorder = RunReportRecorder(GenerationRunReportRepository())
        self.hub = ConnectionHub()
        self.tracker = GenerationStatusTracker()
        self.broadcaster = GenerationOutputBroadcaster(self.hub, self.tracker, self.recorder)
        self.controller = GenerationController(
            Mock(status_tracker=self.tracker),
            GenerationHistoryFacade(
                generation_repo=self.generation_repo,
                file_service=Mock(),
                plugin_registry=Mock(),
                run_report_repository=Mock(),
            ),
            Mock(),
            self.recorder,
            None,
            self.broadcaster,
        )
        owner_id = self.create_test_user("owner-1", "owner", "owner@example.com")
        self.owner = _user(owner_id, AccountType.USER)
        self.admin = _user("admin-1", AccountType.ADMIN)
        self.generation = self.generation_repo.create(Generation(
            id=generate_ulid(), preset_id="preset-1", form_data={"prompt": "a cat"},
            user_id=owner_id, status="running",
        ))
        with patch("src.features.generation.status_tracker.generation_repo"):
            self.tracker.create(self.generation.id, preset_id="preset-1", user_id=owner_id)

    def _run_with_a_subscribed_owner(self):
        socket = _Socket()

        async def run():
            await self.hub.connect(socket, "owner-client")
            await self.hub.subscribe_to_generation("owner-client", self.generation.id)
            await self.broadcaster.handle_output(self.generation.id, ModelsGenerationOutput(
                models=[ModelGenerationOutput(name=MODEL_PATH, type="checkpoint")],
            ))
            await self.broadcaster.handle_output(self.generation.id, DiffTextGenerationOutput(
                index=0, name="Positive Prompt", diff="a cat in the rain",
            ))
            await self.broadcaster.handle_output(self.generation.id, None)

        asyncio.run(run())
        return socket

    def _client(self, user):
        container = SimpleNamespace(_generation_controller=self.controller, media_access=Mock())
        app = FastAPI()
        app.include_router(build_router(container))
        app.include_router(build_admin_router(container))
        app.dependency_overrides[get_current_active_user] = lambda: user
        return TestClient(app)

    def test_the_owner_socket_gets_no_artifact_and_no_server_path(self):
        socket = self._run_with_a_subscribed_owner()

        assert [message["type"] for message in socket.sent] == ["generation_complete"]
        assert MODEL_PATH not in json.dumps(socket.sent)
        assert "a cat in the rain" not in json.dumps(socket.sent)

    def test_the_owner_cannot_read_the_run_report_over_http(self):
        self._run_with_a_subscribed_owner()
        client = self._client(self.owner)

        user_route = client.get(f"/api/generations/{self.generation.id}/run-report")
        admin_route = client.get(f"/api/admin/generations/{self.generation.id}")

        assert user_route.status_code == 404
        assert admin_route.status_code == 403
        assert MODEL_PATH not in user_route.text + admin_route.text

    def test_an_admin_reads_the_artifacts_in_admin_generations(self):
        self._run_with_a_subscribed_owner()
        client = self._client(self.admin)

        response = client.get(f"/api/admin/generations/{self.generation.id}")

        assert response.status_code == 200
        artifacts = response.json()["data"]["run_report"]["artifacts"]
        assert [artifact["artifact_type"] for artifact in artifacts] == ["models", "diff_text"]
        assert artifacts[0]["artifact_data"]["models"][0]["name"] == MODEL_PATH
