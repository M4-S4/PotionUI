from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.features.tab_drafts import validation
from src.features.tab_drafts.manager import TabDraftManager
from src.features.tab_drafts.routes import TabDraftController, build_router
from src.platform.security.current_user import get_current_active_user
from src.platform.security.user import AccountType, User
from tests.features.tab_drafts.test_repository import TabDraftStoreTestBase, set_op, text


def user(user_id):
    return User(
        id=user_id, username=user_id, email=f"{user_id}@example.com",
        password_hash="h", account_type=AccountType.USER,
    )


def op_set(name, definition):
    return set_op(name, definition)


class RoutesTestBase(TabDraftStoreTestBase):

    def setUp(self):
        super().setUp()
        self.manager = TabDraftManager(self.repository)
        app = FastAPI()
        app.include_router(build_router(SimpleNamespace(tab_draft_controller=TabDraftController(self.manager))))
        self.current = ["u1"]
        app.dependency_overrides[get_current_active_user] = lambda: user(self.current[0])
        self.app = app
        self.client = TestClient(app)

    def as_user(self, user_id):
        self.current[0] = user_id


class TestRoutes(RoutesTestBase):

    def test_put_creates_and_get_returns_the_draft(self):
        put = self.client.put("/api/tab-drafts/t1", json={"doc": {"variables": {"a": text("x")}}})
        got = self.client.get("/api/tab-drafts/t1")

        self.assertEqual(put.status_code, 200)
        self.assertTrue(put.json()["success"])
        self.assertEqual(got.json()["data"]["doc"], {"variables": {"a": text("x")}})
        self.assertEqual(got.json()["data"]["rev"], 1)
        self.assertEqual(got.json()["data"]["tab_id"], "t1")

    def test_ops_returns_the_merged_doc_and_rev(self):
        self.client.put("/api/tab-drafts/t1", json={"doc": {"variables": {"a": text("x")}}})

        response = self.client.post(
            "/api/tab-drafts/t1/ops",
            json={"base_rev": 1, "ops": [op_set("b", text("y")), {"op": "remove", "name": "a"}]},
        )

        data = response.json()["data"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["rev"], 2)
        self.assertEqual(data["doc"], {"variables": {"b": text("y")}})

    def test_ops_with_a_stale_base_rev_still_applies(self):
        self.client.post("/api/tab-drafts/t1/ops", json={"ops": [op_set("a", text("1"))]})
        self.client.post("/api/tab-drafts/t1/ops", json={"ops": [op_set("b", text("2"))]})

        response = self.client.post(
            "/api/tab-drafts/t1/ops", json={"base_rev": 1, "ops": [op_set("c", text("3"))]}
        )

        self.assertEqual(set(response.json()["data"]["doc"]["variables"]), {"a", "b", "c"})

    def test_list_returns_summaries_only(self):
        self.client.put("/api/tab-drafts/t1", json={"doc": {"variables": {"a": text("x"), "b": text("y")}}})

        response = self.client.get("/api/tab-drafts")

        rows = response.json()["data"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"tab_id", "variable_count", "rev", "updated_at"})
        self.assertEqual((rows[0]["tab_id"], rows[0]["variable_count"], rows[0]["rev"]), ("t1", 2, 1))

    def test_delete_removes_the_draft(self):
        self.client.put("/api/tab-drafts/t1", json={"doc": {}})

        self.assertEqual(self.client.delete("/api/tab-drafts/t1").status_code, 200)
        self.assertIsNone(self.client.get("/api/tab-drafts/t1").json()["data"])


class TestAuthScoping(RoutesTestBase):

    def setUp(self):
        super().setUp()
        self.client.put("/api/tab-drafts/t1", json={"doc": {"variables": {"a": text("owner")}}})
        self.as_user("u2")

    def test_another_users_tab_reads_as_absent_and_is_not_deletable(self):
        got = self.client.get("/api/tab-drafts/t1")
        deleted = self.client.delete("/api/tab-drafts/t1")

        self.assertEqual(got.status_code, 200)
        self.assertTrue(got.json()["success"])
        self.assertIsNone(got.json()["data"])
        self.assertEqual(deleted.status_code, 404)
        self.assertEqual(deleted.json()["detail"]["error"], "tab_draft_not_found")
        self.as_user("u1")
        self.assertEqual(self.client.get("/api/tab-drafts/t1").json()["data"]["doc"], {"variables": {"a": text("owner")}})

    def test_an_unknown_tab_reads_as_an_empty_200(self):
        response = self.client.get("/api/tab-drafts/never-existed")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertIsNone(response.json()["data"])

    def test_foreign_and_missing_reads_do_not_differ(self):
        foreign = self.client.get("/api/tab-drafts/t1")
        missing = self.client.get("/api/tab-drafts/never-existed")

        self.assertEqual(foreign.status_code, missing.status_code)
        self.assertEqual(foreign.json(), missing.json())

    def test_listing_never_shows_another_users_drafts(self):
        self.assertEqual(self.client.get("/api/tab-drafts").json()["data"], [])

    def test_writes_by_another_user_land_in_their_own_row(self):
        self.client.post("/api/tab-drafts/t1/ops", json={"ops": [op_set("a", text("intruder"))]})
        self.client.put("/api/tab-drafts/t1", json={"doc": {"variables": {}}})
        self.client.delete("/api/tab-drafts/t1")

        self.as_user("u1")
        owner = self.client.get("/api/tab-drafts/t1").json()["data"]
        self.assertEqual(owner["rev"], 1)
        self.assertEqual(owner["doc"]["variables"]["a"], text("owner"))

    def test_without_the_auth_dependency_override_the_routes_are_not_public(self):
        bare = TestClient(self.app)
        self.app.dependency_overrides.clear()

        self.assertIn(bare.get("/api/tab-drafts").status_code, (401, 403))
        self.assertIn(bare.post("/api/tab-drafts/t1/ops", json={"ops": []}).status_code, (401, 403))


class TestErrorShapes(RoutesTestBase):

    def test_invalid_op_is_422_with_per_op_errors(self):
        response = self.client.post(
            "/api/tab-drafts/t1/ops",
            json={"ops": [op_set("ok", text("1")), op_set("bad name", text("2"))]},
        )

        detail = response.json()["detail"]
        self.assertEqual(response.status_code, 422)
        self.assertEqual(detail["error"], "tab_draft_invalid")
        self.assertEqual([o["index"] for o in detail["operations"]], [1])
        self.assertEqual(detail["operations"][0]["name"], "bad name")
        self.assertTrue(detail["operations"][0]["errors"])
        self.assertIsNone(self.client.get("/api/tab-drafts/t1").json()["data"])

    def test_invalid_replace_is_422(self):
        response = self.client.put("/api/tab-drafts/t1", json={"doc": {"variables": {"a": {"type": "x"}}}})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["error"], "tab_draft_invalid")

    def test_oversized_document_is_413(self):
        big = {"type": "choice", "options": ["w" * validation.MAX_OPTION_CHARS] * validation.MAX_OPTIONS, "mode": "shuffle"}

        response = self.client.put(
            "/api/tab-drafts/t1", json={"doc": {"variables": {f"v{i}": big for i in range(6)}}}
        )

        self.assertEqual(response.status_code, 413)
        detail = response.json()["detail"]
        self.assertEqual(detail["error"], "tab_draft_too_large")
        self.assertEqual(detail["limit"], validation.MAX_DOC_BYTES)

    def test_oversized_ops_batch_is_413(self):
        big = {"type": "choice", "options": ["w" * validation.MAX_OPTION_CHARS] * validation.MAX_OPTIONS, "mode": "shuffle"}

        response = self.client.post(
            "/api/tab-drafts/t1/ops", json={"ops": [op_set(f"v{i}", big) for i in range(6)]}
        )

        self.assertEqual(response.status_code, 413)

    def test_malformed_tab_id_is_422(self):
        response = self.client.get("/api/tab-drafts/has%20space")

        self.assertEqual(response.status_code, 422)

    def test_missing_body_fields_are_rejected_by_the_framework(self):
        self.assertEqual(self.client.put("/api/tab-drafts/t1", json={}).status_code, 422)
        self.assertEqual(self.client.post("/api/tab-drafts/t1/ops", json={}).status_code, 422)

    def test_a_new_draft_past_the_per_user_cap_is_409(self):
        with patch.object(validation, "MAX_DRAFTS_PER_USER", 1):
            self.client.put("/api/tab-drafts/t1", json={"doc": {}})

            created = self.client.put("/api/tab-drafts/t2", json={"doc": {}})
            via_ops = self.client.post("/api/tab-drafts/t3/ops", json={"ops": [op_set("a", text("1"))]})
            existing = self.client.put("/api/tab-drafts/t1", json={"doc": {}})

        self.assertEqual((created.status_code, via_ops.status_code, existing.status_code), (409, 409, 200))
        self.assertEqual(created.json()["detail"]["error"], "tab_draft_limit")
        self.assertEqual(created.json()["detail"]["limit"], 1)
