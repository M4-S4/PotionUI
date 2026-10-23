from src.features.collections.repository import CollectionRepository
from src.features.generation.records import Generation
from src.features.generation.repository import GenerationRepository
from src.platform.database import get_cursor
from src.platform.util.ids import generate_ulid
from tests.fixtures.persistence_base import PersistenceTestBase

HISTORY = "history"


class TestDeleteKeepsCollectionsConsistent(PersistenceTestBase):
    def setUp(self):
        super().setUp()
        self.collections = CollectionRepository()
        self.generations = GenerationRepository()
        self.user_id = self.create_test_user()
        self.generation_ids = [self._make_generation() for _ in range(3)]
        self.first = self.collections.create("First", self.user_id, HISTORY)
        self.second = self.collections.create("Second", self.user_id, HISTORY)
        for collection in (self.first, self.second):
            self.collections.add_members(collection.id, self.generation_ids, self.user_id, HISTORY)

    def _make_generation(self) -> str:
        generation = Generation(
            id=generate_ulid(),
            preset_id="test_preset",
            form_data={"prompt": "test prompt"},
            user_id=self.user_id,
            status="completed",
            preset_version="1.0",
        )
        self.generations.create(generation)
        return generation.id

    def _counts(self):
        return {c.name: c.item_count for c in self.collections.list(self.user_id, HISTORY)}

    def _membership_rows(self, generation_ids):
        placeholders = ",".join("?" for _ in generation_ids)
        with get_cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM collection_generations WHERE generation_id IN ({placeholders})",
                generation_ids,
            )
            return cursor.fetchone()[0]

    def test_single_delete_drops_every_containing_collection_by_one(self):
        self.assertEqual(self._counts(), {"First": 3, "Second": 3})
        self.generations.delete(self.generation_ids[0])
        self.assertEqual(self._counts(), {"First": 2, "Second": 2})
        self.assertEqual(self._membership_rows([self.generation_ids[0]]), 0)
        self.assertEqual(self.collections.smart_counts(self.user_id, HISTORY)["all"], 2)

    def test_repeated_deletes_leave_no_orphan_memberships(self):
        removed = self.generation_ids[:2]
        for generation_id in removed:
            self.generations.delete(generation_id)
        self.assertEqual(self._counts(), {"First": 1, "Second": 1})
        self.assertEqual(self._membership_rows(removed), 0)
        self.assertEqual(self.collections.smart_counts(self.user_id, HISTORY)["all"], 1)
