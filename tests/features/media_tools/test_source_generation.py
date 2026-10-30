from src.features.media_tools.publisher import to_item
from tests.features.media_tools.support import MediaToolsTestBase
from src.platform.media_tools.ops import register_core_tools

SHRINK = [{"tool": "shrink", "params": {"size": 20}}]


class TestSourceGeneration(MediaToolsTestBase):

    def setUp(self):
        super().setUp()
        register_core_tools(self.registry)

    def generation_of(self, record):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT generation_id FROM generation_files WHERE file_id = ?", (record.id,))
            return cursor.fetchone()["generation_id"]

    def row(self, entry):
        return self.upload_repo.get_by_id(entry.id, self.user_id)

    def test_a_run_on_a_generation_file_records_the_generation(self):
        record, _ = self.generation_file()

        run = self.finish(SHRINK, [{"generation_file_id": record.id}])

        entry = run.items[0]
        self.assertEqual(self.row(entry).source_generation_id, self.generation_of(record))
        self.assertEqual(entry.source_generation_id, self.generation_of(record))

    def test_a_run_on_a_library_upload_with_no_generation_stays_null(self):
        item = self.image_item()

        run = self.finish(SHRINK, [{"item_id": item.id}])

        self.assertIsNone(self.row(run.items[0]).source_generation_id)
        self.assertIsNone(run.items[0].source_generation_id)

    def test_inputs_from_different_generations_leave_it_null(self):
        first, _ = self.generation_file()
        second, _ = self.generation_file()

        run = self.finish(SHRINK, [{"generation_file_id": first.id}, {"generation_file_id": second.id}])

        self.assertEqual(len(run.items), 2)
        for entry in run.items:
            self.assertIsNone(self.row(entry).source_generation_id)

    def test_inputs_from_one_generation_keep_it(self):
        first, _ = self.generation_file()
        second, _ = self.generation_file()
        target = self.generation_of(first)
        self.file_repo.associate_with_generation(target, second.id)
        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM generation_files WHERE file_id = ? AND generation_id != ?", (second.id, target))

        run = self.finish(SHRINK, [{"generation_file_id": first.id}, {"generation_file_id": second.id}])

        self.assertEqual({self.row(entry).source_generation_id for entry in run.items}, {target})

    def test_a_library_item_input_passes_its_own_generation_on(self):
        item = self.image_item()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE uploads SET source_generation_id = 'gen-original' WHERE id = ?", (item.id,))

        run = self.finish(SHRINK, [{"item_id": item.id}])

        self.assertEqual(run.items[0].source_generation_id, "gen-original")
        self.assertEqual(self.row(run.items[0]).source_generation_id, "gen-original")

    def test_replace_keeps_the_existing_value(self):
        item = self.image_item()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE uploads SET source_generation_id = 'gen-original' WHERE id = ?", (item.id,))

        run = self.finish(SHRINK, [{"item_id": item.id}], mode="replace")

        self.assertTrue(run.replaced)
        self.assertEqual(run.items[0].source_generation_id, "gen-original")
        self.assertEqual(self.row(item).source_generation_id, "gen-original")

    def test_replace_of_an_item_without_a_generation_stays_null(self):
        item = self.image_item()

        self.finish(SHRINK, [{"item_id": item.id}], mode="replace")

        self.assertIsNone(self.row(item).source_generation_id)

    def test_to_item_serialises_the_generation(self):
        item = self.image_item()
        self.assertIsNone(to_item(item).model_dump()["source_generation_id"])
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE uploads SET source_generation_id = 'g1' WHERE id = ?", (item.id,))
        self.assertEqual(to_item(self.row(item)).model_dump()["source_generation_id"], "g1")
