import unittest

from src.features.library import operations
from src.features.library.mappers import upload_to_item
from src.features.media.records import Upload
from tests.features.library.test_operations import LibraryTestBase


class TestUploadSourceGeneration(LibraryTestBase):

    def _row(self, **overrides):
        values = dict(user_id=self.user_id, filename="s.png", media_type="image", file_size=10)
        values.update(overrides)
        return self.upload_repo.create(Upload(**values))

    def test_create_round_trips_the_generation(self):
        row = self._row(source_generation_id="gen-1")

        self.assertEqual(row.source_generation_id, "gen-1")
        self.assertEqual(self.upload_repo.get_by_id(row.id, self.user_id).source_generation_id, "gen-1")

    def test_an_upload_with_no_generation_stays_null(self):
        row = self._row()

        self.assertIsNone(self.upload_repo.get_by_id(row.id, self.user_id).source_generation_id)

    def test_library_item_serialises_the_generation(self):
        item = upload_to_item(self._row(source_generation_id="gen-1"), [])

        self.assertEqual(item.source_generation_id, "gen-1")
        self.assertEqual(item.model_dump()["source_generation_id"], "gen-1")
        self.assertIsNone(upload_to_item(self._row(filename="n.png"), []).source_generation_id)

    def test_copy_from_generation_records_the_generation(self):
        generation_id, file_record, _ = self._generated_file()

        item = operations.copy_generation_file(self.collaborators, file_record.id, self.user_id)

        self.assertEqual(item.source_generation_id, generation_id)
        self.assertEqual(self.upload_repo.get_by_id(item.id, self.user_id).source_generation_id, generation_id)

    def test_copy_of_a_file_with_no_generation_link_stays_null(self):
        _, file_record, _ = self._generated_file()
        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM generation_files WHERE file_id = ?", (file_record.id,))

        item = operations.copy_generation_file(self.collaborators, file_record.id, self.user_id)

        self.assertIsNone(item.source_generation_id)


if __name__ == "__main__":
    unittest.main()
