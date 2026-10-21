from tests.fixtures.persistence_base import PersistenceTestBase
from src.features.model_library.repository.user_model_meta_repository import UserModelMetaRepository
from src.features.models.records import Model, ModelInfo
from src.features.models.repository import ModelRepository


class TestSubstringSearchMatchesDisplayName(PersistenceTestBase):
    def setUp(self):
        super().setUp()
        self.repository = ModelRepository()
        self.user_id = self.create_test_user()
        self.other_user_id = self.create_test_user("other_user", "otheruser", "other@example.com")
        self.gouache = self._model("axiom_gouache_v2.safetensors")
        self.portrait = self._model("plain_file_a.safetensors")
        self.titled = self._model("plain_file_b.safetensors")
        self.other = self._model("unrelated.safetensors")
        self.slashed = self._model("back\\slash.safetensors")
        self.repository.create_provider(
            ModelInfo(model_id=self.titled.id, provider="civitai", name="Cinematic Grain Pack")
        )
        UserModelMetaRepository().set_custom_name(self.user_id, self.portrait.id, "My Portrait Look")

    def _model(self, filename: str) -> Model:
        return self.repository.create(Model(
            filename=filename,
            file_size=1,
            sha256=filename.ljust(64, "0")[:64],
            model_type="lora",
        ))

    def _names(self, search: str, **kwargs) -> set:
        found = self.repository.get_all(search=search, include_files=False, library_user_id=self.user_id, **kwargs)
        return {m.filename for m in found}

    def test_filename_still_matches(self):
        self.assertEqual(self._names("gouache"), {"axiom_gouache_v2.safetensors"})

    def test_custom_name_matches(self):
        self.assertEqual(self._names("portrait look"), {"plain_file_a.safetensors"})

    def test_provider_name_matches(self):
        self.assertEqual(self._names("grain pack"), {"plain_file_b.safetensors"})

    def test_custom_name_is_private_to_its_user(self):
        found = self.repository.get_all(
            search="portrait look", include_files=False, library_user_id=self.other_user_id
        )
        self.assertEqual(found, [])

    def test_percent_and_underscore_are_literal(self):
        self.assertEqual(self._names("%"), set())
        self.assertEqual(self._names("file_a"), {"plain_file_a.safetensors"})
        self.assertEqual(self._names("file-a"), set())

    def test_backslash_is_literal(self):
        self.assertEqual(self._names("back\\slash"), {"back\\slash.safetensors"})
        self.assertEqual(self._names("backslash"), set())

    def test_trailing_backslash_does_not_raise(self):
        self.assertEqual(self._names("slash\\"), set())
        self.assertEqual(self._names("back\\"), {"back\\slash.safetensors"})

    def test_count_total_uses_the_same_clause(self):
        for text, expected in (("portrait look", 1), ("grain pack", 1), ("plain_file", 2), ("zzz", 0)):
            self.assertEqual(self.repository.count_total(search=text, library_user_id=self.user_id), expected)

    def test_paging_counts_name_matches(self):
        first = self.repository.get_all(
            search="plain_file", include_files=False, library_user_id=self.user_id, limit=1, offset=0
        )
        second = self.repository.get_all(
            search="plain_file", include_files=False, library_user_id=self.user_id, limit=1, offset=1
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].id, second[0].id)

    def test_tag_filtered_search_matches_names(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("INSERT INTO tags (id, name) VALUES ('t1', 'style')")
            for model in (self.portrait, self.other):
                cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 't1')", (model.id,))
        found = self.repository.get_all(
            search="portrait look", tag_ids=["t1"], include_files=False, library_user_id=self.user_id
        )
        self.assertEqual({m.filename for m in found}, {"plain_file_a.safetensors"})
        self.assertEqual(
            self.repository.count_total(search="portrait look", tag_ids=["t1"], library_user_id=self.user_id), 1
        )
