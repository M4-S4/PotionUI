import json

from tests.fixtures.persistence_base import PersistenceTestBase
from src.features.generation.file_repository import FileRepository
from src.features.generation.history_query import GenerationHistoryQuery
from src.features.generation.prompt_batch import PromptBatch, prompt_state_for
from src.features.generation.prompt_batch_split_repository import PromptBatchSplitRepository
from src.features.generation.records import File, Generation
from src.features.generation.repository import generation_repo

PER_PROMPT = 2
COUNT = 3
BASE_SEED = 1000


def _rows(cursor, sql, *params):
    cursor.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]


class SplitFixture(PersistenceTestBase):
    def setUp(self):
        super().setUp()
        self.user_id = self.create_test_user()
        self.files = FileRepository()
        self.generation_id = "orig"
        generation_repo.create(Generation(
            id=self.generation_id,
            preset_id="p",
            form_data={"quantity": PER_PROMPT, "seed": -1, "steps": 20},
            user_id=self.user_id,
            status="completed",
            prompt_state={"shared": "x", "prompts": [{"text": "cat"}, {"text": "dog"}, {"text": "owl"}]},
            backend_id=None,
            tab_id="tab-1",
            idempotency_key="key-1",
        ))
        self.batch = PromptBatch(per_prompt=PER_PROMPT, base_seed=BASE_SEED)
        self.batch.pairs = [{"positive": f"p{i}", "negative": ""} for i in range(PER_PROMPT * COUNT)]
        self.paths = {}
        for slot in range(PER_PROMPT * COUNT):
            path = f"outputs/2026-10-12/orig/{slot}.png"
            created = self.files.create(File(
                file_path=path, file_type="IMAGE", user_id=self.user_id, file_size=1, is_final=True,
            ))
            self.files.associate_with_generation(self.generation_id, created.id)
            self.paths[slot] = (path, created.id)
            self.batch.files[path] = slot // PER_PROMPT
        self.artifact_id = self.files.create(File(
            file_path="outputs/2026-10-12/orig/guide.png", file_type="IMAGE", user_id=self.user_id,
            file_size=1, is_final=False,
        )).id
        self.files.associate_with_generation(self.generation_id, self.artifact_id)

    def split(self):
        return PromptBatchSplitRepository().split(self.generation_id, self.batch)

    def file_ids_of(self, generation_id):
        with self.db.get_cursor() as cursor:
            rows = _rows(cursor, "SELECT file_id FROM generation_files WHERE generation_id = ?", generation_id)
        return {row["file_id"] for row in rows}


class TestSplitRows(SplitFixture):
    def test_one_new_row_per_later_prompt_and_the_original_stays_first(self):
        new_ids = self.split()

        self.assertEqual(len(new_ids), COUNT - 1)
        self.assertNotIn(self.generation_id, new_ids)
        with self.db.get_cursor() as cursor:
            self.assertEqual(len(_rows(cursor, "SELECT id FROM generations")), COUNT)

    def test_every_row_has_its_own_quantity_seed_and_prompt_state(self):
        new_ids = self.split()

        for index, generation_id in enumerate([self.generation_id, *new_ids]):
            row = generation_repo.get_by_id(generation_id)
            self.assertEqual(row.form_data["quantity"], PER_PROMPT)
            self.assertEqual(row.form_data["seed"], BASE_SEED + index * PER_PROMPT)
            self.assertEqual(row.form_data["steps"], 20)
            self.assertEqual(row.prompt_state, {"shared": "x", "text": ["cat", "dog", "owl"][index]})

    def test_split_rows_mirror_the_original_run(self):
        new_ids = self.split()

        original = generation_repo.get_by_id(self.generation_id)
        for generation_id in new_ids:
            row = generation_repo.get_by_id(generation_id)
            self.assertEqual(row.status, "completed")
            self.assertEqual(row.user_id, self.user_id)
            self.assertEqual(row.preset_id, original.preset_id)
            self.assertEqual(row.tab_id, "tab-1")
            self.assertIsNone(row.idempotency_key)
            self.assertEqual(row.created_at, original.created_at)

    def test_files_move_to_the_row_of_their_prompt(self):
        new_ids = self.split()

        ids_by_slot = {slot: file_id for slot, (_, file_id) in self.paths.items()}
        self.assertEqual(
            self.file_ids_of(self.generation_id) - {self.artifact_id},
            {ids_by_slot[0], ids_by_slot[1]},
        )
        self.assertEqual(self.file_ids_of(new_ids[0]), {ids_by_slot[2], ids_by_slot[3]})
        self.assertEqual(self.file_ids_of(new_ids[1]), {ids_by_slot[4], ids_by_slot[5]})

    def test_files_keep_their_ids_and_paths(self):
        self.split()

        for path, file_id in self.paths.values():
            self.assertEqual(self.files.get_by_id(file_id).file_path, path)

    def test_non_final_files_stay_with_the_original(self):
        self.split()

        self.assertIn(self.artifact_id, self.file_ids_of(self.generation_id))

    def test_a_split_row_lists_its_files_through_history(self):
        new_ids = self.split()

        listed = generation_repo.get_by_id(new_ids[0], include_files=True)
        self.assertEqual(len(listed.files), PER_PROMPT)

    def test_nothing_happens_when_only_the_first_prompt_produced_files(self):
        self.batch.files = {path: 0 for path, _ in self.paths.values()}

        self.assertEqual(self.split(), [])
        with self.db.get_cursor() as cursor:
            self.assertEqual(len(_rows(cursor, "SELECT id FROM generations")), 1)

    def test_only_prompts_that_produced_files_get_a_row(self):
        self.batch.files = {
            path: index for path, index in self.batch.files.items() if index != 1
        }

        new_ids = self.split()

        self.assertEqual(len(new_ids), 1)
        self.assertEqual(self.file_ids_of(new_ids[0]), {self.paths[4][1], self.paths[5][1]})
        row = generation_repo.get_by_id(new_ids[0])
        self.assertEqual(row.form_data["seed"], BASE_SEED + 2 * PER_PROMPT)
        self.assertEqual(row.prompt_state["text"], "owl")

    def test_a_cancelled_run_splits_what_exists(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE generations SET status = 'cancelled' WHERE id = ?", (self.generation_id,))
        self.batch.files = {self.paths[slot][0]: slot // PER_PROMPT for slot in (0, 1, 2)}

        new_ids = self.split()

        self.assertEqual(len(new_ids), 1)
        self.assertEqual(generation_repo.get_by_id(new_ids[0]).status, "cancelled")
        self.assertEqual(self.file_ids_of(new_ids[0]), {self.paths[2][1]})

    def test_a_run_without_a_pinned_seed_is_left_alone(self):
        self.batch.base_seed = None

        self.assertEqual(self.split(), [])


class TestBatchLookup(SplitFixture):
    def setUp(self):
        super().setUp()
        self.query = GenerationHistoryQuery(generation_repo)

    def test_every_row_of_the_batch_carries_the_batch_id(self):
        new_ids = self.split()

        for generation_id in [self.generation_id, *new_ids]:
            self.assertEqual(generation_repo.get_by_id(generation_id).batch_id, self.generation_id)

    def test_an_unsplit_generation_has_no_batch_id(self):
        self.assertIsNone(generation_repo.get_by_id(self.generation_id).batch_id)

    def test_include_batch_returns_every_file_ordered_by_prompt(self):
        new_ids = self.split()

        data = self.query.get_by_id(self.generation_id, self.user_id, include_batch=True)

        ids_by_slot = [file_id for _, file_id in self.paths.values()]
        listed = [entry for entry in data["files"] if entry["id"] in ids_by_slot]
        self.assertEqual([entry["id"] for entry in listed], ids_by_slot)
        self.assertEqual([entry["prompt_index"] for entry in listed], [0, 0, 1, 1, 2, 2])
        self.assertEqual(
            [entry["generation_id"] for entry in listed],
            [self.generation_id] * 2 + [new_ids[0]] * 2 + [new_ids[1]] * 2,
        )

    def test_each_file_names_the_prompt_of_its_row(self):
        self.split()

        data = self.query.get_by_id(self.generation_id, self.user_id, include_batch=True)

        ids_by_slot = [file_id for _, file_id in self.paths.values()]
        listed = [entry for entry in data["files"] if entry["id"] in ids_by_slot]
        self.assertEqual([entry["prompt"] for entry in listed], ["cat", "cat", "dog", "dog", "owl", "owl"])

    def test_the_prompt_falls_back_to_the_rows_segments(self):
        with self.db.get_cursor() as cursor:
            for prompt_index, text in enumerate(("red", "green", "blue")):
                cursor.execute(
                    "INSERT INTO generation_segments (id, generation_id, channel, prompt_index, segment_index, "
                    "segment_type, text) VALUES (?, ?, 'positive', ?, 0, 'content', ?)",
                    (f"s{prompt_index}", self.generation_id, prompt_index, text),
                )
            cursor.execute("UPDATE generations SET prompt_state = NULL WHERE id = ?", (self.generation_id,))
        self.split()

        data = self.query.get_by_id(self.generation_id, self.user_id, include_batch=True)

        ids_by_slot = [file_id for _, file_id in self.paths.values()]
        listed = [entry for entry in data["files"] if entry["id"] in ids_by_slot]
        self.assertEqual([entry["prompt"] for entry in listed], ["red", "red", "green", "green", "blue", "blue"])

    def test_asking_through_a_split_row_returns_the_same_batch(self):
        new_ids = self.split()

        data = self.query.get_by_id(new_ids[1], self.user_id, include_batch=True)

        self.assertEqual(sum(1 for entry in data["files"] if entry["prompt_index"] == 2), PER_PROMPT)
        self.assertEqual(len([e for e in data["files"] if e["id"] in {i for _, i in self.paths.values()}]), 6)

    def test_without_the_flag_only_the_rows_own_files_come_back(self):
        self.split()

        data = self.query.get_by_id(self.generation_id, self.user_id)

        self.assertEqual(len([e for e in data["files"] if e["id"] in {i for _, i in self.paths.values()}]), PER_PROMPT)
        self.assertTrue(all("prompt_index" not in entry for entry in data["files"]))

    def test_a_generation_outside_a_batch_ignores_the_flag(self):
        plain = self.query.get_by_id(self.generation_id, self.user_id)
        flagged = self.query.get_by_id(self.generation_id, self.user_id, include_batch=True)

        self.assertEqual(plain["files"], flagged["files"])

    def test_other_users_siblings_are_never_returned(self):
        new_ids = self.split()
        stranger = self.create_test_user('stranger', 'stranger', 'stranger@example.com')
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE generations SET user_id = ? WHERE id = ?", (stranger, new_ids[0]))

        data = self.query.get_by_id(self.generation_id, self.user_id, include_batch=True)

        self.assertNotIn(new_ids[0], {entry["generation_id"] for entry in data["files"]})

    def test_a_stranger_cannot_read_the_batch_at_all(self):
        from src.features.generation.exceptions import GenerationNotFoundException

        self.split()
        stranger = self.create_test_user('stranger', 'stranger', 'stranger@example.com')

        with self.assertRaises(GenerationNotFoundException):
            self.query.get_by_id(self.generation_id, stranger, include_batch=True)


class TestSplitIsSafe(SplitFixture):
    def test_a_failure_part_way_leaves_nothing_behind(self):
        from unittest.mock import patch

        with patch.object(PromptBatchSplitRepository, '_split_parameters', side_effect=RuntimeError('boom')):
            with self.assertRaises(RuntimeError):
                self.split()

        with self.db.get_cursor() as cursor:
            self.assertEqual(len(_rows(cursor, "SELECT id FROM generations")), 1)
        self.assertEqual(len(self.file_ids_of(self.generation_id)), PER_PROMPT * COUNT + 1)
        original = generation_repo.get_by_id(self.generation_id)
        self.assertEqual(original.form_data["seed"], -1)
        self.assertEqual(len(original.prompt_state["prompts"]), COUNT)

    def test_splitting_twice_creates_no_duplicates(self):
        first = self.split()
        second = self.split()

        self.assertEqual(len(first), COUNT - 1)
        self.assertEqual(second, [])
        with self.db.get_cursor() as cursor:
            self.assertEqual(len(_rows(cursor, "SELECT id FROM generations")), COUNT)
        self.assertEqual(len(self.file_ids_of(first[0])), PER_PROMPT)

    def test_a_derived_file_travels_with_its_prompt(self):
        derived = self.files.create(File(
            file_path="outputs/2026-10-12/orig/hires.png", file_type="IMAGE", user_id=self.user_id,
            file_size=1, is_final=True, is_derived=True,
        ))
        self.files.associate_with_generation(self.generation_id, derived.id)
        self.batch.files["outputs/2026-10-12/orig/hires.png"] = 2

        new_ids = self.split()

        self.assertIn(derived.id, self.file_ids_of(new_ids[1]))
        self.assertNotIn(derived.id, self.file_ids_of(self.generation_id))


class TestSplitLinkedTables(SplitFixture):
    def test_tags_models_and_collections_are_copied_to_every_row(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("INSERT INTO tags (id, name, user_id) VALUES ('t1', 'fox', ?)", (self.user_id,))
            cursor.execute("INSERT INTO generation_tags (generation_id, tag_id) VALUES (?, 't1')", (self.generation_id,))
        new_ids = self.split()

        with self.db.get_cursor() as cursor:
            for generation_id in new_ids:
                tags = _rows(cursor, "SELECT tag_id FROM generation_tags WHERE generation_id = ?", generation_id)
                self.assertEqual([row["tag_id"] for row in tags], ["t1"])

    def test_input_sources_are_copied_to_every_row(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generation_sources (id, generation_id, field_name, source_generation_id, "
                "source_file_index) VALUES ('src1', ?, 'image', ?, 3)",
                (self.generation_id, self.generation_id),
            )
        new_ids = self.split()

        with self.db.get_cursor() as cursor:
            for generation_id in new_ids:
                rows = _rows(
                    cursor,
                    "SELECT field_name, source_file_index FROM generation_sources WHERE generation_id = ?",
                    generation_id,
                )
                self.assertEqual(rows, [{"field_name": "image", "source_file_index": 3}])

    def test_segments_follow_their_prompt_and_are_renumbered(self):
        with self.db.get_cursor() as cursor:
            for prompt_index in range(COUNT):
                cursor.execute(
                    "INSERT INTO generation_segments (id, generation_id, channel, prompt_index, segment_index, "
                    "segment_type, text) VALUES (?, ?, 'positive', ?, 0, 'content', ?)",
                    (f"seg{prompt_index}", self.generation_id, prompt_index, f"text{prompt_index}"),
                )
        new_ids = self.split()

        with self.db.get_cursor() as cursor:
            for index, generation_id in enumerate([self.generation_id, *new_ids]):
                segments = _rows(
                    cursor, "SELECT text, prompt_index FROM generation_segments WHERE generation_id = ?", generation_id,
                )
                self.assertEqual(segments, [{"text": f"text{index}", "prompt_index": 0}])

    def test_per_image_parameters_follow_their_images(self):
        with self.db.get_cursor() as cursor:
            for slot in range(PER_PROMPT * COUNT):
                cursor.execute(
                    "INSERT INTO generation_parameters (id, generation_id, parameter_name, parameter_value, "
                    "parameter_index) VALUES (?, ?, 'seed', ?, ?)",
                    (f"par{slot}", self.generation_id, json.dumps(BASE_SEED + slot), slot),
                )
        new_ids = self.split()

        with self.db.get_cursor() as cursor:
            for index, generation_id in enumerate([self.generation_id, *new_ids]):
                params = _rows(
                    cursor,
                    "SELECT parameter_value, parameter_index FROM generation_parameters "
                    "WHERE generation_id = ? ORDER BY parameter_index",
                    generation_id,
                )
                self.assertEqual(
                    [(json.loads(row["parameter_value"]), row["parameter_index"]) for row in params],
                    [(BASE_SEED + index * PER_PROMPT + local, local) for local in range(PER_PROMPT)],
                )

    def test_single_valued_parameters_are_copied_to_every_row(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generation_parameters (id, generation_id, parameter_name, parameter_value, "
                "parameter_index) VALUES ('one', ?, 'sampler', '\"euler\"', 0)",
                (self.generation_id,),
            )
        new_ids = self.split()

        with self.db.get_cursor() as cursor:
            for generation_id in new_ids:
                rows = _rows(
                    cursor, "SELECT parameter_value FROM generation_parameters WHERE generation_id = ?", generation_id,
                )
                self.assertEqual(len(rows), 1)

    def test_split_never_writes_limit_events(self):
        with self.db.get_cursor() as cursor:
            before = _rows(cursor, "SELECT id FROM limit_events")
        self.split()
        with self.db.get_cursor() as cursor:
            self.assertEqual(_rows(cursor, "SELECT id FROM limit_events"), before)

    def test_deleting_one_row_leaves_the_others_files(self):
        new_ids = self.split()

        generation_repo.delete(new_ids[0])

        self.assertEqual(len(self.file_ids_of(new_ids[1])), PER_PROMPT)
        self.assertEqual(len(self.file_ids_of(self.generation_id) - {self.artifact_id}), PER_PROMPT)


class TestFilesInGenerationDirectory(SplitFixture):
    def test_a_moved_file_is_still_found_under_the_original_directory(self):
        self.split()

        found = self.files.get_in_generation_directory(self.user_id, self.generation_id, "4.png")

        self.assertEqual(found.id, self.paths[4][1])

    def test_a_generation_without_files_falls_back_to_a_directory_match(self):
        generation_repo.create(Generation(id="bare", preset_id="p", form_data={}, user_id=self.user_id))
        self.files.create(File(
            file_path="outputs/2026-10-12/bare/solo.png", file_type="IMAGE", user_id=self.user_id,
            file_size=1, is_final=True,
        ))

        found = self.files.get_in_generation_directory(self.user_id, "bare", "solo.png")

        self.assertEqual(found.file_path, "outputs/2026-10-12/bare/solo.png")

    def test_the_directory_comes_from_a_file_the_generation_still_owns(self):
        generation_repo.create(Generation(id="named", preset_id="p", form_data={}, user_id=self.user_id))
        kept = self.files.create(File(
            file_path="outputs/2026-10-12/elsewhere/0.png", file_type="IMAGE", user_id=self.user_id,
            file_size=1, is_final=True,
        ))
        self.files.associate_with_generation("named", kept.id)
        moved = self.files.create(File(
            file_path="outputs/2026-10-12/elsewhere/1.png", file_type="IMAGE", user_id=self.user_id,
            file_size=1, is_final=True,
        ))

        found = self.files.get_in_generation_directory(self.user_id, "named", "1.png")

        self.assertEqual(found.id, moved.id)

    def test_a_missing_name_is_not_found(self):
        self.split()

        self.assertIsNone(self.files.get_in_generation_directory(self.user_id, self.generation_id, "nope.png"))

    def test_other_users_files_are_not_found(self):
        self.assertIsNone(self.files.get_in_generation_directory("someone-else", self.generation_id, "4.png"))


class TestPromptStateFor(SplitFixture):
    def test_without_a_matching_list_the_state_is_copied_whole(self):
        state = {"a": 1, "prompts": [{"x": 1}]}

        self.assertEqual(prompt_state_for(state, 1, 3), state)

    def test_none_stays_none(self):
        self.assertIsNone(prompt_state_for(None, 0, 3))
