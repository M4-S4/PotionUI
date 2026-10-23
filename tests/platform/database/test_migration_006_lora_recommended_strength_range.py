"""006 turns the built-in LoRA `strength` attribute from a slider into a range,
and every value already recorded for it from a scalar into the degenerate band
`[x, x]`. Both value layers carry those scalars - the shared `models.model_metadata`
blob and the per-user `user_model_attributes` overlay - so both have to be widened
or the picker reads one of them as "no recommendation".
"""

import json
import unittest

from tests.platform.database._support import MigrationCase


class TestMigration006LoraRecommendedStrengthRange(MigrationCase):
    MIGRATION = "006_lora_recommended_strength_range"

    def pre_state(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_attribute_definitions
                    (id, key, label, field_type, model_types, config, default_value,
                     description, per_user, admin_only, system, source)
                VALUES ('d1', 'strength', 'Strength', 'slider', '["lora"]',
                        ?, '1.0', 'Default strength', 0, 0, 1, 'core')
                """,
                (json.dumps({"min": 0, "max": 2, "step": 0.05}),),
            )
            cursor.execute(
                "INSERT INTO models (id, filename, model_type, model_metadata) VALUES (?, ?, ?, ?)",
                ("m1", "detail.safetensors", "lora", json.dumps({"strength": 0.8, "triggers": ["detail"]})),
            )
            cursor.execute(
                "INSERT INTO models (id, filename, model_type, model_metadata) VALUES (?, ?, ?, ?)",
                ("m2", "style.safetensors", "lora", json.dumps({"triggers": ["style"]})),
            )
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('u1', 'u1', 'u1@test.com', 'x')"
            )
            cursor.execute(
                "INSERT INTO user_model_attributes (user_id, model_id, key, value) VALUES (?, ?, ?, ?)",
                ("u1", "m1", "strength", json.dumps(0.65)),
            )

    def _definition(self):
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT field_type, label, config, default_value, description "
                "FROM model_attribute_definitions WHERE key = 'strength'"
            ).fetchone()

    def _shared_metadata(self, model_id):
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT model_metadata FROM models WHERE id = ?", (model_id,)).fetchone()
        return json.loads(row[0])

    def _overlay(self):
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_model_attributes "
                "WHERE user_id = 'u1' AND model_id = 'm1' AND key = 'strength'"
            ).fetchone()
        return json.loads(row[0])

    def test_definition_becomes_a_range_with_no_stand_in_default(self):
        self.migration.up()

        definition = self._definition()
        self.assertEqual(definition["field_type"], "range")
        self.assertEqual(definition["label"], "Recommended strength")
        self.assertIsNone(definition["default_value"])

    def test_seeded_config_widens_to_admit_inverted_loras(self):
        self.migration.up()

        self.assertEqual(json.loads(self._definition()["config"])["min"], -2)

    def test_admin_edited_config_is_left_alone(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE model_attribute_definitions SET config = ? WHERE key = 'strength'",
                (json.dumps({"min": 0, "max": 5, "step": 0.1}),),
            )

        self.migration.up()

        self.assertEqual(json.loads(self._definition()["config"]), {"min": 0, "max": 5, "step": 0.1})

    def test_shared_scalar_widens_to_a_1_to_1_band(self):
        self.migration.up()

        self.assertEqual(self._shared_metadata("m1")["strength"], [0.8, 0.8])

    def test_other_attributes_and_models_are_untouched(self):
        self.migration.up()

        self.assertEqual(self._shared_metadata("m1")["triggers"], ["detail"])
        self.assertEqual(self._shared_metadata("m2"), {"triggers": ["style"]})

    def test_user_overlay_scalar_widens_too(self):
        self.migration.up()

        self.assertEqual(self._overlay(), [0.65, 0.65])

    def test_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertEqual(self._shared_metadata("m1")["strength"], [0.8, 0.8])
        self.assertEqual(self._overlay(), [0.65, 0.65])
        self.assertEqual(self._definition()["field_type"], "range")

    def test_an_already_widened_band_is_left_as_it_is(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE models SET model_metadata = ? WHERE id = 'm1'",
                (json.dumps({"strength": [0.7, 1.0]}),),
            )

        self.migration.up()

        self.assertEqual(self._shared_metadata("m1")["strength"], [0.7, 1.0])


if __name__ == '__main__':
    unittest.main()
