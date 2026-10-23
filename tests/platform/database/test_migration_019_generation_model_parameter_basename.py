"""019 rewrites leaked depot/backend paths out of already-recorded `model`
display parameters - see
`src/platform/database/migrations/019_generation_model_parameter_basename.py`
for what moves and what deliberately doesn't.
"""

import json
import unittest

from tests.platform.database._support import MigrationCase


class TestMigration019GenerationModelParameterBasename(MigrationCase):
    MIGRATION = "019_generation_model_parameter_basename"

    def pre_state(self):
        self._seed()


    def _seed(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generations (id, form_data) VALUES ('g1', '{}')"
            )
            cursor.executemany(
                "INSERT INTO generation_parameters "
                "(id, generation_id, parameter_name, parameter_value, parameter_index) "
                "VALUES (?, 'g1', ?, ?, ?)",
                [
                    (
                        "p_native",
                        "model",
                        json.dumps(
                            "models/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"
                        ),
                        0,
                    ),
                    (
                        "p_comfy",
                        "model",
                        json.dumps("QWEN/Qwen-Image-Lightning-8steps-V2.0.safetensors"),
                        1,
                    ),
                    ("p_bare", "model", json.dumps("clip_l.safetensors"), 2),
                    ("p_seed", "seed", json.dumps(12345), 0),
                    (
                        "p_other_param_path",
                        "upscale_by",
                        json.dumps("models/upscalers/4x_esrgan.safetensors"),
                        0,
                    ),
                ],
            )

    def _values(self):
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, parameter_value FROM generation_parameters"
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def test_rewrites_a_native_depot_path_to_its_basename(self):
        self.migration.up()

        self.assertEqual(
            self._values()["p_native"], "qwen_image_2512_fp8_e4m3fn.safetensors"
        )

    def test_rewrites_a_comfyui_ref_with_a_subdirectory_to_its_basename(self):
        self.migration.up()

        self.assertEqual(
            self._values()["p_comfy"], "Qwen-Image-Lightning-8steps-V2.0.safetensors"
        )

    def test_leaves_a_bare_filename_untouched(self):
        self.migration.up()

        self.assertEqual(self._values()["p_bare"], "clip_l.safetensors")

    def test_leaves_non_model_parameters_untouched(self):
        """Scoped to `parameter_name = 'model'` only - the forward-looking
        generalization to other display parameters lives in the handler
        fix, not in this historical-data migration."""
        self.migration.up()

        values = self._values()
        self.assertEqual(values["p_seed"], 12345)
        self.assertEqual(
            values["p_other_param_path"], "models/upscalers/4x_esrgan.safetensors"
        )

    def test_is_idempotent(self):
        self.migration.up()
        first = self._values()

        self.migration.up()

        self.assertEqual(self._values(), first)


if __name__ == "__main__":
    unittest.main()
